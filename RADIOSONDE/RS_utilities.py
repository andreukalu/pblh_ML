import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import glob
import concurrent.futures
from scipy.optimize import curve_fit
from scipy import special
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from datetime import datetime
import netCDF4
import gc
import re
from scipy.interpolate import interp1d
import math


pd.set_option('display.max_colwidth', None)

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

########### PARAMETER NAMES CONVENTIONS ################
# -id:        Station ID
# -geopot:    Height/Geopotential height
# -pres:      Pressure
# -temp:      Temperature
# -rh:        Relative Humidity
# -hws:       Horizontal Wind Speed
# -lat:       Latitude
# -lon:       Longitude
# -time:      Datetime


########### ATMOSPHERIC METHODS ########################
# Compute the virtual potential temperature
def virtual_potential_temperature(temp, pres, rh):
    # Constants
    Rd = 287.05  # Specific gas constant for dry air in J/(kg·K)
    Rv = 461.51  # Specific gas constant for water vapor in J/(kg·K)
    Cp = 1004.0  # Specific heat capacity at constant pressure for dry air in J/(kg·K)
    epsilon = Rd / Rv  # Ratio of gas constants
    P0 = 1000 #Reference pressure in hPa
    
    temp_k = temp + 273.15

    #Compute saturation water vapor pressure with respect to liquid water at temperature temp_k: https://en.wikipedia.org/wiki/Dew_point
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))  # Saturation vapor pressure in hPa
    
    #Compute partial pressure of water vapor
    e = (rh / 100) * es

    #Compute mixing ratio
    w = 0.622 * e /(pres - e)
    
    #Calculate the potential temperature
    theta = temp_k * (P0/pres) ** (Rd / Cp)

    #Compute the virtual potential temperature
    theta_v = theta * (1 + 0.61 * (w / (1 + w) ))
    
    return theta_v

# Compute the relative humidity from temperature and dewpoint temperature
def relative_humidity(T, Td):
    e_s = 6.112 * math.exp((17.62 * T) / (243.12 + T))
    e_d = 6.112 * math.exp((17.62 * Td) / (243.12 + Td))
    RH = (e_d / e_s) * 100
    return RH

#Compute the bulk-Richardson number
def bulk_richardson_number(temp, pres, rh, hws, altitude):
    
    #Get virtual potential temperature
    vpt = virtual_potential_temperature(temp, pres, rh)

    i = 0
    while i < len(vpt):
        if np.isnan(vpt[i]):
            i += 1
            continue
        else:
            vpt0 = vpt[i]
            break
    
    #Compute bulk-Richardson Number
    Rib = (9.81 * altitude / vpt0) * ((vpt - vpt0) / (hws)**2)
    return Rib

#Compute PBLH using the parcel method for day scenarios (CBL)
def parcel_method_day(altitude,temp,pres,rh,plot_flag=0):

    #Get virtual potential temperature and its value at the lowest altitude
    vpt = virtual_potential_temperature(temp, pres, rh)
    vpt0 = vpt[0]

    #Get the altitudes in which vpt exceedes vpt at ground level
    aux = altitude[vpt>vpt0]
    
    if plot_flag == 1:
        plt.figure()
        plt.plot(vpt,altitude)
        plt.plot([vpt0,vpt0],[0,8000])

    #Check if this value exist and return the pblh
    if len(aux)==0:
        ablh = 0
    else:
        ablh = aux[0]
        
    return ablh

#Compute PBLH using the parcel method for night scenarios (SBL)
def parcel_method_night(altitude,temp,plot_flag=0):

    try:
        #Get temperatures lower than 1000 m height for SBL top detection
        tempsmooth = temp[altitude<1000]
        #Smooth the temperature profile
        tempsmooth = smooth(tempsmooth,10)

        #Get the maximum value of temperature profile to determine the beginning of the inversion at the PBL top
        ablh = altitude[np.argmax(tempsmooth)]
        
        if plot_flag == 1:
            plt.figure()
            plt.plot(tempsmooth,altitude[altitude<1000])
            plt.plot([min(tempsmooth),max(tempsmooth)],[aux,aux])
    except:
        ablh = 0
    return ablh

#Compute PBLH using the bulk-Richardson Number method
def get_ablh_Ri(altitude,temp,pres,rh,hws,plot_flag=0):

    #Compute the bulk-Richardson number profile
    bRi = bulk_richardson_number(temp,pres,rh,hws,altitude)

    #Define the bRI threshold
    th = 0.25

    #Interpolate the RS data every 30 m (emulate CALIPSO)
    min_alt = altitude.min()
    max_alt = 10000
    new_altitude = np.arange(min_alt, max_alt + 1, 30)
    
    # Identify finite values
    finite_mask = np.isfinite(bRi)
    
    # Filter out rows with infinite values
    altitude = altitude[finite_mask]
    bRi = bRi[finite_mask]
    
    # Create the interpolation function
    interp_func = interp1d(altitude, bRi, kind='linear', fill_value="extrapolate")
    
    # Interpolate rBi values for the new altitude array
    bRi = interp_func(new_altitude)
    
    #Get the altitudes at which bRI exceeds the threshold
    aux = new_altitude[bRi>th]

    #Check if value empty
    if len(aux)==0:
        ablh = 0
    else:
        if (aux[0] == altitude[1]):
            ablh = 0
        else:
            ablh = aux[0]  
    
    if plot_flag:
        plt.figure()
        plt.plot(bRi[new_altitude<5000],new_altitude[new_altitude<5000])
        plt.plot([th,th],[ablh,ablh])
        plt.xlim(-1,1)
        
    return ablh

#Function to compute PBL top altitude using either Parcel method or the bulk-Richardson number method
def get_ablh(df,method='Ri',plot_flag = 0,get_clouds=0):
    try:
        #LIMIT ALTITUDE TO 8000
        df_aux = df[df['geopot']<8000]

        #Get RS data
        temp = df_aux['temp'].to_numpy() # Temperature in Celsius
        pres = df_aux['pres'].to_numpy() # Pressure in hpa
        rh = df_aux['rh'].to_numpy() # RH in %
        altitude = df_aux['geopot'].to_numpy() # Geopotential altitude in m
        hws = df_aux['hws'].to_numpy() # HWS in m/s

        altitude_res = np.max(np.diff(altitude[altitude<4000])) #Compute the resolution of altitude
        
        if plot_flag == 1:
            plt.figure()
            plt.plot(temp,altitude)
            plt.plot(rh,altitude)
            plt.plot(hws,altitude)

        #Get if it is day or night
        # night = int(datetime.strptime(str(df['time'].iloc[0]),'%Y%m%d%H').hour < 9)

        #Switch between PBLH estimation methods
        
        # PARCEL METHOD
        if method == 'parcel': 
            if night == 0:
                ablh = parcel_method_day(altitude,temp,pres,rh,plot_flag)
            else:
                ablh = parcel_method_night(altitude,temp,plot_flag)

        # BULK-RICHARDSON NUMBER METHOD
        elif method == 'Ri':
            ablh = get_ablh_Ri(altitude,temp,pres,rh,hws,plot_flag)
        
        df_out = df[0:1]    
        df_out['ablh_rs'] = ablh
        df_out['min_altitude_rs'] = altitude[0]
        df_out['altitude_res'] = altitude_res
        df_out = df_out.drop(columns=['temp','pres','geopot','rh','hws'])
    except Exception as e: 
        # print(e)
        df_out = pd.DataFrame()
    return df_out

#Function to compute PBL top altitude using either Parcel method or the bulk-Richardson number method
def get_rh(df):
    try:
        #LIMIT ALTITUDE TO 8000
        df_aux = df[df['geopot']<8000]
    
        #Get RS data
        rh = df_aux['rh'].to_numpy() # RH in %
        altitude = df_aux['geopot'].to_numpy() # Geopotential altitude in m
    
        altitude_res = np.max(np.diff(altitude[altitude<4000])) #Compute the resolution of altitude
    
        #Interpolate the RS data every 30 m (emulate CALIPSO)
        min_alt = altitude.min()
        max_alt = 3000
        out_altitude = np.zeros((101,1))
        new_altitude = np.arange(round(min_alt/30)*30, max_alt + 1, 30).reshape(-1, 1)
        out_altitude[round(min_alt/30):] = new_altitude
        
        # Identify finite values
        finite_mask = np.isfinite(rh)
        
        # Filter out rows with infinite values
        altitude = altitude[finite_mask]
        rhi = rh[finite_mask]
    
        out_rh = np.zeros((101,1))
        
        # Create the interpolation function
        interp_func = interp1d(altitude, rhi, kind='linear', fill_value="extrapolate")
        
        # Interpolate rBi values for the new altitude array
        rhi = interp_func(new_altitude)
        out_rh[round(min_alt/30):] = rhi.reshape(-1, 1)
    
        column_names = [f"RH{i+1}" for i in range(len(out_rh))]
        
        df_out = df[0:1]    
        df_out['min_altitude_rs'] = altitude[0]
        df_out['altitude_res'] = altitude_res
        
        df_out = pd.concat([df_out, pd.DataFrame(out_rh.T, columns=column_names, index=df_out.index)], axis=1)
        df_out = df_out.drop(columns=['temp','pres','geopot','rh','hws'])
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

############ UTILITIES FUNCTIONS #######################
# Function to retrieve the BLH top altitude from Radiosonde profiles
def get_ablh_group(group,method='Ri',plot_flag = 0):
    try:
        #Get data (df) and group (g) information
        df = group[1]
        g = group[0]

        #Generate output dataframe with ablh
        df_out = get_ablh(df,method,plot_flag)

        #Add date to output dataframe
        df_out['time'] = pd.to_datetime(g[0],format='%Y%m%d%H')
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

def get_rh_group(group,vert_res=300,plot_flag = 0):
    try:
        #Get data (df) and group (g) information
        df = group[1]
        g = group[0]

        #Generate output dataframe with ablh
        df_out = get_rh(df)

        #Add date to output dataframe
        df_out['time'] = pd.to_datetime(g[0],format='%Y%m%d%H')
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

def read_all_RS_data_rh(base_path,sources=['GERMANY','BCN','US','GRUAN','UW'],num_threads=256):

    df_global = pd.DataFrame()

    if 'GERMANY' in sources:
        name = 'produkt*.txt'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_GERMANY_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'BCN' in sources:
        name = '*.TXT'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_BCN_rh, filepath, 1) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

        name = 'DORS*'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_BCN_rh, filepath, 2) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'US' in sources:
        name = 'US*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'NOAA', 'DATA', name), recursive=True)
        print(os.path.join(base_path, 'NOAA', 'DATA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'CHINA' in sources:
        name = 'CH*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'NOAA', 'DATA', name), recursive=True)
        print(os.path.join(base_path, 'NOAA', 'DATA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'AFRICA' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'AFRICA', name), recursive=True)
        print(os.path.join(base_path, 'AFRICA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'SOUTHAMERICA' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'SOUTHAMERICA', name), recursive=True)
        print(os.path.join(base_path, 'SOUTHAMERICA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])
    
    if 'REMAINING' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'REMAINING', name), recursive=True)
        print(os.path.join(base_path, 'REMAINING', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'GRUAN' in sources:
        name = '*.nc'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_GRUAN_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'UW' in sources:
        name = 'sounding*.txt'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_UW_rh, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])
    return df_global

def read_all_RS_data(base_path,sources=['GERMANY','BCN','US','GRUAN','UW'],num_threads=256):
    
    df_global = pd.DataFrame()

    if 'GERMANY' in sources:
        name = 'produkt*.txt'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_GERMANY, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'BCN' in sources:
        name = '*.TXT'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_BCN, filepath, 1) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

        name = 'DORS*'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_BCN, filepath, 2) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'US' in sources:
        name = 'US*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'NOAA', 'DATA', name), recursive=True)
        print(os.path.join(base_path, 'NOAA', 'DATA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'CHINA' in sources:
        name = 'CH*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'NOAA', 'DATA', name), recursive=True)
        print(os.path.join(base_path, 'NOAA', 'DATA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'AFRICA' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'AFRICA', name), recursive=True)
        print(os.path.join(base_path, 'AFRICA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'SOUTHAMERICA' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'SOUTHAMERICA', name), recursive=True)
        print(os.path.join(base_path, 'SOUTHAMERICA', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])
    
    if 'REMAINING' in sources:
        name = '*.txt'
        filepaths = glob.glob(os.path.join(base_path, 'REMAINING', name), recursive=True)
        print(os.path.join(base_path, 'REMAINING', name))
        filepaths = [file.replace("\\", "/") for file in filepaths]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_US, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'GRUAN' in sources:
        name = '*.nc'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_GRUAN, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])

    if 'UW' in sources:
        name = 'sounding*.txt'
        filepaths = glob.glob(os.path.join(base_path, '**', name), recursive=True)
        filepaths = [file.replace("\\", "/") for file in filepaths]
        print(filepaths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit each file to the executor
            futures = [executor.submit(process_file_rs_UW, filepath) for filepath in filepaths]
            
            # Gather the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        df = pd.concat(results)
        df_global = pd.concat([df_global,df])
    return df_global

def process_ABLH_from_all_RS_data(df,num_threads=256):
    result_df = apply_multithreaded_groupby(df)
    return result_df

# Function to read Germany RS measurements files
def process_file_rs_GERMANY(path):
    print(path)
    df = pd.read_csv(path,sep=';',names=['id','date','point','qn','lat','lon','geopot','pres','temp','rh','td','hws','wd','eor'], header=1)
    df = df.drop(columns=['point','qn','td','wd','eor'])
    df = df.rename(columns={'date':'time'})

    df_out = apply_multithreaded_groupby(df)
    print(path, ' Finished')
    return df_out

# Function to read Barcelona RS measurement files
def process_file_rs_BCN(path,type):
    print(path)
    try:
        if type==1:
            datef = path[-12:-4]
            datef = datetime.strptime(datef, '%y%m%d%H')
        else:
            datef = path[-10:]
            datef = datetime.strptime(datef,'%H%Y%m%d')
        
        if datef <= datetime(2011,2,16):
            df = pd.read_csv(path,sep=r'\s+',names=['min','s','vel','geopot','pres','temp','rh','td','wd','hws','lon','lat'], header=1)
            df = df.drop(columns=['min','s','vel','td','wd'])
            df = df.assign(time=datef)
            df['time'] = pd.to_datetime(df['date'],format='%y%m%d%H')
        else:
            df = pd.read_csv(path,sep=r'\s+',names=['t','geopot','pres','temp','rh','td','hws','wd','vef','vnf','lat','lon'], header=1)
            df = df.drop(columns=['vef','vnf','t','td','wd','vef','vnf'])
            df = df.assign(time=datef)
        df['id'] = 'BCN'
        df_out = apply_multithreaded_groupby(df,num_threads=8)
        return df_out
    except: 
        return

# Function to read NOAA RS measurement files
def process_file_rs_US(path):
    print(path)
    # Initialize lists to store parsed data
    headers = []
    data = []

    try:
        # Read the file line by line
        with open(path, 'r') as file:
            lines = file.readlines()

            # Iterate through lines
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('#'):
                    # Parse header line
                    header = line.strip()
                    headrec = header[0]
                    id = header[1:12]
                    try:
                        date = datetime.strptime(header[13:26], "%Y %m %d %H")
                    except:
                        date = datetime.strptime(header[13:23], "%Y %m %d")
                        
                    reltime = int(header[27:31])
                    numlev = int(header[32:36])
                    p_src = header[37:45]
                    np_src = header[56:54]
                    lat = float(header[55:62])/10000
                    lon = float(header[63:71])/10000

                    if date.year<2010:
                        i = i + numlev + 1
                    else:
                        i += 1
                else:
                    # Parse data lines until the next header or end of file
                    while i < len(lines) and not lines[i].startswith('#'):
                        data_line = lines[i].strip()
                        LVLTYP1 = data_line[0]
                        LVLTYP2 = data_line[1]
                        ETIME = data_line[3:8]
                        PRESS = float(data_line[9:15])/100
                        PFLAG = data_line[15]
                        GPH = int(data_line[16:21])
                        ZFLAG = data_line[21]
                        TEMP = float(data_line[22:27])/10
                        TFLAG = data_line[27]
                        RH = float(data_line[28:33])/10
                        DPDP = float(data_line[34:39])/10
                        WDIR = int(data_line[40:45])
                        WSPD = float(data_line[46:51])/10
                        data.append([id,date,reltime,numlev,p_src,np_src,lat,lon,LVLTYP1,LVLTYP2,ETIME,PRESS,PFLAG,GPH,ZFLAG,TEMP,TFLAG,RH,DPDP,WDIR,WSPD])
                        i += 1
        column_names = ['id', 'time', 'RELTIME', 'NUMLEV', 'P_SRC', 'NP_SRC', 'lat', 'lon',\
                        'LVLTYP1', 'LVLTYP2', 'ETIME', 'pres', 'PFLAG', 'geopot', 'ZFLAG', 'temp', 'TFLAG', 'rh', 'td', 'wd', 'hws']
        df = pd.DataFrame(data,columns=column_names)
        df = df.drop(columns=['RELTIME','NUMLEV','P_SRC','NP_SRC','LVLTYP1','LVLTYP2','ETIME','PFLAG','ZFLAG','TFLAG','wd'])
        df['rh'] = df.apply(lambda x: relative_humidity(x['temp'], x['td']), axis=1)
        df = df[~df.isin([-999.9, -9999]).any(axis=1)]
        df_out = apply_multithreaded_groupby(df)
        #FREE MEMORY
        del df
        # Force garbage collection
        gc.collect()
    except Exception as e:
        print(e)
        df_out = pd.DataFrame() 

    return df_out

# Function to read GRUAN RS measurement files
def process_file_rs_GRUAN(path):
    try:
        print(path)
        # Open the NetCDF file
        dataset = netCDF4.Dataset(path, 'r')
        
        data = dict()
        data['lat'] = dataset.variables['lat'][:]
        data['lon'] = dataset.variables['lon'][:]
        data['temp'] = dataset.variables['temp'][:] - 273.15
        data['rh'] = dataset.variables['rh'][:]
        data['hws'] = dataset.variables['wspeed'][:]
        data['geopot'] = dataset.variables['alt'][:]
        data['pres'] = dataset.variables['press'][:]
        
        df = pd.DataFrame.from_dict(data)
        timestamp = pd.to_datetime(dataset.getncattr('g.General.Timestamp'))
        df['time'] = timestamp
        df['id'] = dataset.__dict__['g.MeasuringSystem.ID']
        df_out = apply_multithreaded_groupby(df)
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

# Function to read University of Wyoming RS measurement files
def process_file_rs_UW(path):
    try:
        print(path)
        # Open the NetCDF file
        with open(path, 'r') as file:
            lines = file.readlines()
        
            i = 0
            
            flag = 0
            read_data = 0
            header_data = 0
        
            data_columns = ['pres', 'geopot', 'temp', 'rh', 'hws']
            columns = ['id','time','lat','lon']
            columns = columns + data_columns
            df = pd.DataFrame(columns=columns)
            while i < len(lines):
                line = lines[i].strip()
        
                if (line.startswith('----------')) & (flag == 0):
                    read_data = 0
                    flag = 1
                    i += 1
                    continue
                    
                if (line.startswith('----------')) & (flag == 1):
                    read_data = 1
                    flag = 0
                    i += 1
                    continue  
        
                if read_data == 1:
                    # Parse data lines until the next header or end of file
                    data_aux = []
                    while i < len(lines) and not lines[i].startswith('</PRE>'):
                        data_line = lines[i]
                        pres = to_float(data_line[0:7])
                        geopot = to_float(data_line[8:14])
                        temp = to_float(data_line[15:20])
                        rh = to_float(data_line[32:35])
                        hws = to_float(data_line[53:56])*0.51444 
                        data_aux.append([pres,geopot,temp,rh,hws])
                        i += 1
                        
                    df_aux = pd.DataFrame(data=data_aux,columns = data_columns)
                    read_data = 0
                    header_data = 1
                                    
                if (line.startswith('Station number')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['id'] = to_float(data_line[15:19])
                    i += 1
                    continue
    
                if (line.startswith('Observation time')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['time'] = data_line[18:27]
                    i += 1
                    continue    
                
                if (line.startswith('Station latitude')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['lat'] = to_float(data_line[18:24])
                    i += 1
                    continue
        
                
                if (line.startswith('Station longitude')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['lon'] = to_float(data_line[19:25])
                    df = pd.concat([df,df_aux])
                    i += 1
                    header_data = 0
                    continue
        
                i += 1
        df['time'] = pd.to_datetime(df['time'],format='%y%m%d/%H')        
        df = df.reset_index(drop=True)
        df_out = apply_multithreaded_groupby(df)
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

# Function to read Germany RS measurements files
def process_file_rs_GERMANY_rh(path):
    print(path)
    df = pd.read_csv(path,sep=';',names=['id','date','point','qn','lat','lon','geopot','pres','temp','rh','td','hws','wd','eor'], header=1)
    df = df.drop(columns=['point','qn','td','wd','eor'])
    df = df.rename(columns={'date':'time'})

    df_out = apply_multithreaded_groupby(df,func=get_rh_group)
    print(path, ' Finished')
    return df_out

# Function to read Barcelona RS measurement files
def process_file_rs_BCN_rh(path,type):
    print(path)
    try:
        if type==1:
            datef = path[-12:-4]
            datef = datetime.strptime(datef, '%y%m%d%H')
        else:
            datef = path[-10:]
            datef = datetime.strptime(datef,'%H%Y%m%d')
        
        if datef <= datetime(2011,2,16):
            df = pd.read_csv(path,sep=r'\s+',names=['min','s','vel','geopot','pres','temp','rh','td','wd','hws','lon','lat'], header=1)
            df = df.drop(columns=['min','s','vel','td','wd'])
            df = df.assign(time=datef)
            df['time'] = pd.to_datetime(df['date'],format='%y%m%d%H')
        else:
            df = pd.read_csv(path,sep=r'\s+',names=['t','geopot','pres','temp','rh','td','hws','wd','vef','vnf','lat','lon'], header=1)
            df = df.drop(columns=['vef','vnf','t','td','wd','vef','vnf'])
            df = df.assign(time=datef)
        df['id'] = 'BCN'
        df_out = apply_multithreaded_groupby(df,num_threads=8,func=get_rh_group)
        return df_out
    except: 
        return

# Function to read NOAA RS measurement files
def process_file_rs_US_rh(path):
    print(path)
    # Initialize lists to store parsed data
    headers = []
    data = []

    try:
        # Read the file line by line
        with open(path, 'r') as file:
            lines = file.readlines()

            # Iterate through lines
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('#'):
                    # Parse header line
                    header = line.strip()
                    headrec = header[0]
                    id = header[1:12]
                    try:
                        date = datetime.strptime(header[13:26], "%Y %m %d %H")
                    except:
                        date = datetime.strptime(header[13:23], "%Y %m %d")
                        
                    reltime = int(header[27:31])
                    numlev = int(header[32:36])
                    p_src = header[37:45]
                    np_src = header[56:54]
                    lat = float(header[55:62])/10000
                    lon = float(header[63:71])/10000

                    if date.year<2010:
                        i = i + numlev + 1
                    else:
                        i += 1
                else:
                    # Parse data lines until the next header or end of file
                    while i < len(lines) and not lines[i].startswith('#'):
                        data_line = lines[i].strip()
                        LVLTYP1 = data_line[0]
                        LVLTYP2 = data_line[1]
                        ETIME = data_line[3:8]
                        PRESS = float(data_line[9:15])/100
                        PFLAG = data_line[15]
                        GPH = int(data_line[16:21])
                        ZFLAG = data_line[21]
                        TEMP = float(data_line[22:27])/10
                        TFLAG = data_line[27]
                        RH = float(data_line[28:33])/10
                        DPDP = float(data_line[34:39])/10
                        WDIR = int(data_line[40:45])
                        WSPD = float(data_line[46:51])/10
                        data.append([id,date,reltime,numlev,p_src,np_src,lat,lon,LVLTYP1,LVLTYP2,ETIME,PRESS,PFLAG,GPH,ZFLAG,TEMP,TFLAG,RH,DPDP,WDIR,WSPD])
                        i += 1
        column_names = ['id', 'time', 'RELTIME', 'NUMLEV', 'P_SRC', 'NP_SRC', 'lat', 'lon',\
                        'LVLTYP1', 'LVLTYP2', 'ETIME', 'pres', 'PFLAG', 'geopot', 'ZFLAG', 'temp', 'TFLAG', 'rh', 'td', 'wd', 'hws']
        df = pd.DataFrame(data,columns=column_names)
        df = df.drop(columns=['RELTIME','NUMLEV','P_SRC','NP_SRC','LVLTYP1','LVLTYP2','ETIME','PFLAG','ZFLAG','TFLAG','wd'])
        df['rh'] = df.apply(lambda x: relative_humidity(x['temp'], x['td']), axis=1)
        df = df[~df.isin([-999.9, -9999]).any(axis=1)]
        df_out = apply_multithreaded_groupby(df,func=get_rh_group)
        #FREE MEMORY
        del df
        # Force garbage collection
        gc.collect()
    except Exception as e:
        print(e)
        df_out = pd.DataFrame() 

    return df_out

# Function to read GRUAN RS measurement files
def process_file_rs_GRUAN_rh(path):
    try:
        print(path)
        # Open the NetCDF file
        dataset = netCDF4.Dataset(path, 'r')
        
        data = dict()
        data['lat'] = dataset.variables['lat'][:]
        data['lon'] = dataset.variables['lon'][:]
        data['temp'] = dataset.variables['temp'][:] - 273.15
        data['rh'] = dataset.variables['rh'][:]
        data['hws'] = dataset.variables['wspeed'][:]
        data['geopot'] = dataset.variables['alt'][:]
        data['pres'] = dataset.variables['press'][:]
        
        df = pd.DataFrame.from_dict(data)
        timestamp = pd.to_datetime(dataset.getncattr('g.General.Timestamp'))
        df['time'] = timestamp
        df['id'] = dataset.__dict__['g.MeasuringSystem.ID']
        df_out = apply_multithreaded_groupby(df,func=get_rh_group)
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

# Function to read University of Wyoming RS measurement files
def process_file_rs_UW_rh(path):
    try:
        print(path)
        # Open the NetCDF file
        with open(path, 'r') as file:
            lines = file.readlines()
        
            i = 0
            
            flag = 0
            read_data = 0
            header_data = 0
        
            data_columns = ['pres', 'geopot', 'temp', 'rh', 'hws']
            columns = ['id','time','lat','lon']
            columns = columns + data_columns
            df = pd.DataFrame(columns=columns)
            while i < len(lines):
                line = lines[i].strip()
        
                if (line.startswith('----------')) & (flag == 0):
                    read_data = 0
                    flag = 1
                    i += 1
                    continue
                    
                if (line.startswith('----------')) & (flag == 1):
                    read_data = 1
                    flag = 0
                    i += 1
                    continue  
        
                if read_data == 1:
                    # Parse data lines until the next header or end of file
                    data_aux = []
                    while i < len(lines) and not lines[i].startswith('</PRE>'):
                        data_line = lines[i]
                        pres = to_float(data_line[0:7])
                        geopot = to_float(data_line[8:14])
                        temp = to_float(data_line[15:20])
                        rh = to_float(data_line[32:35])
                        hws = to_float(data_line[53:56])*0.51444 
                        data_aux.append([pres,geopot,temp,rh,hws])
                        i += 1
                        
                    df_aux = pd.DataFrame(data=data_aux,columns = data_columns)
                    read_data = 0
                    header_data = 1
                                    
                if (line.startswith('Station number')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['id'] = to_float(data_line[15:19])
                    i += 1
                    continue
    
                if (line.startswith('Observation time')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['time'] = data_line[18:27]
                    i += 1
                    continue    
                
                if (line.startswith('Station latitude')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['lat'] = to_float(data_line[18:24])
                    i += 1
                    continue
        
                
                if (line.startswith('Station longitude')) & (header_data == 1):
                    data_line = lines[i].strip()
                    df_aux['lon'] = to_float(data_line[19:25])
                    df = pd.concat([df,df_aux])
                    i += 1
                    header_data = 0
                    continue
        
                i += 1
        df['time'] = pd.to_datetime(df['time'],format='%y%m%d/%H')        
        df = df.reset_index(drop=True)
        df_out = apply_multithreaded_groupby(df,func=get_rh_group)
    except Exception as e:
        print(e)
        df_out = pd.DataFrame()
    return df_out

# Function to apply using multi-threading
def apply_multithreaded_groupby(df, group_col=['time','id'], func=get_ablh_group, num_threads=256):
    # Group the DataFrame by the specified column
    grouped = df.groupby(group_col)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Apply function to each group in parallel
        results = list(executor.map(lambda group: func(group), grouped))
    
    # Concatenate the results back into a single DataFrame
    result_df = pd.concat(results)
    
    return result_df

############ UTILITIES FUNCTIONS #######################
def to_float(val):
    try:
        return float(val)
    except ValueError:
        return np.nan  # Return NaN if conversion fails
        
############ SIGPRO FUNCTIONS ##########################
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

############ PLOTTING FUNCTIONS ########################
#Custom scatter with statistical indicators and regression lines
def scatter(x,y,label_x,label_y,limx,limy,filename='example.png'):
        x = x[(x>limx[0])&(x<limx[1])&(y>limy[0])&(y<limy[1])]
        y = y[(x>limx[0])&(x<limx[1])&(y>limy[0])&(y<limy[1])]
        model = LinearRegression()
        model.fit(np.array(x).reshape(-1, 1), y)
        y_pred = model.predict(np.array(x).reshape(-1, 1))

        plt.grid('on')
        plt.scatter(x,y,s=5)
        plt.plot(x,y_pred,color='red')
        plt.ylim(limy[0],limy[1])
        plt.xlim(limx[0],limx[1])
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        R2 = np.corrcoef(x,y)**2
        RMSE = np.sqrt(np.sum((x-y)**2)/x.size)
        MD = np.sum((x-y))/x.size

        # Plotting the linear regression equation
        plt.text(0.1, 0.9, f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}', 
                horizontalalignment='left', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=10, color='black')
        plt.text(0.1, 0.85, f'$R^2$ = {R2[0,1]**2:.2f}', 
                horizontalalignment='left', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=10, color='black')
        plt.text(0.1, 0.8, f'RMSE = {RMSE:.2f} m', 
                horizontalalignment='left', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=10, color='black')
        plt.text(0.1, 0.75, f'MD = {MD:.2f} m', 
                horizontalalignment='left', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=10, color='black')
        
        plt.show()
        plt.savefig(filename)

############ OLD FUNCTIONS NOT USED ####################
def potential_temperature(temp,pres):
    pt = (temp+274.15)*(pres[0]/pres)**(287/1004)
    return pt

def potential_temperature_0(temp):
    pt = (temp[0]+0.5+274.15)*(1)**(287/1004)
    return pt
    
def ablh_model(z,bm,bu,zm,s):
    return (bm+bu)/2+(bm-bu)/2*special.erf((z-zm)/s)

def hw_transform(x,altitude,deltah):
    tr = np.array([])
    for z in altitude:
        idx_pos = (altitude >= z - deltah/2) & (altitude <= z)
        idx_neg = (altitude >= z) & (altitude <= z + deltah/2)
    
        tf_pos = np.mean(x[idx_pos])
        tf_neg = -np.mean(x[idx_neg])
    
        tf = tf_pos + tf_neg
        tr = np.append(tr,tf)
    return tr
    
def combined_method(altitude,vpt,plot_flag=0):
    try:
        #PREPARE DATA
        #FIRST APPLY HAAR-WAVELET (HW) TRANSFORM
    
        #COMPUTE DELTA-H
        deltah = 20*(altitude[1]-altitude[0])
    
        #COMPUTE HW TRANSFORM
        #LIMIT TO 3000 m height
        idx = altitude<3000
        hw = hw_transform(vpt[idx],altitude[idx],deltah)

        plt.figure()
        plt.plot(hw,altitude[idx])
        #GET INITIAL POINTS USING HW FOR CURVE FIT
        #GET PROVISIONAL ABLH TO BE USED AS INITIAL POINT FOR CURVEFIT
        zm_i = altitude[np.nanargmin(hw)]
        print(zm_i)
        s_i = zm_i/2.77
        
        #GET INITIAL POINT FOR BM AND BU
        BM_i = np.nanmean(vpt[0:np.nanargmin(hw)])
        BU_i = np.nanmean(vpt[np.nanargmin(hw):])

        popt, pcov = curve_fit(
            f=ablh_model,       # model function
            xdata=altitude,   # x data
            ydata=vpt,   # y data
            p0=(BM_i, BU_i, zm_i, s_i),      # initial value of the parameters (bm, bu, zm, s)
        )
    except:
        return 0
        
    if plot_flag == 1:
        plt.figure()
        print('a')
        plt.plot(ablh_model(altitude,popt[0],popt[1],popt[2],popt[3]),altitude)
        plt.plot(vpt,altitude)
    return popt[2]

def get_clouds_idx(altitude,temp,dtemp):
    cloud_altitudes_idx = (temp[altitude>100]-dtemp[altitude>100])<2
    cloud_sum = np.sum(cloud_altitudes_idx.astype(int))
    return cloud_altitudes_idx, cloud_sum
