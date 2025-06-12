from pyhdf.SD import *
from pyhdf import HDF, VS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import os
import glob
import gc
import scipy
from scipy.optimize import curve_fit
from scipy import special
import concurrent.futures
from sklearn.model_selection import train_test_split
import psutil

########### TERMINAL VISUALIZATION CONFIGURATION #######
pd.set_option('display.max_columns', None)

#FUNCTION TO READ CALIPSO FILES AND PROCESS AS A FUNCTION OF THE DESIRED OUTPUT
# INPUTS:
# - f: Filename
# - df_rs: Dataframe containing all Radiosonde PBLH measurements
# - res: horizontal resolution of the output windows in [lat/lon deg]
# OUTPUTS:
# - df: dataframe containing the 2D TAB profiles intersecting RS stations 
def read_file_CALIPSO(f, df_rs, res=0.1):

    print(f)
    #READ FILE TO GET BACKSCATTER AND ALTITUDES
    try:
        hdf = SD(f,SDC.READ)
        h = HDF.HDF(f)
    except:
        print('FAILED: ',f)
        return
    
    try:    
        #GET LONGITUDE; LATITUDE; AND MEASUREMENT ALTITUDES
        longitudes, latitudes = get_longitude_latitude(hdf,type)
        altitudes = np.flip(get_altitudes(h))

        longitudes = np.array(longitudes)
        latitudes = np.array(latitudes)

        longitudes_ERA5 = longitudes//res*res
        latitudes_ERA5 = latitudes//res*res
        
        coordinates = np.concatenate((longitudes_ERA5,latitudes_ERA5),axis=1)
        coordinates = np.unique(coordinates,axis=0)

        #GET BACKSCATTER
        backscatter = np.flip(np.array(hdf.select('Total_Attenuated_Backscatter_532').get()),axis=1)
    except Exception as e:
        print('FAILED: ',f,e)
        return
    
    #GET MIN ALTITUDE
    min_altitude = hdf.select('Surface_Elevation').get()

    #GET TIME
    ref_time = datetime(1993,1,1)
    time = np.array([ref_time + timedelta(seconds=s.item()) for s in hdf.select('Profile_Time').get()])
    
    dict = {}
    
    df = pd.DataFrame(data=dict)
    for i in np.arange(0,coordinates.shape[0]-1,1):
        
        is_within_any_bounds = ((coordinates[i,0] >= df_rs['lon_min']) & 
                        (coordinates[i,0] <= df_rs['lon_max']) &
                        (coordinates[i,1] >= df_rs['lat_min']) & 
                        (coordinates[i,1] <= df_rs['lat_max'])).any()
        
        if not is_within_any_bounds:
            continue

        lon_target = (coordinates[i,0]-res/2,coordinates[i,0]+res/2)
        lat_target = (coordinates[i,1]-res/2,coordinates[i,1]+res/2)

        idx = (longitudes<lon_target[1])&(longitudes>lon_target[0])&(latitudes<lat_target[1])&(latitudes>lat_target[0])
        
        idx = np.where(idx[:,0])[0]
        
        if len(longitudes[idx])<=2:
            continue

        idx_mean = idx[len(idx)//2]
        idx1 = round(idx_mean - res*333/2)
        idx2 = round(idx_mean + res*333/2)
        
        if (idx1<=0)|(idx2>=len(longitudes)):
            continue
            
        night = int((time[0].hour<9) | (time[0].hour>23))
        
        try:
            backscatter_target = backscatter[idx1:idx2,:]
            
            time_target = time[idx1:idx2]
        except Exception as e:
            print('FAILED: ',f,e)
            continue

        try:
            backscatter_target = backscatter_target[:,:200]
            
            dict = {'hour':[time_target[0].hour],'month':[time_target[0].month],'lat':[lat_target],\
                    'lon':[lon_target],'time':[time_target[0]],'backscatter':[backscatter_target]}
            
            combined_df = pd.DataFrame(data=dict)
            df = pd.concat([df,combined_df],axis=0)
        except Exception as e:
            print('FAILED: ',f,e)
            continue

    pid = os.getpid()

    # Get the memory usage of the current process
    process = psutil.Process(pid)
    memory_info = process.memory_info()

    # Print memory usage in MB
    del longitudes, latitudes, altitudes, backscatter, min_altitude, time, coordinates
    print(f"Memory usage: {memory_info.rss / (1024 * 1024)} MB")
    hdf.end()  # Properly close the SD object
    h.close()
    del hdf, h
    gc.collect()

    return df

#FUNCTION TO GET THE ALTITUDES OF A CALIPSO BACKSCATTER FILE
# INPUTS:
# - hdf: File HDF variable 
# OUTPUT: ARRAY WITH ALTITUDE    
def get_altitudes(hdf):
    vs = hdf.vstart()
    xid = vs.find('metadata')
    altid = vs.attach(xid)
    altid.setfields('Lidar_Data_Altitudes')
    nrecs, _, _, _, _ = altid.inquire()
    altitude = altid.read(nRec=nrecs)
    altid.detach()
    alti = np.array(altitude[0][0])
    return alti

# FUNCTION TO GET THE LATITUDE AND LONGITUDE OF A CALIPSO BACKSCATTER FILE
# INPUTS:
# - hdf: File HDF variable 
# - type: integer stating level of CALIPSO data file. 1=L1 and 2=L2
# OUTPUTS:
# - longitudemid: mean longitude of the three CALIPSO laser measurements 
# - latitudemid: mean latitude of the three CALIPSO laser measurements
def get_longitude_latitude(hdf,type=1):
    latitude = hdf.select('Latitude').get()
    longitude = hdf.select('Longitude').get()
    
    latitudemid = [item for item in latitude]
    longitudemid = [item for item in longitude]
        
    return longitudemid, latitudemid

def read_all_CALIPSO_data(base_path, df_rs, num_threads= 256):
    
    filepaths = glob.glob(os.path.join(base_path, '**', '*.hdf'), recursive=True)
    filepaths = [file.replace("\\", "/") for file in filepaths]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        
        # Submit each file to the executor
        futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25) for filepath in filepaths]
        
        # Gather the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    df_out = pd.concat(results).reset_index()

    return df_out