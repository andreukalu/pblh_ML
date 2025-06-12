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
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
from scipy.ndimage import percentile_filter
from sklearn.model_selection import train_test_split
import psutil

pd.set_option('display.max_columns', None)

#FUNCTION TO READ CALIPSO FILES AND PROCESS AS A FUNCTION OF THE DESIRED OUTPUT
def read_file_CALIPSO(f,df_rs,res=0.1,type=1,method = 'CNN',plot_flag=0,filter_ground = 1):

    print(f)
    #READ FILE TO GET BACKSCATTER AND ALTITUDES
    try:
        hdf = SD(f,SDC.READ)
        h = HDF.HDF(f)
    except:
        print('FAILED: ',f)
        return
    
    #GET FILE TYPE
    if 'L1' in f:
        type = 1
    else:
        type = 2
    
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
    
    if (method == 'ML') | (method == 'CNN'):
        dict = {}
    else:
        dict = {'time_CALIPSO':[],'lon_CALIPSO':[],'lat_CALIPSO':[],'ablh_CALIPSO':[],\
                    'min_altitude':[],'power':[],'ratio':[],'cloud_flag':[]}
    df = pd.DataFrame(data=dict)
    for i in np.arange(0,coordinates.shape[0]-1,1):
        
        is_within_any_bounds = ((coordinates[i,0] >= df_rs['lon_min']) & 
                        (coordinates[i,0] <= df_rs['lon_max']) &
                        (coordinates[i,1] >= df_rs['lat_min']) & 
                        (coordinates[i,1] <= df_rs['lat_max'])).any()
        # print(coordinates[i,:])
        # print(df_rs[df_rs['lon_min']<-160])#,'lat_min','lat_max']])
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
            min_altitude_target = min_altitude[idx1:idx2]
            
            time_target = time[idx1:idx2]
            longitudes_target = lon_target
            latitudes_target = lat_target
        except Exception as e:
            print('FAILED: ',f,e)
            continue
    
        #GET ABLH
        l = len(np.arange(1,len(time),1).tolist())

        if method == 'CNN':
            try:
                backscatter_target = backscatter_target[:,:200]
                print('idxdiff:' + str(backscatter_target.shape))
                dict = {'hour':[time_target[0].hour],'month':[time_target[0].month],'lat':[lat_target],'lon':[lon_target],'time':[time_target[0]],'backscatter':[backscatter_target]}
                        # Determine the number of columns in the NumPy array
                
                combined_df = pd.DataFrame(data=dict)
                df = pd.concat([df,combined_df],axis=0)
            except Exception as e:
                print('FAILED: ',f,e)
                continue

        elif method == 'ML':
            
            backscatter_out,altitude,min_altitude,cloud_flag = get_backscatter_profile(backscatter_target,altitudes,min_altitude_target,0,0)
            dict = {'hour':[time_target[0].hour],'month':[time_target[0].month],'cloud':[cloud_flag],'lat':[lat_target],'lon':[lon_target],'time':[time_target[0]]}
                    # Determine the number of columns in the NumPy array
            
            df_aux = pd.DataFrame(data=dict)

            backscatter_out = backscatter_out[altitude<8000]
            num_new_columns = backscatter_out.shape[0]
            
            # Generate column names for the new data
            new_columns = [f'Feature_{i+1}' for i in range(num_new_columns)]

            # Create a new DataFrame from the NumPy array
            new_df = pd.DataFrame(np.transpose(backscatter_out[:, np.newaxis]), columns=new_columns)
            
            # Combine the original DataFrame with the new DataFrame
            combined_df = pd.concat([df_aux.reset_index(drop=True), new_df], axis=1)
            
            df = pd.concat([df,combined_df],axis=0)
        else:
            # try:
            ablh, ratio, power, cloud_flag = get_ABLH(backscatter_target,altitudes,min_altitude_target,night,method,plot_flag,filter_ground)
            
            min_altitude_out = [np.nanmean(min_altitude_target)]
            longitudes_out = coordinates[i,0]
            latitudes_out = coordinates[i,1]
            time_out = time_target[0]
            
            dict = {'time_CALIPSO':time_out,'lon_CALIPSO':longitudes_out,'lat_CALIPSO':latitudes_out,'ablh_CALIPSO':ablh,\
                    'min_altitude':min_altitude_out,'power':power,'ratio':ratio,'cloud_flag':cloud_flag}
            
            df_aux = pd.DataFrame(data=dict)
            df = pd.concat([df,df_aux],axis=0)
        # except:
        #     continue
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
    if type == 1:
        latitudemid = [item for item in latitude]
        longitudemid = [item for item in longitude]
    elif type == 2:
        latitudemid = [item[1] for item in latitude]
        longitudemid = [item[1] for item in longitude]
        
    return longitudemid, latitudemid

def read_reflectance_profile(f,lon_target=(10,12),lat_target = (50,51),filter_ground=0):
    #READ FILE TO GET BACKSCATTER AND ALTITUDES
    hdf = SD(f,SDC.READ)
    h = HDF.HDF(f)

    #GET FILE TYPE
    if 'L1' in f:
        type = 1
    else:
        type = 2
    
    altitudes = np.flip(get_altitudes(h))
    #GET LONGITUDE; LATITUDE; AND MEASUREMENT ALTITUDES
    longitudes, latitudes = get_longitude_latitude(hdf,type)
    altitudes = np.flip(get_altitudes(h))

    longitudes = np.array(longitudes)
    latitudes = np.array(latitudes)
    
    idx = (longitudes<lon_target[1])&(longitudes>lon_target[0])&(latitudes<lat_target[1])&(latitudes>lat_target[0])
    idx = idx[:,0]
    if type == 1:
        #GET BACKSCATTER
        backscatter = np.flip(hdf.select('Total_Attenuated_Backscatter_532').get(),axis=1)
        
        #GET TIME
        ref_time = datetime(1993,1,1)
        time = [ref_time + timedelta(seconds=s.item()) for s in hdf.select('Profile_Time').get()]

        #GET SURFACE ELEVATION
        min_altitude = hdf.select('Surface_Elevation').get()
        if filter_ground == 1:
            backscatter = remove_ground(backscatter,altitudes,min_altitude)
    elif type == 2:
        #GET BACKSCATTER
        backscatter = hdf.select('Backscatter_Coefficient_1064').get()

        #GET TIME
        ref_time = datetime(1993,1,1)
        time = [ref_time + timedelta(seconds=s) for s in hdf.select('Profile_Time').get()[:,1]]

        #GET MIN ALTITUDE
        min_altitude = np.flip(hdf.select('Surface_Base_Altitude_532').get())

    backscatter = backscatter[idx,:]
    min_altitude = min_altitude[idx]
    time = np.array(time)[idx]

    return time, altitudes, backscatter, min_altitude

def round_to_interval(dt, interval_hours):
    """Round datetime 'dt' to the nearest interval of 'interval_hours' hours"""
    # Calculate the base time for the interval
    base_hour = (dt.hour // interval_hours) * interval_hours
    base_time = datetime.combine(dt.date(), datetime.strptime(f"{base_hour:02d}:00", "%H:%M").time())

    # Calculate the next interval time
    if dt.hour % interval_hours >= interval_hours / 2:
        rounded_time = base_time + timedelta(hours=interval_hours)
    else:
        rounded_time = base_time

    return rounded_time

def round_to_6_hour_interval(dt):
    """Round datetime 'dt' to the nearest 6-hour interval (00:00, 06:00, 12:00, 18:00)"""
    interval_hours = 6

    # Round to nearest 6-hour interval
    rounded_datetime = round_to_interval(dt, interval_hours)
    return rounded_datetime

def get_night(hour):
    if (hour<9)|(hour>21):
        df = pd.dataframe([1,0])
    else:
        df = pd.dataframe([0,1])
    return night

def ablh_model(z,bm,bu,zm,s):
    return ((bm+bu)+bu)/2-((bm+bu)-bu)/2*special.erf((z-zm)/s)

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

def get_initial_point(x,y,min_altitude=0,night=0,plot_flag = 0):
    #COMPUTE DELTA-H
    deltah = 20*(x[1]-x[0])

    #COMPUTE HW TRANSFORM
    #LIMIT TO 3000 m height
    
    bm_lw = 0
    bm_ub = 0.2
    bu_lb = 0
    bu_ub = 0.2
    if night == 1:
        zm_lb = 0
        zm_ub = 1500 + min_altitude
        s_lb = 0
        s_ub = 200
        idx = x<1500
    else:
        zm_lb = 500
        zm_ub = 3000 + min_altitude
        s_lb = 0
        s_ub = 1000
        idx = x<5000

    hw = hw_transform(y[idx],x[idx],deltah)
    
    #GET INITIAL POINTS USING HW FOR CURVE FIT
    #GET PROVISIONAL ABLH TO BE USED AS INITIAL POINT FOR CURVEFIT
    zm_i = np.nanmin([x[np.nanargmax(hw)],zm_ub])
    zm_i = np.nanmax([zm_i,zm_lb])
    
    s_i = np.nanmin([zm_i/2.77,s_ub])
    
    #GET INITIAL POINT FOR BM AND BU
    BM_i = np.nanmin([np.nanmean(y[0:np.nanargmax(hw)]),bm_ub])
    BU_i = np.nanmin([np.abs(np.nanmean(y[np.nanargmax(hw):])),bu_ub])

    BM_i = np.nanmax([BM_i-BU_i,0])

    # if plot_flag==1:
    #     plt.figure()
    #     plt.plot(hw,x[idx])

    return BM_i, BU_i, zm_i, s_i 
    
def combined_method(x,y,night,min_altitude=0,plot_flag=0):
    try:
            # PREPARE DATA
            # GET INITIAL POINTS
    
        BM_i, BU_i, zm_i, s_i  = get_initial_point(x,y,night,min_altitude,0)
        
        bm_lw = 0
        bm_ub = 0.2
        bu_lb = 0
        bu_ub = 0.2
        
        if night:
            zm_lb = 0
            zm_ub = 1500 + min_altitude
            s_lb = 0
            s_ub = 200
        else:
            zm_lb = 0
            zm_ub = 3000 + min_altitude
            s_lb = 0
            s_ub = 1000
    
        # print((BM_i, BU_i, zm_i, s_i))
        popt, pcov = curve_fit(
            f=ablh_model,       # model function
            xdata=x,   # x data
            ydata=y,   # y data
            bounds = ([bm_lw,bu_lb,zm_lb,s_lb],[bm_ub,bu_ub,zm_ub,s_ub]),
            p0=(BM_i, BU_i, zm_i, s_i))      # initial value of the parameters (bm, bu, zm, s))
    
        if (popt[2]>3000 + min_altitude)|(popt[2]<min(x)+150)|(popt[3]>5000):
            ablh = min(x)
        else:
            ablh = popt[2]
        ratio = popt[3]
        
        if plot_flag == 1:
            plt.figure()
            plt.plot(ablh_model(x,popt[0],popt[1],popt[2],popt[3]),x)
            plt.plot(y,x)
            # plt.xlim(0,0.01)
    except:
        ablh = min(x)
        ratio = 0
        
    return ablh, ratio

def hw_method(x,y,min_altitude=0,night=0,plot_flag=0):
    try:
        # PREPARE DATA
        # GET INITIAL POINTS
        
        BM_i, BU_i, ablh, s_i  = get_initial_point(x,y,night,min_altitude,plot_flag)
        ratio = BU_i/BM_i
        if plot_flag:
            plt.figure()
            plt.plot(y,x)
    except:
        ablh = min(x)
        ratio = 0
        
    return ablh, ratio

def smooth(arr, sigma):
    smoothed_arr = gaussian_filter(arr, sigma=sigma)
    return smoothed_arr

def smooth_uniform(arr, samples):
    smoothed_arr = uniform_filter(arr, size=samples)
    return smoothed_arr

def remove_ground(backscatter_in,altitudes,min_altitude):
    idx_ground = [np.where(altitudes>min_a+0.05)[0][0] for min_a in min_altitude]
    
    for i in np.arange(0,min_altitude.shape[0]-1,1).tolist():
        backscatter_in[i,0:idx_ground[i]] = np.nan#np.nanmean(backscatter[i,idx_ground[i]:idx_ground[i]+filter_samples])
    return backscatter_in

# FUNCTION TO RETRIEVE THE BACKSCATTER VERTICAL PROFILE USING A HORIZONTAL WINDOW
# INPUTS:
# - backscatter_in: numpy array having the backscatter values.
# - altitude: numpy array having the measurement altitudes.
# - min_altitude: numpy array having the ground altitudes in backscatter_in
# - filter_ground: flag if the ground is to be filtered from the backscatter
# - fitler_clouds: flag if the clouds are to be filtered out from the backscatter.
def get_backscatter_profile(backscatter_in,altitude,min_altitude,filter_ground=0,filter_clouds=1):
    if filter_ground==1:
        backscatter_in = remove_ground(backscatter_in,altitude,min_altitude)

    backscatter_in[backscatter_in<-1000] = np.nan

    if len(backscatter_in.shape) == 2:
        
        cloud = np.zeros(backscatter_in.shape[0])
        for i in np.arange(0,backscatter_in.shape[0],1):
            if len(backscatter_in[i,backscatter_in[i,:]>0.04])>0:
                cloud[i] = 1
        cloud_flag = len(cloud[cloud==1])/len(cloud)
        # backscatter = np.nanmean(backscatter_in[cloud==0,:],axis=0)
        if filter_clouds == 1:
            backscatter = np.hamming(backscatter_in[cloud==0,:].shape[0]).dot(backscatter_in[cloud==0,:])
        else:
            # backscatter = np.hamming(backscatter_in.shape[0]).dot(backscatter_in)
            backscatter = np.nanmean(backscatter_in,axis=0)
        # backscatter = np.nanpercentile(backscatter_in[cloud==0,:],50,axis=0)        
    else:
        backscatter = backscatter_in

    #Convert altitude to meters
    altitude = altitude*1000
    min_altitude = min_altitude*1000

    return backscatter, altitude, min_altitude, cloud_flag

# FUNCTION TO RETRIEVE THE ABLH FROM CALIPSO BACKSCATTER USING CLASSIC/SIGNAL PROCESSING METHODS    
def get_ABLH(backscatter_in,altitude,min_altitude,night,method = 'combined',plot_flag = 0,filter_ground = 0,filter_clouds = 1):
    
    backscatter, altitude, min_altitude, cloud_flag = get_backscatter_profile(backscatter_in,altitude,min_altitude,filter_ground,filter_clouds)

    min_altitude_max = min(min_altitude)[0]
    
    # min_altitude = 0
    if night == 1:
        idx = (altitude < 1300) & (altitude > 0) & (backscatter < 5) & (backscatter > -1)
        power = np.nansum((backscatter-np.nanmean(backscatter))**2)/len(backscatter)
        
        if len(backscatter[idx])==0:
            return 0, 0, power, cloud_flag
        backscatter = backscatter + abs(np.min(backscatter[idx]))
    else:
        idx = (altitude < 5000) & (altitude > min_altitude_max+150) & (backscatter < 5) & (backscatter > -1)
        power = np.nansum((backscatter-np.nanmean(backscatter))**2)/len(backscatter)
        if len(backscatter[idx])==0:
            return 0, 0, power, cloud_flag
        backscatter = backscatter - np.min(backscatter[idx])
        
    backscatter = backscatter[idx]
    altitude = altitude[idx]

    if night == 1:
        ablh, ratio = hw_method(altitude,backscatter,night,min_altitude_max,plot_flag)
        power = np.sum((backscatter-np.nanmean(backscatter))**2)/len(backscatter)
    else:
        ablh, ratio = combined_method(altitude,backscatter,night,min_altitude_max,plot_flag)
        power = np.sum((backscatter-np.nanmean(backscatter))**2)/len(backscatter)
    
    return ablh, ratio, power, cloud_flag

############################## PLOT FUNCTIONS ##############################################################
def plot_trajectory(data,figsize,trace_size):
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data,str):
        time,altitude,backscatter, surf_elevation = read_reflectance_profile(data)
        df = rc.read_file(data)

    #PRINT CALIPSO COORDINATES ON EARTH MAP
    fig = plt.subplots(figsize = figsize)
    ax = plt.subplot()
    europe = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

    Nrows = len(df['lon_CALIPSO'])
    Ncols = 1
    df_ones = pd.DataFrame(np.ones((Nrows,Ncols)))
    gdf = gpd.GeoDataFrame(df_ones, geometry = gpd.points_from_xy(df['lon_CALIPSO'], df['lat_CALIPSO']), crs = 'EPSG:4326')
    gdf.plot(ax= ax,markersize=trace_size)
    ctx.add_basemap(ax, crs='EPSG:4326')
    plt.show()

def plot_cmap(data,time,altitude,figsize,ylim,xidx,clim):
    data[data<=0] = 1e-9
    plt.figure(figsize=figsize)
    plt.pcolor(time[xidx[0]:xidx[1]],altitude*1000,data[xidx[0]:xidx[1]-1,:-1].transpose(),norm = mpl.colors.LogNorm())
    plt.colorbar()
    plt.clim(clim[0],clim[1])
    plt.ylim(ylim[0],ylim[1])

def display(array1, array2):
    """Displays ten random images from each array."""
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices]
    images2 = array2[indices]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        print(image1.shape)
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig('images.png')