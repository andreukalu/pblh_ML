import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import glob
import concurrent.futures
from datetime import timedelta
from datetime import datetime
import gc
from sklearn.model_selection import train_test_split

from PIL import Image
from sklearn.metrics import r2_score
import random
import pytz
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew, kurtosis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy
import json
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

from timezonefinder import TimezoneFinder
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import shapefile  # PyShp to read coastline without geopandas

from scipy.spatial import cKDTree

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from paths_definition import pickles_path

########### TERMINAL VISUALIZATION CONFIGURATION #######
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

########### LOAD COASTLINE MAPS #######################
maps_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES'
shp_file = os.path.join(maps_path,'ne_10m_coastline.shp')

# Read shapefile using PyShp
sf = shapefile.Reader(shp_file)
coastline_points = []

# Extract coastline points
for shape_rec in sf.shapes():
    if shape_rec.shapeType == 3:  # Type 3 = Polyline
        for x, y in shape_rec.points:
            coastline_points.append((x, y))
print('finished loading maps')

# Convert to NumPy array
coastline_array = np.array(coastline_points)

# Build KDTree for fast nearest neighbor search
coastline_tree = cKDTree(coastline_array)

############### METHODS ###############################
def fast_distance_to_coast(latitudes, longitudes):
    """Fast computation of distances to the nearest coastline."""
    query_points = np.column_stack([longitudes, latitudes])  # (lon, lat) format
    distances, _ = coastline_tree.query(query_points)  # Find nearest coastline point
    return distances * 111  # Convert degrees to km
    
# Function to calculate solar zenith angle
def solar_zenith(lat, lon, dt):
    """ Approximate the solar zenith angle based on the given latitude, longitude, and datetime. """
    # Convert the datetime to UTC if it's not already
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    
    # Approximate the solar declination angle (in degrees)
    delta = 23.44 * math.sin(math.radians(360/365 * (dt.timetuple().tm_yday - 81)))
    
    # Approximate the solar hour angle (in degrees)
    time_offset = (dt.hour - 12) * 15  # 1 hour = 15 degrees
    solar_hour_angle = time_offset + lon
    
    # Calculate the solar zenith angle
    zenith_angle = math.acos(
        math.sin(math.radians(lat)) * math.sin(math.radians(delta)) + 
        math.cos(math.radians(lat)) * math.cos(math.radians(delta)) * 
        math.cos(math.radians(solar_hour_angle))
    )
    
    # Convert zenith angle to degrees
    return math.degrees(zenith_angle)

def round_to_interval(dt, interval_hours):
    """Round datetime 'dt' to the nearest interval of 'interval_hours' hours"""
    # Calculate the base time for the interval (rounding to the nearest)
    base_hour = (dt.hour // interval_hours) * interval_hours
    base_time = datetime.combine(dt.date(), datetime.strptime(f"{base_hour:02d}:00", "%H:%M").time())

    # Calculate the next interval time
    next_time = base_time + timedelta(hours=interval_hours)

    # Determine the halfway point between base_time and next_time
    halfway_time = base_time + timedelta(hours=interval_hours / 2)

    # If the current time is equal to or after halfway_time, round up, else round down
    if dt >= halfway_time:
        rounded_time = next_time
    else:
        rounded_time = base_time

    return rounded_time

def round_to_6_hour_interval(dt):
    """Round datetime 'dt' to the nearest 6-hour interval (00:00, 06:00, 12:00, 18:00)"""
    interval_hours = 6

    # Round to nearest 6-hour interval
    rounded_datetime = round_to_interval(dt, interval_hours)
    return rounded_datetime

# Print the best results for each scoring metric
def print_best_grid_search_results(results, scoring_metrics):
    for score in scoring_metrics:
        print(f"\nBest result for {score}:")
        
        # Get the best score and the corresponding parameters
        best_score = results[f'mean_test_{score}'][results[f'rank_test_{score}'] == 1].max()
        best_params = results['params'][results[f'mean_test_{score}'].argmax()]
        
        print(f"Best score: {best_score:.4f}")
        print(f"Best parameters: {best_params}")

def compute_mean(matrix, axis=0):    
    mean_vertical_profile = np.mean(matrix, axis=axis)
    # mean_vertical_profile = (mean_vertical_profile - np.median(mean_vertical_profile))/(np.median(mean_vertical_profile))
    out = scipy.signal.medfilt(mean_vertical_profile,11)
    return out

def compute_sum(matrix, axis=0):    
    out = np.cumsum(matrix)
    return out

def compute_sector_mean(vec,sec=10):
    means = []
    segment_length = int(np.round(len(vec)/sec))
    
    for i in range(sec):
        vec_aux = vec[i*segment_length:(i+1)*segment_length]
        m = np.mean(vec_aux)
        means.append(m)
    return np.array(means)