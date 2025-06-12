import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
import glob
import concurrent.futures
from datetime import timedelta
from datetime import datetime
import gc
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import r2_score
import random
from astral.sun import sun
from astral import LocationInfo
import pytz
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew, kurtosis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import scipy
import json
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import pywt
from timezonefinder import TimezoneFinder
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter
import shap
import pywt
import pickle
import shapefile  # PyShp to read coastline without geopandas
from shapely.geometry import Point
from scipy.spatial import cKDTree

base_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES'

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

shp_file = os.path.join(base_path,'ne_10m_coastline.shp')

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

def fast_distance_to_coast(latitudes, longitudes):
    """Fast computation of distances to the nearest coastline."""
    query_points = np.column_stack([longitudes, latitudes])  # (lon, lat) format
    distances, _ = coastline_tree.query(query_points)  # Find nearest coastline point
    return distances * 111  # Convert degrees to km

def is_daytime(lat, lon, dt):
    try:
        # Ensure dt is timezone-aware (assuming input is always UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)

        # Get the time zone from latitude & longitude
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon, lat=lat)
        if timezone_str is None:
            raise ValueError("Could not determine the time zone for the given coordinates.")

        location = LocationInfo(latitude=lat, longitude=lon, timezone=timezone_str)
        
        # Define the detected time zone
        timezone = pytz.timezone(timezone_str)

        # Convert UTC time to local time
        local_dt = dt.astimezone(timezone)

        # Get sunrise and sunset for the **local date** (not just dt.date())
        s = sun(location.observer, date=local_dt.date(), tzinfo=timezone)

        # Convert sunrise/sunset to UTC
        sunrise_utc = s['sunrise'].astimezone(pytz.UTC)
        sunset_utc = s['sunset'].astimezone(pytz.UTC)

        # Compare UTC times correctly
        return sunrise_utc <= dt <= sunset_utc  # Returns True if it's daytime

    except Exception as e:
        print(f"Error: {e}")
        return None
    

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

# Simple check if it's daytime using zenith angle
def is_daytime_approx(lat, lon, dt):
    """ Check if it's daytime based on solar zenith angle approximation. """
    zenith_angle = solar_zenith(lat, lon, dt)
    return zenith_angle < 90  # If the angle is less than 90°, it's daytime

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

def shift_image_multiple_times(image_array,label_array,num_shifts=3,max_value=3500):

    shifted_images = []
    shifted_labels = []

    if isinstance(image_array, Image.Image):
        # Convert numpy array to a PIL image (needed for torchvision transforms)
        img = np.array(image_array)
    else:
        img = image_array

    shifted_images.append(img.copy())
    shifted_labels.append(label_array)

    for _ in range(num_shifts):
        shift_idx = random.randint(1,20)
        shifted_label = label_array + 30*shift_idx/max_value
        
        if shifted_label > 1:
            continue
        
        shifted_image = np.zeros_like(image_array)
        shifted_image[:, shift_idx:] = img[:, :-shift_idx]  # Shift pixels
        shifted_image[:, :shift_idx] = np.mean(img[:, :shift_idx])  # Fill the left side with zeros

        shifted_images.append(shifted_image.copy())
        shifted_labels.append(shifted_label)
    
    return shifted_images, shifted_labels