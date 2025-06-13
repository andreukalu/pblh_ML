# pblh_ML

Code to prepare the data and fit a machine learning (ML) model to estimate the Planetary Boundary Layer Height (PBLH) from CALIOP measurements of TAB profiles.

The data used comprises radiosonde vertical profiles from different sources and CALIOP TAB profiles. 

Radiosonde data can be openly accessed through:\
DWD: German Weather Service. https://opendata.dwd.de/\
NOAA: National Oceanic and Atmospheric Administration https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive\
GRUAN: Global Climate Observing System Reference Upper-Air Network https://www.gruan.org/data/measurements/sonde-launches\
UWYO: University of Wyoming upper-air sounding data-set https://weather.uwyo.edu/upperair/sounding.html\

CALIOP measurements can be openly accessed through:\
https://subset.larc.nasa.gov/calipso/

Code to process radiosonde measurements in folder /RADIOSONDE\
Code to process CALIOP measurements in folder /CALIPSO\
Code to train and test ML methods for PBLH retrieval in folder /ML
