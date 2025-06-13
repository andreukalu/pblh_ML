# ML METHOD FOR PBLH RETRIEVAL FROM CALIOP TAB MEASUREMENTS

Code to prepare the data and fit a machine learning (ML) model to estimate the Planetary Boundary Layer Height (PBLH) from CALIOP measurements of TAB profiles.

The data used comprises radiosonde vertical profiles from different sources and CALIOP TAB profiles. 

## DATA ACCESS 
### PROCESSED DATA ACCESS
* Processed radiosonde, CALIOP and training and test datasets can be accessed at: 
> https://zenodo.org/records/15639750

### RAW DATA ACCESS
* Raw radiosonde data can be openly accessed through:<br/>
> DWD: German Weather Service. https://opendata.dwd.de/<br/>
> NOAA: National Oceanic and Atmospheric Administration https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive  <br/>
> GRUAN: Global Climate Observing System Reference Upper-Air Network https://www.gruan.org/data/measurements/sonde-launches  <br/>
> UWYO: University of Wyoming upper-air sounding data-set https://weather.uwyo.edu/upperair/sounding.html  <br/>

* Raw CALIOP measurements can be openly accessed through:  <br/>
> https://subset.larc.nasa.gov/calipso/

## CODE FOLDERS TREE
* Code to process radiosonde measurements in folder /RADIOSONDE <br/>
* Code to process CALIOP measurements in folder /CALIPSO <br/>
* Code to train and test ML methods for PBLH retrieval in folder /ML <br/>

## INSTALLATION AND USAGE PROCESS
1. Pull github repository from https://github.com/andreukalu/pblh_ML/
2. cd to project folder and activate conda environment from environment.yml:
   '''
   conda env create -f environment.yml
   '''
3. Set each
      * pickles_path: path at which the processed data is stored [Download processed data](#processed-data-access)
      * (optional) rs_path: path at which the raw radiosonde data is stored [Download raw data](#raw-data-access)
      * (optional) c_path: path at which the raw caliop data is stored [Download raw data](#raw-data-access)
4. Now you can run each part of the code of [code folders tree](#code-folders-tree)
