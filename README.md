
# ML-Based PBLH Retrieval from CALIOP TAB Measurements

This repository contains code to preprocess data and train a machine learning (ML) model to estimate the **Planetary Boundary Layer Height (PBLH)** using **CALIOP** measurements of **TAB** profiles.

The dataset combines **radiosonde vertical profiles** from multiple sources with corresponding **CALIOP TAB** profiles.

---

## Data Access

### Processed Data

The preprocessed dataset (radiosonde, CALIOP, training and test sets) is available at:

[Zenodo Repository – DOI: 10.5281/zenodo.15639750](https://zenodo.org/records/15639750)

---

### Raw Data Sources

#### Radiosonde Data

You can access radiosonde data from the following public archives:

- [DWD – German Weather Service](https://opendata.dwd.de/)
- [NOAA – IGRA Dataset](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive)
- [GRUAN – Reference Upper-Air Network](https://www.gruan.org/data/measurements/sonde-launches)
- [University of Wyoming – Upper-Air Soundings](https://weather.uwyo.edu/upperair/sounding.html)

#### CALIOP Data

- [NASA CALIPSO Subsetter](https://subset.larc.nasa.gov/calipso/)

---

## Code Structure

The repository is organized as follows:

- `/RADIOSONDE/` — Code for preprocessing radiosonde profiles  
- `/CALIPSO/` — Code for processing CALIOP TAB profiles  
- `/ML/` — Code for training and evaluating ML models for PBLH estimation  

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/andreukalu/pblh_ML.git
cd pblh_ML
````

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate pblh
```

### 3. Configure local data paths

Edit the `paths_definition.py` file:

```python
pickles_path = '/your/path/to/processed/data'
rs_path = '/optional/path/to/raw/radiosonde/data'
c_path = '/optional/path/to/raw/calipso/data'
```

You can find these files from the [Processed Data](#processed-data) and [Raw Data Sources](#raw-data-sources) sections above.

### 4. Run the pipeline

Use the code in the `/RADIOSONDE/`, `/CALIPSO/`, and `/ML/` directories to preprocess radiosonde data, calipso data, and train and test ML models.

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the repository maintainer.
