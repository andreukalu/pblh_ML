
# ML-Based PBLH Retrieval from CALIOP TAB Measurements

This repository contains code to preprocess data and train a machine learning (ML) model to estimate the **Planetary Boundary Layer Height (PBLH)** using **CALIOP** measurements of **TAB** profiles.

The dataset combines **radiosonde vertical profiles** from multiple sources with corresponding **CALIOP TAB** profiles.

---

Here's a polished and clearer version of your **Data Access** section:

---

## Data Access

### Processed Data

The preprocessed datasets—including radiosonde, CALIOP, and machine learning training and test sets—are publicly available at:

[Zenodo Repository – DOI: 10.5281/zenodo.15639750](https://zenodo.org/records/15639750)

**Files and descriptions:**

* **radiosonde\_pblh.pkl**: Processed planetary boundary layer height (PBLH) dataset derived from radiosonde measurements. Output of `RADIOSONDE/process_RS.py`.
* **calipso.pkl**: Processed Two-dimensional Atmospheric Boundary Layer Height (TAB) dataset from CALIPSO measurements. Output of `CALIPSO/process_CALIPSO.py`.
* **Xtrain**: Machine learning training dataset. Output of `ML/prepare_data.py`.
* **Xval**: Machine learning validation dataset. Output of `ML/prepare_data.py`.
* **Xtest**: Machine learning test dataset. Output of `ML/prepare_data.py`.

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

pickles_path should include the processed pickles downloaded from [Processed Data](#processed-data).
rs_path and c_path should contain radiosonde and calipso raw data, respectively. They can be downloaded from [Raw Data Sources](#raw-data-sources).

### 4. Run the code

Run `/RADIOSONDE/process_RS.py` to generate the PBLH radiosonde dataset from raw radiosonde measurements. 
* Inputs: Raw radiosonde measurement data.
* Outputs: radiosonde\_pblh.pkl pickle with the PBLH radiosonde dataset.
Run `/CALIPSO/process_CALIPSO.py` to generate the CALIPSO dataset from raw TAB CALIPSO measurements. 
* Inputs: Raw level 1 CALIPSO data.
* Outputs: calipso.pkl pickle with the CALIPSO pre-processed dataset.
* Requirements: radiosonde\_pblh.pkl pickle to be stored at pickles_path.
Run `/ML/prepare_data.py` to generate train and test datasets for ML.
* Inputs: radiosonde\_pblh.pkl and calipso.pkl.
* Outputs: Xtrain, Xval, and Xtest pickles.
Run `/ML/train_ML.py` to carry out hyper-parameter tuning, training and testing of the ML model. 
* Inputs: Xtrain, Xval, and Xtest pickles.
* Outputs: predictions pickle with the ML results. If param and random_flag are set to True, Hyper-parameter tuning results are stored at {method}_grid_search_iterations.txt.
Run `/ML/ablation_study.py` to run the ablation study.
* Inputs: Xtrain, Xval, and Xtest pickles.
* Outputs: ablation_results.txt with the ablation study results.

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the repository maintainer.
