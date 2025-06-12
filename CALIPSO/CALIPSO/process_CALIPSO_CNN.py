from calipso_utilities import *

res = 1.0
base_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES'
df_rs_1 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh'))
df_rs_2 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_south_america'))
df_rs_3 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_remaining'))
df_rs_4 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_china'))
df_rs_5 = pd.read_pickle(os.path.join(base_path,'radiosonde_ablh_africa'))
df = pd.concat([df_rs_1,df_rs_2,df_rs_3,df_rs_4,df_rs_5],axis=0).reset_index()

print(df.head(5))
df['lon'] = df['lon'].astype(float).round(1)
df['lat'] = df['lat'].astype(float).round(1)
df_rs = df[['lon','lat']].drop_duplicates()

del df
gc.collect()
df_rs['lon_min'] = df_rs['lon']-res
df_rs['lon_max'] = df_rs['lon']+res
df_rs['lat_min'] = df_rs['lat']-res
df_rs['lat_max'] = df_rs['lat']+res

# df_c1 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_us'))
# df_c2 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_gruan'))
# df_c3 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_germany'))
# df_c4 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_bcn'))
# df_c5 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_uw'))
# df_c6 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_ch'))
# df_c7 = pd.read_pickle(os.path.join(base_path,'calipso_CNN_rem'))

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/uw_coincidental_hr/'

name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
print(glob.glob(os.path.join(path, '**', '*.hdf')))
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_uw')

del df_out
gc.collect()

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/us_coincidental_hr/'

name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
print(glob.glob(os.path.join(path, '**', '*.hdf')))
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_us')

del df_out
gc.collect()

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/gruan_coincidental_hr/'
name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_gruan')

del df_out
gc.collect()

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/germany_coincidental_hr/'
name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_germany')

del df_out
gc.collect()

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/ch_coincidental_hr/'
name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_ch')

del df_out
gc.collect()

num_threads = 1024
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/barcelona_coincidental_hr/'
name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_bcn')

del df_out
gc.collect()

num_threads = 256
path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/CALIPSO/sa_af_coincidental_hr/'
name = '*.hdf'
filepaths = glob.glob(os.path.join(path, '**', '*.hdf'), recursive=True)
filepaths = [file.replace("\\", "/") for file in filepaths]
print(filepaths)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    
    # Submit each file to the executor
    futures = [executor.submit(read_file_CALIPSO,filepath,df_rs,0.25,1,'CNN') for filepath in filepaths]
    
    # Gather the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
df_out = pd.concat(results).reset_index()
df_out.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/calipso_CNN_rem')

del df_out
gc.collect()