from RS_utilities import *

base_path = '/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/RADIOSONDE'

num_threads = 16 #Limit to 16 as the GERMANY FILES ARE REALLY BIG

df_ablh = read_all_RS_data(base_path,sources=['REMAINING','UW','GERMANY','BCN','US','GRUAN'],num_threads=num_threads)

df_ablh.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/radiosonde_ablh_remaining')

#['UW','GERMANY','BCN','US','GRUAN']

# df_ablh = process_ABLH_from_all_RS_data(df_raw)

# df_ablh.to_pickle('/mnt/csl/work/andreu.salcedo/Articles/ABLH/02Data/PICKLES/radiosonde_ablh')

# print(df_ablh[['ablh_rs','lat','lon','time']])