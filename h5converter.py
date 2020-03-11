import pandas as pd
import pandas.io.common
import pyarrow.parquet as pq

files = ["chi_asympt", "F_DM", "GAMMA_DM_FULLRANGE", "vert_chi","g0m", "g0mand", "gm_wim"]
dirs = ["trilex_omega", "tripamp_omega", "trip_omega"]

files_not_found = []

# =========================================================================== 
# =                             chi_asympt                                  =
# =========================================================================== 
fn = "chi_asympt"
try:
    df = pd.read_csv(fn, sep=' ', skipinitialspace=True, header=0, names=["omega","Re_chi_up",\
                     "Im_chi_up", "Re_chi_do", "Im_chi_do",\
                     "Re_chi_ch", "Im_chi_ch"])
    df.to_hdf('results.h5', key='chi_asympt', mode='w')
            
except pandas.io.common.EmptyDataError:
    print("WARNING: chi_asympt not found!")
    files_not_found.append(fn)


# =========================================================================== 
# =                                F_DM                                     =
# =========================================================================== 
fn = "F_DM"
try:
    df = pd.read_csv(fn, sep=' ', skipinitialspace=True, header=0, names=["Re_F_density",\
                     "Im_F_density", "Re_F_magnetic", "Im_F_magnetic"])
    df.to_hdf('results.h5', key='F_DM')
            
except pandas.io.common.EmptyDataError:
    print("WARNING: FM not found!")
    files_not_found.append(fn)

# =========================================================================== 
# =                             chi_asympt                                  =
# =========================================================================== 


# =========================================================================== 
# =                             chi_asympt                                  =
# =========================================================================== 


# =========================================================================== 
# =                             chi_asympt                                  =
# =========================================================================== 

