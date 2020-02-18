import os
import subprocess
#import pathlib
import toml
from run_helpers import *


# =========================================================================== 
# =                             Setup DMFT                                  =
# =========================================================================== 
# --------------------------- read input file -------------------------------
config = toml.load("config.toml")
config['general']['codeDir'] = os.path.abspath(os.path.expanduser(config['general']['codeDir']))
# TODO: check for consistency of parameters

# -------------------------- create directories -----------------------------
dirName = config['general']['name']
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:
    reset_flag = query_yn("Directory " + dirName +  " already exists. Should everything be reset?", "no")
    if reset_flag:
        reset_dir(dirName)

# ---------------------- copy/edit/compile ed_dmft --------------------------
subDirName = dirName + "/ed_dmft"
files_list = ["tpri.dat", "init.h", "hubb.dat", "hubb.andpar"]
if not os.path.exists(subDirName):
    os.mkdir(subDirName)
create_and_populate_files(subDirName, files_list, config)
 
    

# copy code, generate hubb.dat and hubb.andpar


# generate run scripts, build dependency pipeline
# check for consistency and postproicess
# copy data to directory

cmd_cp_data = '''
mkdir -p data
cp ed_dmft/{gm_wim,g0m,g0mand,hubb.dat,hubb.andpar} data/
cp ed_vertex/gamma_dir/ data/ -r
cp ed_vertex/chi_dir/ data/ -r
cp ed_susc/chi_asympt data
cp ed_trilex/trip_omega data -r
cp ed_trilex/tripamp_omega data -r
cp ed_trilex/trilex_omega data -r
'''
