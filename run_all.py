import os
import shutil
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
runDir = config['general']['runDir']
if not os.path.exists(runDir):
    os.mkdir(runDir)
    print("Directory " , runDir ,  " Created ")
else:
    reset_flag = query_yn("Directory " + runDir +  " already exists. Should everything be reset?", "no")
    if reset_flag:
        reset_dir(runDir)

# ---------------------- copy/edit/compile ed_dmft --------------------------
subRunDir = runDir + "/ed_dmft"
subCodeDir = config['general']['codeDir'] + "/ED_dmft"
files_list = ["tpri.dat", "init.h", "hubb.dat", "hubb.andpar"]
src_files  = ["ed_dmft_parallel_frequencies.f"]

if not os.path.exists(subRunDir):
    os.mkdir(subRunDir)
create_and_populate_files(subRunDir, files_list, config)

for src_file in src_files:
    shutil.copyfile(subCodeDir + "/" + src_file, subRunDir + "/" + src_file)
#TODO: edit parameters in file or change code in order to include as external
compile_command = "mpif90 " + ' '.join(src_files) + " -o ed_dmft.x -llapack -lblas " + config['general']['CFLAGS']
compile(compile_command, cwd=subRunDir ,verbose=config['general']['verbose'])
 

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
