import os
import shutil
import subprocess
#import pathlib
import toml
from run_helpers import *



# TODO: generate run scripts, build dependency pipeline
# TODO: check for consistency and postproicess
# TODO: copy data to directory


# =========================================================================== 
# =                               Setup                                     =
# =========================================================================== 
# --------------------------- read input file -------------------------------
config = toml.load("config.toml")
config['general']['codeDir'] = os.path.abspath(os.path.expanduser(config['general']['codeDir']))
# TODO: check for consistency of parameters
# TODO: check for correctly loaded modules

# -------------------------- create directories -----------------------------
runDir = config['general']['runDir']
if not os.path.exists(runDir):
    os.mkdir(runDir)
    print("Directory " , runDir ,  " Created ")
else:
    reset_flag = query_yn("Directory " + runDir +  " already exists. Should everything be reset?", "no")
    if reset_flag:
        reset_dir(runDir)

# =========================================================================== 
# =                                DMFT                                     =
# =========================================================================== 

# ---------------------------- definitions ----------------------------------
subCodeDir = config['general']['codeDir'] + "/ED_dmft"
files_list = ["tpri.dat", "init.h", "hubb.dat", "hubb.andpar"]
src_files  = ["ed_dmft_parallel_frequencies.f"]

# ----------------------------- create dir ----------------------------------
subRunDir_ED = runDir + "/ed_dmft"
if not os.path.exists(subRunDir_ED):
    os.mkdir(subRunDir_ED)

# ------------------------------ copy/edit ----------------------------------
create_and_populate_files(subRunDir_ED, files_list, config)
subRunDir_ED = runDir + "/ed_dmft"
for src_file in src_files:
    shutil.copyfile(subCodeDir + "/" + src_file, subRunDir_ED + "/" + src_file)

# ----------------------------- compile/run ---------------------------------
#TODO: edit parameters in file or change code in order to include as external
compile_command = "mpif90 " + ' '.join(src_files) + " -o ed_dmft.x -llapack -lblas " + config['general']['CFLAGS']
if not compile(compile_command, cwd=subRunDir_ED ,verbose=config['general']['verbose']):
    raise Exception("Compilation Failed")
if not run_ed_dmft(subRunDir_ED, config):
    raise Exception("Job submit failed")
 

# =========================================================================== 
# =                            DMFT Vertex                                  =
# =========================================================================== 

# ---------------------------- definitions ----------------------------------
subCodeDir = config['general']['codeDir'] + "/ED_vertex"
files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
files_list = ["call_script", "checksum_script", "clean_script", "idw.dat",
              "inversion_pp_fotso.f90", "parameters.dat", "split_script",
              "sum_t_files.f, tpri.dat", "varbeta.dat", "ver_tpri_run.f"]
src_files  = ["ver_tpri_run.f"]

# ----------------------------- create dir ----------------------------------
subRunDir_vert = runDir + "/ed_vertex"
if not os.path.exists(subRunDir_vert):
    os.mkdir(subRunDir_vert)

# ------------------------------ copy/edit ----------------------------------
copy_from_dmft(subRunDir_ED, subRunDir_vert, files_dmft_list)
for ifile in files_list:
    copy_and_edit(subCodeDir, subRunDir, ifile)




# =========================================================================== 
# =                          Postprocessing                                 =
# =========================================================================== 

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
