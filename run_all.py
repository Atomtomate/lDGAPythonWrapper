import os
import shutil
import subprocess
#import pathlib
import toml
from run_helpers import *



# TODO: check for consistency and postproicess


# =========================================================================== 
# =                               Setup                                     =
# =========================================================================== 
# --------------------------- read input file -------------------------------
config = toml.load("config.toml")
config['general']['codeDir'] = os.path.abspath(os.path.expanduser(config['general']['codeDir']))
check_config_consistency(config)
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
if not config['ED']['skip']:
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
    compile_command = "mpif90 " + ' '.join(src_files) + " -o run.x -llapack -lblas " + config['general']['CFLAGS']
    if not compile(compile_command, cwd=subRunDir_ED ,verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    ed_jobid = run_ed_dmft(subRunDir_ED, config)
    if not ed_jobid:
        raise Exception("Job submit failed")
 

# =========================================================================== 
# =                            DMFT Vertex                                  =
# =========================================================================== 

if not config['Vertex']['skip']:
    # ---------------------------- definitions ----------------------------------
    subCodeDir = config['general']['codeDir'] + "/ED_vertex"

    # ----------------------------- create dir ----------------------------------
    subRunDir_vert = runDir + "/ed_vertex"
    if not os.path.exists(subRunDir_vert):
        os.mkdir(subRunDir_vert)

    # ------------------------------ copy/edit ----------------------------------
    copy_and_edit_vertex(subCodeDir, subRunDir_vert, subRunDir_ED, config)

    # ----------------------------- compile/run ---------------------------------
    compile_command = "mpif90 ver_tpri_run.f -o run.x -llapack -lblas " + config['general']['CFLAGS']
    if not compile(compile_command, cwd=subRunDir_vert, verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    if not run_ed_vertex(subRunDir_vert, ed_jobid, config):
        raise Exception("Job submit failed")

    #TODO: edit tpri?
    #TODO: edit/compile/run sum_t_files
    #TODO: run split_script

# =========================================================================== 
# =                             DMFT Susc                                   =
# =========================================================================== 

if not config['Vertex']['susc']:
    # ---------------------------- definitions ----------------------------------
    subCodeDir = config['general']['codeDir'] + "/ED_physical_suscpetibility"

    # ----------------------------- create dir ----------------------------------
    subRunDir_susc = runDir + "/ed_susc"
    if not os.path.exists(subRunDir_susc):
        os.mkdir(subRunDir_susc)

    # ------------------------------ copy/edit ----------------------------------
    copy_and_edit_susc(subCodeDir, subRunDir_susc, subRunDir_ED, config)

    # ----------------------------- compile/run ---------------------------------
    compile_command = "gfortran calc_chi_asymptotics_gfortran.f -o run.x -llapack -lblas " + config['general']['CFLAGS']
    if not compile(compile_command, cwd=subRunDir_susc, verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    if not run_ed_susc(subRunDir_susc, ed_jobid, config):
        raise Exception("Job submit failed")


    # =========================================================================== 
# =                            DMFT Trilex                                  =
# =========================================================================== 

# =========================================================================== 
# =                          Postprocessing                                 =
# =========================================================================== 

#TODO: IMPORTANT!!!! copy code from dmft to other directories
#copy_from_dmft(subRunDir_ED, subRunDir_vert, files_dmft_list)


# TODO: run vertex post processing (sum_t still to do)

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
