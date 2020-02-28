import os
import shutil
import subprocess
#import pathlib
import toml
from run_helpers import *



# TODO: check for consistency and postproicess
# TODO: IMPORTANT! give option to restart from hubb.andpar
# TODO: IMPORTANT! save jobid in file and check on restart if it is still running/exit ok!


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
        confirm = query_yn("This will purge the directory. Do you want to continue?", "no")
        if confirm:
            reset_dir(runDir)

# =========================================================================== 
# =                                DMFT                                     =
# =========================================================================== 
 
# ---------------------------- definitions ----------------------------------
subCodeDir = config['general']['codeDir'] + "/ED_dmft"
files_list = ["tpri.dat", "init.h", "hubb.dat", "hubb.andpar"]
src_files  = ["ed_dmft_parallel_frequencies.f"]
subRunDir_ED = runDir + "/ed_dmft"
compile_command = "mpif90 " + ' '.join(src_files) + " -o run.x -llapack -lblas " + config['general']['CFLAGS']

if not config['ED']['skip']:
    # ----------------------------- create dir ----------------------------------
    if not os.path.exists(subRunDir_ED):
        os.mkdir(subRunDir_ED)

    # ------------------------------ copy/edit ----------------------------------
    create_and_populate_files(subRunDir_ED, files_list, config)
    for src_file in src_files:
        shutil.copyfile(subCodeDir + "/" + src_file, subRunDir_ED + "/" + src_file)

    # ----------------------------- compile/run ---------------------------------
    #TODO: edit parameters in file or change code in order to include as external
    if not compile(compile_command, cwd=subRunDir_ED ,verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    ed_jobid = run_ed_dmft(subRunDir_ED, config)
    if not ed_jobid:
        raise Exception("Job submit failed")
else:
    ed_jobid = None

# =========================================================================== 
# =                            DMFT Vertex                                  =
# =========================================================================== 

# ---------------------------- definitions ----------------------------------
subRunDir_vert = runDir + "/ed_vertex"
subCodeDir = config['general']['codeDir'] + "/ED_vertex"
compile_command = "mpif90 ver_tpri_run.f -o run.x -llapack -lblas " + config['general']['CFLAGS']

if not config['Vertex']['skip']:
    # ----------------------------- create dir ----------------------------------
    if not os.path.exists(subRunDir_vert):
        os.mkdir(subRunDir_vert)

    # ------------------------------ copy/edit ----------------------------------
    copy_and_edit_vertex(subCodeDir, subRunDir_vert, subRunDir_ED, config)

    # ----------------------------- compile/run ---------------------------------
    if not compile(compile_command, cwd=subRunDir_vert, verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    if not run_ed_vertex(subRunDir_vert, config, ed_jobid):
        raise Exception("Job submit failed")

    #TODO: edit tpri?
    #TODO: edit/compile/run sum_t_files
    #TODO: run split_script

# =========================================================================== 
# =                             DMFT Susc                                   =
# =========================================================================== 

# ---------------------------- definitions ----------------------------------
subCodeDir = config['general']['codeDir'] + "/ED_physical_suscpetibility"
compile_command = "gfortran calc_chi_asymptotics_gfortran.f -o run.x -llapack -lblas " + config['general']['CFLAGS']

if not config['Susc']['skip']:
    # ----------------------------- create dir ----------------------------------
    subRunDir_susc = runDir + "/ed_susc"
    if not os.path.exists(subRunDir_susc):
        os.mkdir(subRunDir_susc)

    # ------------------------------ copy/edit ----------------------------------
    copy_and_edit_susc(subCodeDir, subRunDir_susc, subRunDir_ED, config)

    # ----------------------------- compile/run ---------------------------------
    if not compile(compile_command, cwd=subRunDir_susc, verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    if not run_ed_susc(subRunDir_susc, config, ed_jobid):
        raise Exception("Job submit failed")


# =========================================================================== 
# =                            DMFT Trilex                                  =
# =========================================================================== 

# ---------------------------- definitions ----------------------------------
subCodeDir = config['general']['codeDir'] + "/ED_Trilex_Parallel"
compile_command = "mpif90 ver_twofreq_parallel.f -o run.x -llapack -lblas " + config['general']['CFLAGS']

if not config['Trilex']['skip']:
    # ----------------------------- create dir ----------------------------------
    subRunDir_trilex = runDir + "/ed_trilex"
    output_dirs = ["trip_omega", "tripamp_omega", "trilex_omega"]
    if not os.path.exists(subRunDir_trilex):
        os.mkdir(subRunDir_trilex)
    for d in output_dirs:
        fp = os.path.abspath(os.path.join(subRunDir_trilex, d))
        if not os.path.exists(fp):
            os.mkdir(fp)

    # ------------------------------ copy/edit ----------------------------------
    copy_and_edit_trilex(subCodeDir, subRunDir_trilex, subRunDir_ED, config)

    # ----------------------------- compile/run ---------------------------------
    if not compile(compile_command, cwd=subRunDir_trilex, verbose=config['general']['verbose']):
        raise Exception("Compilation Failed")
    if not run_ed_trilex(subRunDir_trilex, config, ed_jobid):
        raise Exception("Job submit failed")



# =========================================================================== 
# =                          Postprocessing                                 =
# =========================================================================== 

# TODO: clean "idw.dat", "tpri.dat", "varbeta.dat", tmp output, extract data to dir
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
