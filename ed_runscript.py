import sys
import os
import shutil
import subprocess
import toml
import argparse
import numpy as np
from helpers import *



# TODO: obtain compiler and modify clean_script etc
# TODO: IMPORTANT! shift compilation task to job itself (include in jobfile)
# TODO: IMPORTANT! save jobid in file and check on restart if it is still running/exit ok!
# TODO: IMPORTANT! skip computation if jobid indcates completion


print("TODO: skip computation if jobid indcates completion")

def run(config):
    # =========================================================================== 
    # =                               Setup                                     =
    # =========================================================================== 
    # --------------------------- read input file -------------------------------
    if config['general']['cluster'].lower() == "berlin":
        module_cmd = "openblas/gcc.9/0.3.7 impi/2019.5 intel/19.0.5"
        #exec(open('/usr/share/Modules/init/python.py').read())
        #module('load', 'modulefile', 'modulefile', '...'))
        # TODO: complte check for correctly loaded modules
        if not check_env(config):
            raise RuntimeError("Environment check failed! Please check the error log for more information")

    # -------------------------- create directories -----------------------------
    runDir = config['general']['runDir']
    dataDir = os.path.join(config['general']['runDir'], "data")
    if not os.path.exists(runDir):
        os.mkdir(runDir)
        print("Directory " , runDir ,  " Created ")
    else:
        if not config['general']['auto_continue']:
            reset_flag = query_yn("Directory " + runDir +  " already exists. Should everything be reset?", "no")
            if reset_flag:
                confirm = query_yn("This will purge the directory. Do you want to continue?", "no")
                if confirm:
                    reset_dir(runDir)

    # =========================================================================== 
    # =                                DMFT                                     =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "ED_dmft")
    subRunDir_ED = os.path.join(runDir, "ed_dmft")
    src_files  = ["ed_dmft_parallel_frequencies.f"]
    compile_command = "mpif90 " + ' '.join(src_files) + " -o run.x -llapack -lblas " + config['general']['CFLAGS']
    jobid_ed = None

    if not config['ED']['skip']:
        # ----------------------------- create dir ----------------------------------
        if not os.path.exists(subRunDir_ED):
            os.mkdir(subRunDir_ED)

        # ------------------------------ copy/edit ----------------------------------
        copy_and_edit_dmft(subCodeDir, subRunDir_ED, config)

        # ----------------------------- compile/run ---------------------------------
        #TODO: edit parameters in file or change code in order to include as external
        if not compile(compile_command, cwd=subRunDir_ED ,verbose=config['general']['verbose']):
            raise Exception("Compilation Failed")
        jobid_ed = run_ed_dmft(subRunDir_ED, config)
        if not jobid_ed:
            raise Exception("Job submit failed")

        # ---------------------------- save job info --------------------------------
        dmft_logfile = os.path.join(runDir, "job_dmft.log")
        with open(dmft_logfile, 'w') as f:
            f.write(dmft_log(jobid_ed, subRunDir_ED, config))

    # =========================================================================== 
    # =                            DMFT Vertex                                  =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subRunDir_vert = runDir + "/ed_vertex"
    subCodeDir = config['general']['codeDir'] + "/ED_vertex"
    compile_command = "mpif90 ver_tpri_run.f -o run.x -llapack -lblas " + config['general']['CFLAGS']
    jobid_vert = None

    if not config['Vertex']['skip']:
        # ----------------------------- create dir ----------------------------------
        if not os.path.exists(subRunDir_vert):
            os.mkdir(subRunDir_vert)

        # ------------------------------ copy/edit ----------------------------------
        copy_and_edit_vertex(subCodeDir, subRunDir_vert, subRunDir_ED, config)

        # ----------------------------- compile/run ---------------------------------
        if not compile(compile_command, cwd=subRunDir_vert, verbose=config['general']['verbose']):
            raise Exception("Compilation Failed")
        jobid_vert = run_ed_vertex(subRunDir_vert, config, jobid_ed)
        if not jobid_vert:
            raise Exception("Job submit failed")

        # ---------------------------- save job info --------------------------------
        vert_logfile = os.path.join(runDir, "job_vertex.log")
        with open(vert_logfile, 'w') as f:
            f.write(dmft_log(jobid_vert, subRunDir_vert, config))

        #TODO: edit tpri?
        #TODO: edit/compile/run sum_t_files
        #TODO: run split_script

    # =========================================================================== 
    # =                             DMFT Susc                                   =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "ED_physical_suscpetibility")
    compile_command = "gfortran calc_chi_asymptotics_gfortran.f -o run.x -llapack -lblas " + config['general']['CFLAGS']
    subRunDir_susc = os.path.join(runDir, "ed_susc")
    jobid_susc = None

    if not config['Susc']['skip']:
        # ----------------------------- create dir ----------------------------------
        if not os.path.exists(subRunDir_susc):
            os.mkdir(subRunDir_susc)

        # ------------------------------ copy/edit ----------------------------------
        copy_and_edit_susc(subCodeDir, subRunDir_susc, subRunDir_ED, config)

        # ----------------------------- compile/run ---------------------------------
        if not compile(compile_command, cwd=subRunDir_susc, verbose=config['general']['verbose']):
            raise Exception("Compilation Failed")
        jobid_susc = run_ed_susc(subRunDir_susc, config, jobid_ed)
        if not jobid_susc:
            raise Exception("Job submit failed")

        # ---------------------------- save job info --------------------------------
        susc_logfile = os.path.join(runDir, "job_susc.log")
        with open(susc_logfile, 'w') as f:
            f.write(dmft_log(jobid_susc, subRunDir_susc, config))


    # =========================================================================== 
    # =                            DMFT Trilex                                  =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "ED_Trilex_Parallel")
    compile_command = "mpif90 ver_twofreq_parallel.f -o run.x -llapack -lblas " + config['general']['CFLAGS']
    output_dirs = ["trip_omega", "tripamp_omega", "trilex_omega"]
    subRunDir_trilex = os.path.join(runDir, "ed_trilex")
    jobid_trilex = None

    if not config['Trilex']['skip']:
        # ----------------------------- create dir ----------------------------------
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
        jobid_trilex = run_ed_trilex(subRunDir_trilex, config, jobid_ed)
        if not jobid_trilex:
            raise Exception("Job submit failed")

        # ---------------------------- save job info --------------------------------
        trilex_logfile = os.path.join(runDir, "job_vert.log")
        with open(trilex_logfile, 'w') as f:
            f.write(dmft_log(jobid_trilex, subRunDir_trilex, config))


    # =========================================================================== 
    # =                          Postprocessing                                 =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    jobid_pp = None

    # TODO: check all run.err for errors (later also use sacct with job_ids)
    if not config['Postprocess']['skip']:

        # ----------------------------- compile/run ---------------------------------
        jobid_pp = run_postprocess(runDir, dataDir, subRunDir_ED, subRunDir_vert,\
                        subRunDir_susc, subRunDir_trilex, config, jobids=\
                        [jobid_ed, jobid_vert, jobid_susc, jobid_trilex])
        if not jobid_pp:
            raise Exception("Postprocessing job submit failed")


    # =========================================================================== 
    # =                           lDGA Fortran                                  =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    if config['parameters']['Dimensions'] == 2:
        subCodeDir = os.path.join(config['general']['codeDir'], "ladderDGA2D")
    elif config['parameters']['Dimensions'] == 3:
        subCodeDir = os.path.join(config['general']['codeDir'], "ladderDGA3D")
    if config['lDGA']['kInt'].lower() == "fft":
        subCodeDir += "_FFT"

    compile_command_kl = "gfortran dispersion.f90 make_klist.f90 -llapack -o klist.x"
    compile_command = "make run"
    output_dirs = ["chisp_omega", "chich_omega", "chi_bubble", "klist"]
    subRunDir_lDGA_f = os.path.join(runDir, "lDGA_fortran")
    jobid_lDGA_f = None

    if not config['lDGAFortran']['skip']:
        # ----------------------------- create dirs ---------------------------------
        if not os.path.exists(subRunDir_lDGA_f):
            os.mkdir(subRunDir_lDGA_f)
        for d in output_dirs:
            fp = os.path.abspath(os.path.join(subRunDir_lDGA_f, d))
            if not os.path.exists(fp):
                os.mkdir(fp)

        # ------------------------------ copy/edit ----------------------------------
        copy_and_edit_lDGA_f(subCodeDir, subRunDir_lDGA_f, dataDir, config)
        if not compile(compile_command_kl, cwd=subRunDir_lDGA_f, verbose=config['general']['verbose']):
            raise Exception("Compilation Failed")
        jobid_lDGA_f_makeklist = run_lDGA_f_makeklist(subRunDir_lDGA_f, config)
        if not jobid_lDGA_f_makeklist:
            raise Exception("Job submit failed")

        # ----------------------------- compile/run ---------------------------------
        if not compile(compile_command, cwd=subRunDir_lDGA_f, verbose=config['general']['verbose']):
            raise Exception("Compilation Failed")
        jobid_lDGA_f = run_lDGA_f(subRunDir_lDGA_f, config, jobid_pp)
        if not jobid_lDGA_f:
            raise Exception("Job submit failed")

        # ---------------------------- save job info --------------------------------
        lDGA_logfile = os.path.join(runDir, "job_lDGA.log")
        with open(lDGA_logfile, 'w') as f:
            f.write(dmft_log(jobid_lDGA_f, subRunDir_lDGA_f, config))


    # =========================================================================== 
    # =                           lDGA Julia tc                                 =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "ladderDGA_Julia")
    subRunDir_lDGA_j_tc = os.path.join(runDir, "lDGA_julia_tc")
    jobid_lDGA_j_tc = None

    if not config['lDGAJulia']['skip']:
        if config['lDGAJulia']['tail_corrected'].casefold() == "both" or config['lDGAJulia']['tail_corrected'].casefold() == "yes":
            tc = "true"
            postf = "_tc"

            # ----------------------------- create dirs ---------------------------------
            if not os.path.exists(subRunDir_lDGA_j_tc):
                os.mkdir(subRunDir_lDGA_j_tc)

            # ------------------------------ copy/edit ----------------------------------
            copy_and_edit_lDGA_j(subRunDir_lDGA_j_tc, dataDir, config, tc)

            # ----------------------------- compile/run ---------------------------------
            jobid_lDGA_j_tc = run_lDGA_j(subRunDir_lDGA_j_tc, subCodeDir, config, jobid_pp)
            if not jobid_lDGA_j_tc:
                raise Exception("Job submit failed")

            # ---------------------------- save job info --------------------------------
            lDGA_logfile = os.path.join(runDir, "job_lDGA_j"+postf+".log")
            with open(lDGA_logfile, 'w') as f:
                f.write(dmft_log(jobid_lDGA_j_tc, subRunDir_lDGA_j_tc, config))



    # =========================================================================== 
    # =                         lDGA Julia naive                                =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "ladderDGA_Julia")
    subRunDir_lDGA_j_naive = os.path.join(runDir, "lDGA_julia")
    jobid_lDGA_j_naive = None

    if not config['lDGAJulia']['skip']:
        if config['lDGAJulia']['tail_corrected'].casefold() == "both" or config['lDGAJulia']['tail_corrected'].casefold() == "no":
            postf = "_naive"
            tc = "false"

            # ----------------------------- create dirs ---------------------------------
            if not os.path.exists(subRunDir_lDGA_j_naive):
                os.mkdir(subRunDir_lDGA_j_naive)

            # ------------------------------ copy/edit ----------------------------------
            copy_and_edit_lDGA_j(subRunDir_lDGA_j_naive, dataDir, config, tc)

            # ----------------------------- compile/run ---------------------------------
            jobid_lDGA_j_naive = run_lDGA_j(subRunDir_lDGA_j_naive, subCodeDir, config, jobid_pp)
            if not jobid_lDGA_j_naive:
                raise Exception("Job submit failed")

            # ---------------------------- save job info --------------------------------
            lDGA_logfile = os.path.join(runDir, "job_lDGA_j"+postf+".log")
            with open(lDGA_logfile, 'w') as f:
                f.write(dmft_log(jobid_lDGA_j_naive, subRunDir_lDGA_j_naive, config))


    # =========================================================================== 
    # =                              results                                    =
    # =========================================================================== 

    # ---------------------------- definitions ----------------------------------
    subRunDir_results = os.path.join(runDir, "results")
    jobid_results = None
    jobid_pp = run_results_pp(runDir, dataDir, subRunDir_ED, subRunDir_vert,\
                    subRunDir_susc, subRunDir_trilex, config,
                    subRunDir_lDGA_j_tc, subRunDir_lDGA_j_naive,
                    jobids = [jobid_ed, jobid_vert, jobid_susc, jobid_trilex,
                            jobid_lDGA_j_naive, jobid_lDGA_j_tc])

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config = read_preprocess_config("config.toml")
        run(config)
    else:
        arg_str = sys.argv[1]
        if os.path.isfile(arg_str):
            config = read_preprocess_config(arg_str)
            run(config)
        elif os.path.isdir(arg_str):
            for fn in os.listdir(arg_str):
                if fn.startswith("config_"):
                    fp = os.path.abspath(os.path.join(arg_str, fn))
                    config = read_preprocess_config(fp)
                    run(config)
        else:
            raise RuntimeError("Argument provided is not a valid config or directory of configs starting with config_.")


