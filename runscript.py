import sys
import os
import re
import numpy as np
import shutil
from config import *
from helpers import run_bash, check_env, query_yn, reset_dir, dmft_log, \
                    copy_and_edit_dmft, run_ed_dmft, copy_and_edit_vertex, \
                    parse_freq_list, \
                    run_ed_vertex, copy_and_edit_susc, run_ed_susc, \
                    copy_and_edit_trilex, run_ed_trilex, run_postprocess,\
                    copy_and_edit_lDGA_j, run_lDGA_j, \
                    copy_and_edit_w2dyn, run_w2dyn,\
                    read_preprocess_config, grid_pattern

# TODO: obtain compiler and modify clean_script etc
# TODO: IMPORTANT! shift compilation task to job itself (include in jobfile)
# TODO: IMPORTANT! skip computation if jobid indcates completion


def run(config):
    beta_list = []
    U_list = []
    try:
        beta = float(beta_str)
        beta_list = [beta]
    except ValueError:
        beta_list = re.findall("\d+\.\d+",beta_str)
        if len(beta_list) < 3:
            beta_list = re.findall("\d+",beta_str)
        beta_list = list(map(float, beta_list))
        if len(beta_list) < 3:
            print(beta_list)
            raise ValueError("Could not parse beta list, given as " +
                             str(beta_list))
        beta_list = np.arange(beta_list[0], beta_list[2] + beta_list[1]/100,
                              beta_list[1])
        print("Parsed beta list " + str(beta_list) + " and starting scan.")
    try:
        U = float(U_str)
        U_list = [U]
    except ValueError:
        U_list = re.findall("\d+\.\d+",U_str)
        if len(U_list) < 3:
            U_list = re.findall("\d+",U_str)
        U_list = list(map(float, U_list))
        if len(U_list) < 3:
            raise ValueError("Could not parse U list, given as" +str(U_list))
        U_list = np.arange(U_list[0], U_list[2] + U_list[1]/100, U_list[1])
        print("Parsed U list " + str(U_list) + " and starting scan.")

    for beta in beta_list:
        for U in U_list:
            # TODO: generate copy of config for each beta and U, start run
            print("TODO: generate configs for beta x U scan")
            #run_single(config)

def run_single(config, config_path):

    beta_str = config['parameters']['beta']
    U_str = config['parameters']['U']
    print("Running for beta=",beta_str,", U=",U_str)
    # ========================================================================
    # =                            Setup                                     =
    # ========================================================================
    # ------------------------ read input file -------------------------------
    if config['general']['cluster'].lower() == "berlin":
        # exec(open('/usr/share/Modules/init/python.py').read())
        # module('load', 'modulefile', 'modulefile', '...'))
        # TODO: complete check for correctly loaded modules
        if not check_env(config):
            raise RuntimeError("Environment check failed! "
                               "Please check the error log for more "
                               "information")
        config['general']['queue_system'] = "slurm"
        config['general']['submit_str'] = "sbatch "
    elif config['general']['cluster'].lower() == "hamburg":
        config['general']['queue_system'] = "gse"
        config['general']['submit_str'] = "qsub "
    else:
        raise RuntimeError("Environment check failed! "
                           "Unrecognized cluster!")

    # ------------------------ create directories ----------------------------
    runDir = config['general']['runDir']
    print(os.path.abspath(config_path))
    dataDir = os.path.join(runDir, "data")
    if not os.path.exists(runDir):
        os.makedirs(runDir, exist_ok=True)
        print("Directory ", runDir,  " Created ")
    else:
        if not config['general']['auto_continue']:
            reset_flag = query_yn("Directory " + runDir + " already exists. "
                                  "Should everything be reset?", "no")
            if reset_flag:
                confirm = query_yn("This will purge the directory. "
                                   "Do you want to continue?", "no")
                if confirm:
                    reset_dir(runDir)
    config_path = os.path.abspath(config_path)
    config_path_target = os.path.abspath(os.path.join(runDir, "config.toml"))
    if os.path.exists(config_path_target):
        if not os.path.samefile(config_path, config_path_target):
            shutil.copyfile(config_path, config_path_target)
    else:
        shutil.copyfile(config_path, config_path_target)

    # ========================================================================
    # =                             DMFT                                     =
    # ========================================================================

    # -------------------------- definitions ---------------------------------
    subRunDir_ED = os.path.join(runDir, "ed_dmft")
    src_files = ["aux_routines.f90", "lattice_routines.f90",
                 "ed_dmft_parallel_frequencies.f90"]
    if 'custom_dmft_directory' in config['ED'] and config['ED']['custom_dmft_directory'] != '':
        subCodeDir = os.path.join(config['general']['codeDir'], config['ED']['custom_dmft_directory'])
        if 'custom_compile_command' in config['ED'] and config['ED']['custom_compile_command'] != '':
            compile_command = config['ED']['custom_compile_command']
        else:
            compile_command = "mpiifort " + ' '.join(src_files) + \
                          " -o run.x -llapack " + \
                          config['general']['CFLAGS']
    else:
        subCodeDir = os.path.join(config['general']['codeDir'], "ED_codes/ED_dmft")
        compile_command = "mpiifort " + ' '.join(src_files) + \
                          " -o run.x -llapack " + \
                          config['general']['CFLAGS']
    jobid_dmft = None

    if not config['ED']['skip']:
        # ------------------------ save job info -----------------------------
        dmft_logfile = os.path.join(runDir, "job_dmft.log")
        cont = dmft_log(dmft_logfile, jobid_dmft, subRunDir_ED, config)
        if cont:
            # ---------------------- create dir ------------------------------
            if not os.path.exists(subRunDir_ED):
                os.mkdir(subRunDir_ED)

            # ---------------------- copy/edit -------------------------------
            cp_cmd, prev_id = copy_and_edit_dmft(subCodeDir, subRunDir_ED, config)

            # --------------------- compile/run ------------------------------
            if not run_bash(compile_command, cwd=subRunDir_ED,
                           verbose=config['general']['verbose']):
                raise Exception("Compilation Failed")
            jobid_dmft = run_ed_dmft(subRunDir_ED, config, cp_cmd, prev_id)
            if not jobid_dmft:
                raise Exception("Job submit failed")
            if os.path.isfile(dmft_logfile):
                os.remove(dmft_logfile)
            _ = dmft_log(dmft_logfile, jobid_dmft, subRunDir_ED, config)
        else:
            print("Skipping dmft computation, due to completed or active job. "
                  "This behavor can be changed in the config.")

    # ========================================================================
    # =                          DMFT W2Dyn                                  =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    if 'w2dyn' in config and config['w2dyn']['skip'] == False:
        subCodeDir = os.path.abspath(config['w2dyn']['runfile'])
        subRunDir_w2dyn = os.path.join(runDir, "w2dyn")
        jobids_w2dyn = None
        if not os.path.exists(subRunDir_w2dyn):
            os.mkdir(subRunDir_w2dyn)
        # ----------------- save job info --------------------------------
        w2dyn_logfile = os.path.join(runDir, "job_w2dyn.log")
        cont = dmft_log(w2dyn_logfile, jobids_w2dyn, subRunDir_w2dyn, config)
        if cont:
            # ------------------- copy/edit ------------------------------
            copy_and_edit_w2dyn(subRunDir_w2dyn, dataDir, config)
            jobids_w2dyn = run_w2dyn(subRunDir_w2dyn, config)
            if not jobids_w2dyn:
                raise Exception("Job submit failed")
            # ----------------- save job info --------------------------------
            jobid_dmft = jobids_w2dyn[-1]
            w2dyn = os.path.join(runDir, "job_w2dyn.log")
            if os.path.isfile(w2dyn_logfile):
                os.remove(w2dyn_logfile)
            _ = dmft_log(w2dyn_logfile, jobid_dmft, subRunDir_w2dyn, config)
        else:
            print("Skipping fortran DMFT (w2dyn) computation, due to completed or active job. "
                  "This behavor can be changed in the config.")

    # ========================================================================
    # =                           ED Vertex                                  =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    subRunDir_vert = runDir + "/ed_vertex"
    subCodeDir = config['general']['codeDir'] + "/ED_codes/ED_vertex"
    jobid_vert = None

    if not config['Vertex']['skip']:
        # --------------------- save job info --------------------------------
        vert_logfile = os.path.join(runDir, "job_vertex.log")
        cont = dmft_log(vert_logfile, jobid_vert, subRunDir_vert, config)
        if cont:
            # ------------------ create dir ----------------------------------
            if not os.path.exists(subRunDir_vert):
                os.mkdir(subRunDir_vert)

            # ------------------- copy/edit ----------------------------------
            copy_and_edit_vertex(subCodeDir, subRunDir_vert, subRunDir_ED,
                                 dataDir, config)

            # ------------------ compile/run ---------------------------------
            jobid_vert = run_ed_vertex(subRunDir_vert, config, jobid_dmft)
            if not jobid_vert:
                raise Exception("Job submit failed")

            # ----------------- save job info --------------------------------
            vert_logfile = os.path.join(runDir, "job_vertex.log")
            if os.path.isfile(vert_logfile):
                os.remove(vert_logfile)
            dmft_log(vert_logfile, jobid_vert, subRunDir_vert, config)
        else:
            print("Skipping vertex computation, due to completed or active job. "
                  "This behavor can be changed in the config.")

    # ========================================================================
    # =                          DMFT Susc                                   =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'],
                              "ED_codes/ED_physical_suscpetibility")
    compile_command = "gfortran calc_chi_asymptotics_gfortran.f90 -o run.x "\
                      "-llapack " + config['general']['CFLAGS']
    subRunDir_susc = os.path.join(runDir, "ed_susc")
    jobid_susc = None
    cont = True

    if not config['Susc']['skip']:
        # ---------------------- create dir ----------------------------------
        if not os.path.exists(subRunDir_susc):
            os.mkdir(subRunDir_susc)

        # --------------------- save job info --------------------------------
        susc_logfile = os.path.join(runDir, "job_susc.log")
        cont = dmft_log(susc_logfile, jobid_susc, subRunDir_susc, config)
        if cont:
            # ------------------- copy/edit ----------------------------------
            copy_and_edit_susc(subCodeDir, subRunDir_susc, subRunDir_ED,
                               dataDir, config)

            # ------------------ compile/run ---------------------------------
            if not run_bash(compile_command, cwd=subRunDir_susc,
                           verbose=config['general']['verbose']):
                raise Exception("Compilation Failed")
            jobid_susc = run_ed_susc(subRunDir_susc, config, jobid_dmft)
            if not jobid_susc:
                raise Exception("Job submit failed")

            # ----------------- save job info --------------------------------
            susc_logfile = os.path.join(runDir, "job_susc.log")
            if os.path.isfile(susc_logfile):
                os.remove(susc_logfile)
            dmft_log(susc_logfile, jobid_susc, subRunDir_susc, config)
        else:
            print("Skipping susceptibility computation, due to completed or active job."
                  " This behavor can be changed in the config.")

    # ========================================================================
    # =                         DMFT Trilex                                  =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'],
                              "ED_codes/ED_Trilex_Parallel")
    compile_command = "mpifort ver_twofreq_parallel.f -o run.x -llapack " \
                      "-lblas " + config['general']['CFLAGS']
    output_dirs = ["trip_omega", "tripamp_omega", "trilex_omega"]
    subRunDir_trilex = os.path.join(runDir, "ed_trilex")
    jobid_trilex = None

    if not config['Trilex']['skip']:
        # ---------------------- create dir ----------------------------------
        if not os.path.exists(subRunDir_trilex):
            os.mkdir(subRunDir_trilex)
        for d in output_dirs:
            fp = os.path.abspath(os.path.join(subRunDir_trilex, d))
            if not os.path.exists(fp):
                os.mkdir(fp)

        # --------------------- save job info --------------------------------
        trilex_logfile = os.path.join(runDir, "job_trilex.log")
        cont = dmft_log(trilex_logfile, jobid_trilex, subRunDir_trilex, config)
        if cont:
            # ------------------- copy/edit ----------------------------------
            copy_and_edit_trilex(subCodeDir, subRunDir_trilex, subRunDir_ED,
                                 dataDir, config)

            # ------------------ compile/run ---------------------------------
            if not run_bash(compile_command, cwd=subRunDir_trilex,
                           verbose=config['general']['verbose']):
                raise Exception("Compilation Failed")
            jobid_trilex = run_ed_trilex(subRunDir_trilex, config, jobid_dmft)
            if not jobid_trilex:
                raise Exception("Job submit failed")

            # ----------------- save job info --------------------------------
            trilex_logfile = os.path.join(runDir, "job_trilex.log")
            if os.path.isfile(trilex_logfile):
                os.remove(trilex_logfile)
            _ = dmft_log(trilex_logfile, jobid_trilex, subRunDir_trilex,
                         config)
        else:
            print("Skipping trilex computation, due to completed or active job. "
                  "This behavor can be changed in the config.")

    # ========================================================================
    # =                       Postprocessing                                 =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    jobid_pp = None

    # TODO: check all run.err for errors (later also use sacct with job_ids)
    if not config['Postprocess']['skip']:
        pp_logfile = os.path.join(runDir, "job_pp.log")
        cont = dmft_log(pp_logfile, jobid_pp, runDir, config)
        if cont:
            pp_logfile = os.path.join(runDir, "job_pp.log")
            # ---------------------- compile/run ---------------------------------
            jobid_pp = run_postprocess(runDir, dataDir, subRunDir_ED,
                                       subRunDir_vert, subRunDir_susc,
                                       subRunDir_trilex, config, jobids=[
                                        jobid_dmft, jobid_vert, jobid_susc,
                                        jobid_trilex])
            if not jobid_pp:
                raise Exception("Postprocessing job submit failed")
            # ----------------- save job info --------------------------------
            if os.path.isfile(pp_logfile):
                os.remove(pp_logfile)
            _ = dmft_log(pp_logfile, jobid_pp, runDir, config)
        else:
            print("Skipping postprocessing, due to completed or active job. "
                  "This behavor can be changed in the config.")
    # ========================================================================
    # =                          lDGA Julia                                  =
    # ========================================================================

    # ------------------------- definitions ----------------------------------
    subCodeDir = os.path.join(config['general']['codeDir'], "LadderDGA.jl")
    subRunDir_lDGA_j = os.path.join(runDir, "lDGA_julia")
    jobid_lDGA_j = None

    if not config['lDGAJulia']['skip']:
        # ------------------ create dirs ---------------------------------
        if not os.path.exists(subRunDir_lDGA_j):
            os.mkdir(subRunDir_lDGA_j)

        # ----------------- save job info --------------------------------
        lDGA_logfile = os.path.join(runDir, "job_lDGA_j.log")
        cont = dmft_log(lDGA_logfile, jobid_lDGA_j, subRunDir_lDGA_j, config)
        if cont:
            # ------------------- copy/edit ------------------------------
            copy_and_edit_lDGA_j(subRunDir_lDGA_j, dataDir, config)

            # ------------------ compile/run -----------------------------
            jobid_lDGA_j = run_lDGA_j(subRunDir_lDGA_j, dataDir, subCodeDir,
                                             config, jobid_pp)
            if not jobid_lDGA_j:
                raise Exception("Job submit failed")

            # ----------------- save job info ----------------------------
            lDGA_logfile = os.path.join(runDir, "job_lDGA_j.log")
            if os.path.isfile(lDGA_logfile):
                os.remove(lDGA_logfile)
            _ = dmft_log(lDGA_logfile, jobid_lDGA_j,
                         subRunDir_lDGA_j, config)
        else:
            print("Skipping Julia lDGA computation, due to completed or active job."
                  "This behavor can be changed in the config.")

    # ========================================================================
    # =                        lDGA Fortran                                  =
    # ========================================================================

    # ------------------------- definitions ----------------------------------

    compile_command_kl = "gfortran dispersion.f90 make_klist.f90 -llapack -o "\
                         "klist.x"
    compile_command = "make run"
    output_dirs = ["chisp_omega", "chich_omega", "chi_bubble", "klist"]
    subRunDir_lDGA_f = os.path.join(runDir, "lDGA_fortran")
    jobid_lDGA_f = None

    if not config['lDGAFortran']['skip']:
        # ---------------------- create dirs ---------------------------------
        if not os.path.exists(subRunDir_lDGA_f):
            os.mkdir(subRunDir_lDGA_f)
        for d in output_dirs:
            fp = os.path.abspath(os.path.join(subRunDir_lDGA_f, d))
            if not os.path.exists(fp):
                os.mkdir(fp)

        # --------------------- save job info --------------------------------
        lDGA_logfile = os.path.join(runDir, "job_lDGA.log")
        cont = dmft_log(lDGA_logfile, jobid_lDGA_f, subRunDir_lDGA_f, config)
        if cont:
            # ------------------- copy/edit ----------------------------------
            copy_and_edit_lDGA_f(subCodeDir, subRunDir_lDGA_f, dataDir, config)
            if not run_bash(compile_command_kl, cwd=subRunDir_lDGA_f,
                           verbose=config['general']['verbose']):
                raise Exception("Compilation Failed")

            # ------------------ compile/run ---------------------------------
            if not run_bash(compile_command, cwd=subRunDir_lDGA_f,
                           verbose=config['general']['verbose']):
                raise Exception("Compilation Failed")
            jobid_lDGA_f = run_lDGA_f(subRunDir_lDGA_f, config, jobid_pp)
            if not jobid_lDGA_f:
                raise Exception("Job submit failed")

            # ----------------- save job info --------------------------------
            lDGA_logfile = os.path.join(runDir, "job_lDGA.log")
            if os.path.isfile(lDGA_logfile):
                os.remove(lDGA_logfile)
            _ = dmft_log(lDGA_logfile, jobid_lDGA_f, subRunDir_lDGA_f, config)
        else:
            print("Skipping fortran lDGA computation, due to completed or active job. "
                  "This behavor can be changed in the config.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config = read_preprocess_config("config.toml")
        run_single(config, "config.toml")
    else:
        arg_str = sys.argv[1]
        if os.path.isfile(arg_str):
            config = read_preprocess_config(arg_str)
            run_single(config, arg_str)
        elif os.path.isdir(arg_str):
            files = os.listdir(arg_str)
            files.sort()
            for fn in files:
                if fn.endswith(".toml"):
                    fp = os.path.abspath(os.path.join(arg_str, fn))
                    config = read_preprocess_config(fp)
                    run_single(config, fp)
        else:
            raise RuntimeError("Argument provided is not a valid config or "
                               "directory of configs.")
