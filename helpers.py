import os
import shutil
import subprocess
from datetime import datetime
import toml
import re
import stat
from math import isclose, ceil
from file_templates import *
# flake8:  noqa: F405
# from file_templates import , parameters_dat, init_vertex_h,\
#                           bak_files_script, init_susc_h,\
#                           init_trilex_h, ladderDGA_in, q_sum_h, lDGA_julia,\
#                           split_files, tpri_dat, init_h, hubb_dat,\
#                           hubb_andpar


# TODO: refactor replicated code
# ============================================================================
# =                         helper functions                                 =
# ============================================================================

freq_pattern = re.compile(r'(-?\d+):(-?\d+)', re.M)
def match_freq_str(freq_str):
    match_l = re.findall(freq_pattern, freq_str)
    freq_grid = [[],[]]
    if not (len(match_l) == 2):
        raise ValueError("Could not parse frequency grid (did not find two freq \
                ranges). Format should be F1:F2,B1:B2, but got ", freq_str)
    for i,match in enumerate(match_l):
        if not match or (match[0] >= match[1]):
            raise ValueError("Could not parse frequency grid. Format should\
                be F1:F2,B1:B2 with N1 < N2, M1 < M2, but got ", freq_str)
        freq_grid[i] = [int(match[0]), int(match[1])+1]           # +1, we want to include upper lim
    return freq_grid

def format_log_from_sacct(fn, jobid, loc):
    out = """
jobid = {0}
result_dir = {1}
last_check_stamp = {2}
last_status = {3}
run_time = {4}
job_name = {5}
    """
    job_cmd = "sacct -j " + str(jobid) + " --format=User,JobID,Jobname,"\
              "partition,state,elapsed,nnodes,ncpus,nodelist"
    process = subprocess.run(job_cmd, shell=True, capture_output=True)
    stdout = process.stdout.decode("utf-8")
    stderr = process.stderr.decode("utf-8")
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    status = ""

    if not (process.returncode == 0):
        out = out.format(jobid, os.path.abspath(loc), timestamp,
                         "sacct not accessible", "sacct not accessible",
                         "sacct not accessible")
        res = stderr
        print("Warning: could not run sacct in order to check job completion! "
              "Got: \n" + res)
    else:
        if len(stdout.splitlines()) < 3:
            out = out.format(jobid, os.path.abspath(loc), timestamp,
                             "Job not found", "Job not found", "Job not found")
            status = ""
        else:
            res = list(map(str.strip, stdout.splitlines()[2].split()))
            out = out.format(jobid, os.path.abspath(loc), timestamp,
                             res[4], res[5], res[8])
            status = res[4]
    with open(fn, 'w') as f:
        f.write(out)
    return out, status


def check_config_consistency(config):
    mu = config['parameters']['mu']
    U = config['parameters']['U']
    if not isclose(mu, U / 2.0):
        print("Not Calculating at half filling! mu = {0} and U = {1} Make sure"
              " that bath is not forced to be symmetric.".format(
                  mu, U / 2.0
              ))

def check_andpar_result(config, andpar_lines):
    ns = config['parameters']['ns']
    eps = np.zeros(ns)
    tpar = np.zeros(ns)
    for i in range(ns):
        eps[i] = float(andpar_lines[9+i])
    for i in range(ns):
        tpar[i] = float(andpar_lines[ns+1+9+i])
    eps_ssum = np.sum(eps**2)
    tpar_ssum = np.sum(tpar**2)
    checks_success = [True, True, True, True]
    checks_success[0] = abs(eps_ssum*0.25 - tpar_ssum) <= config['ED']['square_sum_diff']
    for i in range(ns):
        for j in range(i+1,ns):
            if abs(eps[i] - eps[j]) <= config['ED']['bathsite_cancel_eps']:
                checks_success[1] = False
        if abs(tpar[i]) <= config['ED']['bathsite_cancel_V']:
            checks_success[2] = False
        if abs(eps[i]) >= (1.0/config['parameters']['beta'])*config['ED']['bathsite_cancel_V']:
            checks_success[3] = False
    return checks_success


def read_preprocess_config(config_string):
    with open(config_string, 'r') as f:
        config_in = f.read()
    config = toml.loads(config_in)
    for k in config['parameters'].keys():
        config_in = config_in.replace("{"+k+"}",
                                      str(config['parameters'][k]))
        config_in = config_in.replace("{"+k.upper()+"}",
                                      str(config['parameters'][k]))
        config_in = config_in.replace("{"+k.lower()+"}",
                                      str(config['parameters'][k]))
    config = toml.loads(config_in)
    config['general']['codeDir'] = os.path.abspath(os.path.expanduser(
                                   config['general']['codeDir']))
    if str(config['parameters']['mu']).lower() == "hf":
        config['parameters']['mu'] = config['parameters']['U']/2.0
    check_config_consistency(config)
    return config


def query_yn(question, default="yes"):
    valid = {"yes": True, "y": True,
             "no": False, "n": False}
    if default is None:
        prompt = " y/n "
    elif default == "yes":
        prompt = " [y]/n "
    elif default == "no":
        prompt = " y/[n] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        os.sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            os.sys.stdout.write("Incorrect input! Please provide y/n answer\n")


def reset_dir(dirName):
    for filename in os.listdir(dirName):
        file_path = os.path.join(dirName, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def run_bash(command, cwd, verbose=False):
    if verbose:
        print("running command:\n" + command)
    process = subprocess.run(command, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Execution did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    return True


def is_dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def check_env(config):
    try:
        import os                       # noqa: F401
        import sys                      # noqa: F401
        import shutil                   # noqa: F401
        import pandas as pd             # noqa: F401
        import pandas.io.common         # noqa: F401
        import pyarrow as pa            # noqa: F401
        import pyarrow.parquet as pq    # noqa: F401
        import tarfile                  # noqa: F401
        from scipy.special import comb  # noqa: F401
    except ImportError:
        print("Environment check failed with: ", sys.exc_info()[0])
        return False
    if (not config['Postprocess']['split']) and \
       (not config['lDGAFortran']['skip']):
        print("WARNING: Splitting of ED results deactivated, but needed for "
              "lDGA code!")
        return False
    if (not config['Postprocess']['split']) and \
       (not config['lDGAJulia']['skip']):
        print("WARNING: Splitting of ED results deactivated, but needed for "
              "lDGA code!")
        return False
    return True


# ============================================================================
# =                          copy functions                                  =
# ============================================================================
def copy_and_edit_dmft(subCodeDir, subRunDir_ED, config):
    files_list = ["tpri.dat", "init.h", "init_2.h", "hubb.dat", "hubb.andpar"]
    src_files = ["ed_dmft_parallel_frequencies.f"]

    old_andpar = config["general"]["custom_init_andpar_file"]
    if old_andpar:
        source_file_path = os.path.abspath(old_andpar)
        target_file_path = os.path.abspath(os.path.join(subRunDir_ED,
                                                        "hubb.andpar"))
        if not config["general"]["custom_init_andpar_vals_only"]:
            print("copying hubb.andpar but not checking for consistency!!")
            shutil.copyfile(source_file_path, target_file_path)
        else:
            with open(source_file_path, 'r') as f:
                andpar_string = f.read()
            start_eps = andpar_string.find("Eps(k)") + 7
            start_tpar = andpar_string.find("tpar(k)") + 8
            eps_str = andpar_string[start_eps:(start_tpar-9)]
            andpar_lines = andpar_string[start_tpar:].splitlines()
            tpar_str = "\n".join(andpar_lines[:len(eps_str.splitlines())])
            tpar_str += "\n"

    for fn in files_list:
        fp = os.path.abspath(os.path.join(subRunDir_ED, fn))
        with open(fp, 'w') as f:
            if fn == "hubb.andpar" and old_andpar:
                f.write(globals()[fn.replace(".", "_")](config, eps_str,
                                                        tpar_str))
            else:
                f.write(globals()[fn.replace(".", "_")](config))

    for src_file in src_files:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, src_file))
        target_file_path = os.path.abspath(os.path.join(subRunDir_ED,
                                                        src_file))
        shutil.copyfile(source_file_path, target_file_path)


def copy_and_edit_vertex(subCodeDir, subRunDir, subRunDir_ED, dataDir, config):
    files_dmft_list = ["hubb.andpar", "tpri.dat", "zpart.dat"]#, , "gm_wim""hubb.dat", "gm_wim"]
    src_files_list = ["cleanup.sh","inversion_pp_fotso.f90",  "split_script","ver_tpri_run.f90"]
    scripts = ["copy_dmft_files", "copy_data_files", "checks.py",
               "cleanup.sh", "split_script"]
    files_list = ["hubb.dat", "parameters.dat", "init_vertex.h"]
    for fn in files_list:
        fp = os.path.abspath(os.path.join(subRunDir, fn))
        with open(fp, 'w') as f:
            f.write(globals()[fn.replace(".", "_")](config, mode="Vertex"))

    fp = os.path.join(subRunDir, "copy_dmft_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))
    fp = os.path.join(subRunDir, "copy_data_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(dataDir, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))

    freq_str = config['Vertex']['freqList']
    target_file_path = os.path.abspath(os.path.join(subRunDir, "freqList.dat"))
    if os.path.exists(freq_str):
        shutil.copyfile(os.path.abspath(freq_str), target_file_path)
        with open(freq_str) as f:
            f.readline()
            f.readline()
            var_line = f.readline()
            nFreq = list(map(int,var_line.split()))[0]
            max_freq = list(map(int,var_line.split()))[1]
    else:
        freq_grid = match_freq_str(freq_str)
        with open(target_file_path, "w") as fp:
            nFreq, freq_str = parse_freq_list(freq_grid)
            fp.write(freq_str)
        max_freq = 2*len(range(*freq_grid[0]))+len(range(*freq_grid[1]))+5
    fp = os.path.abspath(os.path.join(subRunDir, "freq_list.h"))
    with open(fp, 'w') as f:
        f.write(freq_list_h(config, nFreq, max_freq))
    checks_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "checks.py")
    checks_py_target = os.path.abspath(os.path.join(subRunDir, "checks.py"))
    shutil.copyfile(checks_py_path, checks_py_target)

    for filename in src_files_list:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, filename))
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)


def copy_and_edit_susc(subCodeDir, subRunDir, subRunDir_ED, dataDir, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["calc_chi_asymptotics_gfortran.f", "idw.dat",
                  "tpri.dat"]
    scripts = ["copy_dmft_files", "copy_data_files"]
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_susc_h(config))
    fp = os.path.join(subRunDir, "copy_dmft_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))
    fp = os.path.join(subRunDir, "copy_data_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(dataDir, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))

    for filename in files_list:
        source_file_path = os.path.join(subCodeDir, filename)
        target_file_path = os.path.join(subRunDir, filename)
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.join(subRunDir, filename)
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)


def copy_and_edit_trilex(subCodeDir, subRunDir, subRunDir_ED, dataDir, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["ver_twofreq_parallel.f", "idw.dat",
                  "tpri.dat"]
    scripts = ["copy_dmft_files", "copy_data_files"]
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_trilex_h(config))
    fp = os.path.join(subRunDir, "copy_dmft_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))
    fp = os.path.join(subRunDir, "copy_data_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(dataDir, subRunDir,
                                 files_dmft_list, header=True, mode="cp"))

    for filename in files_list:
        source_file_path = os.path.join(subCodeDir, filename)
        target_file_path = os.path.join(subRunDir, filename)
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.join(subRunDir, filename)
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)


def copy_and_edit_lDGA_f(subCodeDir, subRunDir, dataDir, config):
    input_files_list = ["gm_wim", "g0mand"]
    vertex_input = ["chi_dir", "gamma_dir"]
    src_files = ["calc_susc.f90", "dispersion.f90", "lambda_correction.f90",
                 "make_klist.f90", "read.f90", "Selfk_LU_parallel.f90",
                 "sigma.f90", "vardef.f90", "write.f90", "makefile"]
    rm_files = ["lambda_correction_sp.dat", "lambda_correction_ch.dat"]

    for src_file in src_files:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, src_file))
        target_file_path = os.path.abspath(os.path.join(subRunDir, src_file))
        if src_file == "make_klist.f90":
            with open(source_file_path, 'r') as f:
                lines = f.readlines()
            with open(target_file_path, 'w') as f:
                lines[3] = "INTEGER, PARAMETER :: k_range=" + \
                           str(config['lDGA']['k_range']) + "\n"
                f.write("".join(lines))
        else:
            shutil.copyfile(source_file_path, target_file_path)
    copy_file = os.path.abspath(os.path.join(subRunDir, "copy.sh"))
    copy_content = "#!/bin/bash\ncp " + os.path.abspath(dataDir) + "/{"
    for f in input_files_list:
        copy_content += f + ","
    copy_content = copy_content[:-1] + "} " + os.path.abspath(subRunDir) + "\n"
    copy_content += "cp " + os.path.abspath(os.path.join(dataDir)) + "/{"
    for d in vertex_input:
        copy_content += d + ","
    copy_content = copy_content[:-1] + "} " + os.path.abspath(subRunDir)
    copy_content += "/ -r\n"
    for rm_file in rm_files:
        fp = os.path.abspath(os.path.join(subRunDir, rm_file))
        copy_content += "rm -f " + fp + " \n"

    with open(copy_file, 'w') as f:
        f.write(copy_content)
    st = os.stat(copy_file)
    os.chmod(copy_file, st.st_mode | stat.S_IEXEC)

    lDGA_in = ladderDGA_in(config)
    lDGA_in_path = os.path.abspath(os.path.join(subRunDir, "ladderDGA.in"))
    with open(lDGA_in_path, 'w') as f:
        f.write(lDGA_in)

    q_sum = q_sum_h(config)
    q_sum_path = os.path.abspath(os.path.join(subRunDir, "q_sum.h"))
    with open(q_sum_path, 'w') as f:
        f.write(q_sum)


def copy_and_edit_lDGA_j(subRunDir, dataDir, config, tc):
    lDGA_in = lDGA_julia(config, os.path.abspath(dataDir), tc)
    lDGA_in_path = os.path.abspath(os.path.join(subRunDir, "config.toml"))
    with open(lDGA_in_path, 'w') as f:
        f.write(lDGA_in)


# ============================================================================
# =                           run functions                                  =
# ============================================================================
def run_ed_dmft(cwd, config):
    fp = cwd + "/" + "ed_dmft_run.sh"
    cmd = "mpirun ./run.x > run.out 2> run.err"
    procs = (config['parameters']['ns']+1)**2
    cslurm = config['general']['custom_slurm_lines']
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, procs, cslurm, cmd, copy_from_ed=False))
    run_cmd = "sbatch ./ed_dmft_run.sh"
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)

    print("running: " + run_cmd)
    if not (process.returncode == 0):
        print("ED submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_ed_vertex(cwd, config, ed_jobid=None):
    filename = "ed_vertex_run.sh"
    fp = os.path.join(cwd, filename)
    if config['general']['cluster'] == "berlin":
        cores_per_node = 96
        procs = config['Vertex']['nprocs']
        nodes = ceil(procs/cores_per_node)
    else:
        print("WARNING: unrecognized cluster configuration!")
        procs = config['Vertex']['nprocs']
    cmd = "echo \"--- start checks ---- \n\" > run.out\n"
    cmd+= "~/.conda/envs/p3/bin/python checks.py > run.out\n"
    cmd+= "echo \"--- end checks ---- \n\" > run.out\n"
    cmd+= "mpiifort ver_tpri_run.f90 -o run.x -mkl " + config['general']['CFLAGS']+"\n"
    cmd+= "mpirun -np " + str(procs) + " ./run.x > run.out 2> run.err"
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " + run_cmd)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, procs, cslurm, cmd))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Vertex submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_ed_susc(cwd, config, ed_jobid=None):
    filename = "ed_susc_run.sh"
    fp = os.path.join(cwd, filename)
    cmd = "./run.x > run.out 2> run.err"
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " + run_cmd)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, 1, cslurm, cmd))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Vertex submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_ed_trilex(cwd, config, ed_jobid=None):
    filename = "ed_trilex_run.sh"
    cmd = "mpirun ./run.x > run.out 2> run.err"
    procs = 2*int(config['Trilex']['nBoseFreq']) + 1
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " + run_cmd)
    fp = os.path.join(cwd, filename)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, procs, cslurm, cmd))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)

    if not (process.returncode == 0):
        print("Trilex submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_postprocess(cwd, dataDir, subRunDir_ED, subRunDir_vert,
                    subRunDir_susc, subRunDir_trilex, config, jobids=None):
    filename = "postprocess.sh"
    cslurm = config['general']['custom_slurm_lines']
    # procs = 1
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    fp_config = os.path.join(dataDir, "config.toml")
    with open(fp_config, 'w') as f:
        toml.dump(config, f)

    full_remove_script = "rm " + os.path.abspath(subRunDir_ED) + " " + \
                         os.path.abspath(subRunDir_vert) +\
                         " " + os.path.abspath(subRunDir_susc) + " " +\
                         os.path.abspath(subRunDir_trilex) + " -r\n"
    full_remove_script += "rm " + os.path.abspath(os.path.join(cwd, "*.sh")) +\
                          " " + os.path.abspath(os.path.join(cwd, "*.log")) +\
                          " -r\n"
    cp_script = build_collect_data(dataDir, subRunDir_ED, subRunDir_vert,
                                   subRunDir_susc, subRunDir_trilex,
                                   mode=config['Postprocess']['data_bakup'])
    cp_script_path = os.path.abspath(os.path.join(cwd, "copy_data.sh"))
    with open(cp_script_path, 'w') as f:
        f.write(cp_script)
    split_script = split_files(config)
    split_script_path = os.path.abspath(os.path.join(dataDir,
                                                     "split_files.sh"))
    with open(split_script_path, 'w') as f:
        f.write(split_script)
    st = os.stat(split_script_path)
    os.chmod(split_script_path, st.st_mode | stat.S_IEXEC)

    storage_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "storage_io.py")
    storage_py_target = os.path.abspath(os.path.join(cwd, "storage_io.py"))
    print("copy from " + storage_py_path + " to " + storage_py_target)
    shutil.copyfile(storage_py_path, storage_py_target)
    data_path = os.path.abspath(os.path.join(cwd, "data"))
    clean_script_path = os.path.abspath(os.path.join(subRunDir_vert,
                                                     "clean_script_auto"))
    #content = "cp " + os.path.join(subRunDir_ED, "gm_wim") + " " +\
    #          subRunDir_vert + "\n"
    content = str(clean_script_path) + "\n"
    content += str(cp_script_path) + "\n"
    outf_lst = config['Postprocess']['output_format'].split(',')
    if 'text' in map(str.strip, outf_lst):
        content += str(split_script_path) + "\n"
    conda_env = config['general']['custom_conda_env']
    if conda_env:
        content += "eval \"$(conda shell.bash hook)\"\n"
        content += "conda activate " + conda_env + "\n"
    content += "python storage_io.py " + data_path + " " +\
        config['Postprocess']['output_format'].replace(" ", "") + " && "
    content += "rm storage_io.py \n"
    if config['Postprocess']['keep_only_data']:
        content += full_remove_script

    st = os.stat(cp_script_path)
    os.chmod(cp_script_path, st.st_mode | stat.S_IEXEC)
    st = os.stat(clean_script_path)
    os.chmod(clean_script_path, st.st_mode | stat.S_IEXEC)
    st = os.stat(split_script_path)
    os.chmod(split_script_path, st.st_mode | stat.S_IEXEC)

    run_cmd = "sbatch " + filename
    if jobids and any(jobids):
        run_cmd = "sbatch" + " --dependency=afterok"
        for j in jobids:
            if j:
                run_cmd += ":"+j
        run_cmd += " " + filename
    print("running: " + run_cmd)

    fp = os.path.join(cwd, filename)
    with open(fp, 'w') as f:
        job_func = globals()["postprocessing_" + config['general']['cluster']]
        f.write(job_func(content, cslurm, config))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)

    if not (process.returncode == 0):
        print("Postprocessing submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_lDGA_f_makeklist(cwd, config, jobid=None):
    filename = "klist.sh"
    fp = os.path.join(cwd, filename)
    cslurm = config['general']['custom_slurm_lines']
    run_cmd = "sbatch " + filename
    print("running: " + run_cmd)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, 1, cslurm, cmd))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("klist submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_lDGA_f(cwd, config, jobid=None):
    filename = "lDGA_f.sh"
    fp = os.path.join(cwd, filename)
    procs = 2*int(config['Trilex']['nBoseFreq']) + 1
    cmd = "export LD_LIBRARY_PATH=/sw/numerics/fftw3/impi/intel/3.3.8/skl/"\
          "lib:$LD_LIBRARY_PATH\n"

    cmd += "./klist.x > run_klist.out 2> run_klist.err\n"
    cmd += "rm -f *.dat\n"
    cmd += "./copy.sh\nmpirun -np " + str(procs) + " ./Selfk_LU_parallel_3D.x"\
           " > run.out 2> run.err"
    cslurm = config['general']['custom_slurm_lines']
    if not jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+jobid + " " + filename
    print("running: " + run_cmd)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, procs, cslurm, cmd, False))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("lDGA submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_lDGA_j(cwd, codeDir, config, jobid=None):
    filename = "lDGA_j.sh"
    fp = os.path.join(cwd, filename)
    # q_number = config['lDGA']['LQ']
    # + str(procs)
    procs = 1
    cmd = "julia " + " " + codeDir + "/src/ladderDGA_Julia.jl > run.out 2> "\
          "run.err"
    cslurm = config['general']['custom_slurm_lines']
    if not jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+jobid + " " + filename
    print("running: " + run_cmd)
    with open(fp, 'w') as f:
        job_func = globals()["job_" + config['general']['cluster']]
        f.write(job_func(config, procs, cslurm, cmd, False))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Julia lDGA submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


def run_results_pp(runDir, dataDir, subRunDir_ED, subRunDir_vert,
                   subRunDir_susc, subRunDir_trilex, config,
                   subRunDir_lDGA_j_tc, subRunDir_lDGA_j_naive, jobids=None):
    filename = "results_pp.sh"
    cslurm = config['general']['custom_slurm_lines']
    # procs = 1
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    full_remove_script = "rm " + os.path.abspath(subRunDir_lDGA_j_tc) +\
                         "/vars.jl\n" +\
                         "rm " + os.path.abspath(subRunDir_lDGA_j_naive) +\
                         "/vars.jl\n"
    return
    # TODO: copy chi from julia and fortran
    content = "cp " + " " + "\n"
    content += str(clean_script_path) + "\n"
    content += str(cp_script_path) + "\n"
    outf_lst = config['Postprocess']['output_format'].split(',')
    if 'text' in map(str.strip, outf_lst):
        content += str(split_script_path) + "\n"
    conda_env = config['general']['custom_conda_env']
    if conda_env:
        content += "eval \"$(conda shell.bash hook)\"\n"
        content += "conda activate " + conda_env + "\n"
    content += "python storage_io.py " + data_path + " " +\
        config['Postprocess']['output_format'].replace(" ", "") + " && "
    content += "rm storage_io.py \n"
    if config['Postprocess']['keep_only_data']:
        content += full_remove_script

    st = os.stat(cp_script_path)
    os.chmod(cp_script_path, st.st_mode | stat.S_IEXEC)
    st = os.stat(clean_script_path)
    os.chmod(clean_script_path, st.st_mode | stat.S_IEXEC)
    st = os.stat(split_script_path)
    os.chmod(split_script_path, st.st_mode | stat.S_IEXEC)

    run_cmd = "sbatch " + filename
    if jobids and any(jobids):
        run_cmd = "sbatch" + " --dependency=afterok"
        for j in jobids:
            if j:
                run_cmd += ":"+j
        run_cmd += " " + filename
    print("running: " + run_cmd)

    fp = os.path.join(cwd, filename)
    with open(fp, 'w') as f:
        job_func = globals()["postprocessing_" + config['general']['cluster']]
        f.write(job_func(content, cslurm, config))
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Postprocessing submit did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    else:
        res = process.stdout.decode("utf-8")
        jobid = re.findall(r'job \d+', res)[-1].split()[1]
    return jobid


# ============================================================================
# =                   Log and Postprocess Functions                          =
# ============================================================================
def dmft_log(fn, jobid, loc, config):
    old_id = None
    continue_status = True

    if os.path.exists(fn):               # job has been run before
        with open(fn, 'r') as f:
            old_file = f.readlines()
        try:
            old_id = int(old_file[1][8:])
            out, old_status = format_log_from_sacct(fn, old_id, loc)
        except ValueError:
            old_id = None
            old_status = ""
        if jobid is None:                # determine previous status of job
            if not(old_id is None):
                out, status = format_log_from_sacct(fn, old_id, loc)
                if ((not config["general"]["restart_after_success"]) and
                   old_status == "COMPLETED") or\
                   (old_status == "RUNNING"):
                    continue_status = False
        else:                            # compare if job ids match
            if old_id is None:
                out, status = format_log_from_sacct(fn, jobid, loc)
            else:
                if (int(jobid) == int(old_id)) or\
                   ((not config["general"]["restart_after_success"]) and
                   old_status == "COMPLETED") or\
                   (old_status == "RUNNING"):
                    continue_status = False
                    out, status = format_log_from_sacct(fn, old_id, loc)
    else:                           # this is a new job
        if not (jobid is None):     # we will write a log once the id is known
            out, status = format_log_from_sacct(fn, jobid, loc)
    return continue_status


def build_collect_data(target_dir, dmft_dir, vertex_dir, susc_dir, trilex_dir,
                       mode):
    dmft_files = ["hubb.dat", "hubb.andpar", "g0m", "g0mand", "gm_wim"]
    susc_files = ["chi_asympt", "matrix"]
    vertex_files = ["t.tar.gz", "parameters.dat", "GAMMA_DM_FULLRANGE", "F_DM",
                    "vert_chi"]
    trilex_dirs = ["tripamp_omega", "trip_omega", "trilex_omega"]

    copy_script_str = bak_files_script(dmft_dir, target_dir, dmft_files,
                                       header=True, mode=mode)
    copy_script_str += bak_files_script(susc_dir, target_dir, susc_files,
                                        header=False, mode=mode)
    copy_script_str += bak_files_script(vertex_dir, target_dir, vertex_files,
                                        header=False, mode=mode)
    copy_script_str += bak_dirs_script(trilex_dir, target_dir, trilex_dirs,
                                       header=False, mode=mode)
    return copy_script_str


def cleanup(config):
    pass


def consistency_checks(config):
    print("WARNING: consistency checks not implemented yet.")
    # TODO: check hubb.andpar for equal energies
    # TODO: hopping 0
    # TODO: check if all 8 ed_vertex are present
    pass
