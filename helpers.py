import os
import sys
import shutil
import subprocess
from datetime import datetime
#import pathlib
import toml
import re
import stat
from math import isclose
from file_templates import *

   

# =========================================================================== 
# =                         helper functions                                =
# =========================================================================== 

def check_config_consistency(config):
    if not isclose(config['parameters']['mu'], config['parameters']['U']/2.0):
        print("Not Calculating at half filling! mu = {0} and U = {1} Make sure \
              that bath is not forced to be symmetric.".format(
                  config['parameters']['mu'], config['parameters']['U']/2.0
              ))
    if config['ED']['nprocs'] > 130:
        print("Warning! Number of processors for ED DMFT is larger than 130,\
              this sometimes leads to uneaxpected behavior!")

def read_preprocess_config(config_string):
    with open(config_string, 'r') as f:
            config_in = f.read()
    config = toml.loads(config_in)
    for k in config['parameters'].keys():
        config_in = config_in.replace("{"+k+"}", str(config['parameters'][k]))
        config_in = config_in.replace("{"+k.upper()+"}", str(config['parameters'][k]))
        config_in = config_in.replace("{"+k.lower()+"}", str(config['parameters'][k]))
    config = toml.loads(config_in)
    config['general']['codeDir'] = os.path.abspath(os.path.expanduser(config['general']['codeDir']))
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


def compile(command, cwd, verbose=False):
    if verbose:
        print("running command:\n" + command)
    process = subprocess.run(command, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Compilation did not work as expected:")
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
        import os
        import sys
        import shutil
        import pandas as pd
        import pandas.io.common
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tarfile
        from scipy.special import comb
    except:
        print("Environment check failed with: ", sys.exc_info()[0])
        return False
    if (not config['Postprocess']['split']) and (not config['lDGAFortran']['skip']):
        print("WARNING: Splitting of ED results deactivated, but needed for lDGA code!")
        return False
    return True

# =========================================================================== 
# =                          copy functions                                 =
# =========================================================================== 

def copy_and_edit_dmft(subCodeDir, subRunDir_ED, config):
    files_list = ["tpri.dat", "init.h", "hubb.dat", "hubb.andpar"]
    src_files  = ["ed_dmft_parallel_frequencies.f"]

    old_andpar = config["general"]["custom_init_andpar_file"]
    if old_andpar:
        source_file_path = os.path.abspath(old_andpar)
        target_file_path = os.path.abspath(os.path.join(subRunDir_ED, "hubb.andpar"))
        if not config["general"]["custom_init_andpar_vals_only"]:
            print("copying hubb.andpar but not checking for consistency!!")
            shutil.copyfile(source_file_path, target_file_path)
        else:
            with open(source_file_path, 'r') as f:
                andpar_string = f.read()
            start_eps = andpar_string.find("Eps(k)") + 7
            start_tpar = andpar_string.find("tpar(k)") + 8
            eps_str = andpar_string[start_eps:(start_tpar-9)]
            tpar_str = "\n".join(andpar_string[start_tpar:].splitlines()[:len(eps_str.splitlines())])
            tpar_str += "\n"

    for fn in files_list:
        fp = os.path.abspath(os.path.join(subRunDir_ED, fn))
        with open(fp, 'w') as f:
            if fn == "hubb.andpar" and old_andpar:
                f.write(globals()[fn.replace(".","_")](config, eps_str, tpar_str))
            else:
                f.write(globals()[fn.replace(".","_")](config))

    for src_file in src_files:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, src_file))
        target_file_path = os.path.abspath(os.path.join(subRunDir_ED, src_file))
        shutil.copyfile(source_file_path , target_file_path)




def copy_and_edit_vertex(subCodeDir, subRunDir, subRunDir_ED, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["checksum_script", "clean_script_auto", "idw.dat",
                  "inversion_pp_fotso.f90",  "split_script", "sum_t_files.f",
                   "tpri.dat", "varbeta.dat", "ver_tpri_run.f"]
    scripts = ["copy_ed_files" ,"call_script", "checksum_script", "clean_script_auto", "split_script"]

    fp = os.path.join(subRunDir, "call_script")
    with open(fp, 'w') as f:
        f.write(call_script(config))
    fp = os.path.join(subRunDir, "parameters.dat")
    with open(fp, 'w') as f:
        f.write(parameterts_dat(config))
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_vertex_h(config))
    fp = os.path.join(subRunDir, "init_sumt.h")
    with open(fp, 'w') as f:
        f.write(init_sumt_h(config))
    fp = os.path.join(subRunDir, "copy_ed_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir,
                                  files_dmft_list, header=True, mode="cp"))
    for filename in files_list:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, filename))
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)
    
def copy_and_edit_susc(subCodeDir, subRunDir, subRunDir_ED, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["calc_chi_asymptotics_gfortran.f", "idw.dat",
                   "tpri.dat", "varbeta.dat"]
    scripts = ["copy_ed_files"]
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_psc_h(config))
    fp = os.path.join(subRunDir, "copy_ed_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir, \
                                  files_dmft_list, header=True, mode="cp"))

    for filename in files_list:
        source_file_path = os.path.join(subCodeDir, filename)
        target_file_path = os.path.join(subRunDir, filename)
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.join(subRunDir, filename)
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)

    
def copy_and_edit_trilex(subCodeDir, subRunDir, subRunDir_ED, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["ver_twofreq_parallel.f", "idw.dat",
                   "tpri.dat", "varbeta.dat"]
    scripts = ["copy_ed_files"]
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_trilex_h(config))
    fp = os.path.join(subRunDir, "copy_ed_files")
    with open(fp, 'w') as f:
        f.write(bak_files_script(subRunDir_ED, subRunDir,\
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

    for src_file in src_files:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, src_file))
        target_file_path = os.path.abspath(os.path.join(subRunDir, src_file))
        if src_file == "make_klist.f90":
            with open(source_file_path, 'r') as f:
                lines = f.readlines()
            with open(target_file_path, 'w') as f:
                lines[3] = "INTEGER, PARAMETER :: k_range=" + str(config['lDGAFortran']['k_range']) + "\n"
                f.write("".join(lines))
        else:
            shutil.copyfile(source_file_path , target_file_path)

    copy_content = "#!/bin/bash\ncp " + os.path.abspath(dataDir) + "/{"
    for f in input_files_list:
        copy_content += f + ","
    copy_content = copy_content[0:-1] + "} " + os.path.abspath(subRunDir) + "\n"
    copy_content += "cp " + os.path.join(dataDir) + "/{"

    for d in vertex_input:
        copy_content += d + ","
    copy_content = copy_content[0:-1] + "} " + os.path.abspath(subRunDir) + "/ -r\n"
    with open( os.path.abspath(os.path.join(subRunDir,"copy.sh")), 'w') as f:
        f.write(copy_content)

    lDGA_in = ladderDGA_in(config)
    lDGA_in_path = os.path.abspath(os.path.join(subRunDir, "ladderDGA.in"))
    with open(lDGA_in_path, 'w') as f:
        f.write(lDGA_in)



# =========================================================================== 
# =                           run functions                                 =
# =========================================================================== 

def run_ed_dmft(cwd, config):
    fp = cwd + "/" + "ed_dmft_run.sh"
    cmd= "mpirun ./run.x > run.out 2> run.err"
    procs = int(config['ED']['nprocs'])
    cslurm = config['general']['custom_slurm_lines']
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, procs, cslurm, cmd, copy_from_ed=False))
    run_cmd = "sbatch ./ed_dmft_run.sh"
    process = subprocess.run(run_cmd, cwd=cwd, shell=True, capture_output=True)

    print("running: " +run_cmd)
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
    cmd= "./call_script > run.out 2> run.err"
    procs = (2*int(config['Vertex']['nBoseFreq']) - 1)
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " +run_cmd)
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, procs, cslurm, cmd))
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
    cmd= "./run.x > run.out 2> run.err"
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " +run_cmd)
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, 1, cslurm, cmd))
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
    cmd= "mpirun ./run.x > run.out 2> run.err"
    procs = 2*int(config['Trilex']['nBoseFreq']) - 1
    cslurm = config['general']['custom_slurm_lines']
    if not ed_jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+ed_jobid + " " + filename
    print("running: " +run_cmd)
    fp = os.path.join(cwd, filename)
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, procs, cslurm, cmd))
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


def run_postprocess(cwd, dataDir, subRunDir_ED, subRunDir_vert,\
                    subRunDir_susc, subRunDir_trilex, config, jobids=None):
    filename = "postprocess.sh"
    cslurm = config['general']['custom_slurm_lines']
    procs = 1
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    fp_config = os.path.join(dataDir, "config.toml")
    with open(fp_config, 'w') as f:
        toml.dump(config, f)

    full_remove_script = "rm " + os.path.abspath(subRunDir_ED) +" "+ \
            os.path.abspath(subRunDir_vert) +\
            " " +  os.path.abspath(subRunDir_susc) + " " +\
            os.path.abspath(subRunDir_trilex) + " -r\n"
    full_remove_script += "rm " + os.path.abspath(os.path.join(cwd, "*.sh")) + " " +\
            os.path.abspath(os.path.join(cwd, "*.log")) + " " +\
            " -r\n"

    cp_script = build_collect_data(dataDir, subRunDir_ED, subRunDir_vert,\
                                   subRunDir_susc, subRunDir_trilex,\
                                   mode=config['Postprocess']['data_bakup'])
    cp_script_path = os.path.abspath(os.path.join(cwd, "copy_data.sh"))
    with open(cp_script_path, 'w') as f:
        f.write(cp_script)
    split_script = split_files(config)
    split_script_path = os.path.abspath(os.path.join(dataDir, "split_files.sh"))
    with open(split_script_path, 'w') as f:
        f.write(split_script)
    storage_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "storage_io.py")
    storage_py_target = os.path.abspath(os.path.join(cwd, "storage_io.py"))
    print("copy from " + storage_py_path + " to " + storage_py_target)
    shutil.copyfile(storage_py_path, storage_py_target)
    data_path = os.path.abspath(os.path.join(cwd,"data"))
    clean_script_path = os.path.abspath(os.path.join(subRunDir_vert, "clean_script_auto"))
    print("TODO: copying gm_wim from dmft. WHY?")
    content = "cp " + os.path.join(subRunDir_ED, "gm_wim") + " " + subRunDir_vert + "\n"
    content += str(clean_script_path) + "\n"
    content += str(cp_script_path) + "\n"
    if 'text' in map(str.strip, config['Postprocess']['output_format'].split(',')):
        content += str(split_script_path) + "\n"
    conda_env = config['general']['custom_conda_env']
    if conda_env:
        content += "eval \"$(conda shell.bash hook)\"\n"
        content += "conda activate " + conda_env + "\n"
    content += "python storage_io.py " + data_path + " " +\
        config['Postprocess']['output_format'] + " && "
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
    print("running: " +run_cmd)

    fp = os.path.join(cwd, filename)
    with open(fp, 'w') as f:
        f.write(globals()["postprocessing_" + config['general']['cluster']](content, cslurm, config))
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
    cmd= "./klist.x > run_klist.out 2> run_klist.err"
    cslurm = config['general']['custom_slurm_lines']
    if not jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+jobid + " " + filename
    print("running: " +run_cmd)
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, 1, cslurm, cmd))
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
    filename = "lDGA.sh"
    fp = os.path.join(cwd, filename)
    cmd= "./copy.sh\nmpirun ./Selfk_LU_parallel_3D.x > run.out 2> run.err"
    procs = 2*int(config['Trilex']['nBoseFreq']) - 1
    cslurm = config['general']['custom_slurm_lines']
    if not jobid:
        run_cmd = "sbatch " + filename
    else:
        run_cmd = "sbatch" + " --dependency=afterok:"+jobid + " " + filename
    print("running: " +run_cmd)
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config, procs, cslurm, cmd, False))
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


# =========================================================================== 
# =                   Log and Postprocess Functions                         =
# =========================================================================== 

def dmft_log(jobid, loc, config):
    out = """
jobid = {0}
result_dir = {1}
last_check_stamp = {2}
last_status = {3}
run_time = {4}
job_info = {5}
"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

    out = out.format(jobid, os.path.abspath(loc), timestamp,
                     "TODO: not implemented yet", "TODO: not implemented yet",
                     "TODO: not implemented yet")
    return out

def build_collect_data(target_dir, dmft_dir, vertex_dir, susc_dir, trilex_dir, mode):
    dmft_files = ["hubb.dat", "hubb.andpar", "g0m", "g0mand", "gm_wim"]
    susc_files = ["chi_asympt", "matrix"]
    vertex_files = ["t.tar.gz", "parameters.dat", "GAMMA_DM_FULLRANGE", "F_DM" , "vert_chi"]
    trilex_dirs = ["tripamp_omega", "trip_omega", "trilex_omega"]

    copy_script_str = bak_files_script(dmft_dir, target_dir, dmft_files, header=True, mode=mode)
    copy_script_str += bak_files_script(susc_dir, target_dir, susc_files, header=False, mode=mode)
    copy_script_str += bak_files_script(vertex_dir, target_dir, vertex_files, header=False, mode=mode)
    copy_script_str += bak_dirs_script(trilex_dir, target_dir, trilex_dirs, header=False, mode=mode)

    return copy_script_str
