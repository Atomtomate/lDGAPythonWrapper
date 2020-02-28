import os
import shutil
import subprocess
#import pathlib
import toml
import re
import stat
from math import isclose
from file_templates import *

   

# =========================================================================== 
# =                         helper functions                                =
# =========================================================================== 
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

def check_config_consistency(config):
    if not isclose(config['parameters']['mu'], config['parameters']['U']/2.0):
        println("Not Calculating at half filling! Make sure that bath is not forced to be symmetric.")

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

def create_and_populate_files(dirName, flist, config):
    for fn in flist:
        fp = dirName + "/" + fn
        with open(fp, 'w') as f:
            f.write(globals()[fn.replace(".","_")](config))

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
    

# =========================================================================== 
# =                          copy functions                                 =
# =========================================================================== 

def copy_and_edit_vertex(subCodeDir, subRunDir, subRunDir_ED, config):
    files_dmft_list = ["hubb.andpar", "hubb.dat", "gm_wim"]
    files_list = ["checksum_script", "clean_script", "idw.dat",
                  "inversion_pp_fotso.f90",  "split_script",
                   "tpri.dat", "varbeta.dat", "ver_tpri_run.f"]
    scripts = ["copy_ed_files" ,"call_script", "checksum_script", "clean_script", "split_script"]

    fp = os.path.join(subRunDir, "call_script")
    with open(fp, 'w') as f:
        f.write(call_script(config))
    fp = os.path.join(subRunDir, "parameters.dat")
    with open(fp, 'w') as f:
        f.write(parameterts_dat(config))
    fp = os.path.join(subRunDir, "init.h")
    with open(fp, 'w') as f:
        f.write(init_vertex_h(config))
    fp = os.path.join(subRunDir, "copy_ed_files")
    with open(fp, 'w') as f:
        f.write(copy_from_ed(os.path.abspath(subRunDir_ED), subRunDir, files_dmft_list))

    for filename in files_list:
        source_file_path = os.path.abspath(os.path.join(subCodeDir, filename))
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.abspath(os.path.join(subRunDir, filename))
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)

    src_files  = ["sum_t_files.f"]
    
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
        f.write(copy_from_ed(os.path.abspath(subRunDir_ED),
                             os.path.abspath(subRunDir), files_dmft_list))

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
        f.write(copy_from_ed(os.path.abspath(subRunDir_ED),
                             os.path.abspath(subRunDir), files_dmft_list))

    for filename in files_list:
        source_file_path = os.path.join(subCodeDir, filename)
        target_file_path = os.path.join(subRunDir, filename)
        shutil.copyfile(source_file_path, target_file_path)
    for filename in scripts:
        target_file_path = os.path.join(subRunDir, filename)
        st = os.stat(target_file_path)
        os.chmod(target_file_path, st.st_mode | stat.S_IEXEC)



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
    process = subprocess.run("sbatch ./ed_dmft_run.sh", cwd=cwd, shell=True, capture_output=True)
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
    procs = 2*int(config['Vertex']['nBoseFreq']) - 1
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
    cmd= "./run.x > run.out 2> run.err"
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
