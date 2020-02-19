import os
import shutil
import subprocess
#import pathlib
import toml
from scipy.special import comb


# =========================================================================== 
# =                        Job File Templates                               =
# =========================================================================== 

def job_berlin(config):
    out = '''#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks {0}
#SBATCH -p large96
module load openblas/gcc.9/0.3.7 impi/2019.5 intel/19.0.5
export SLURM_CPU_BIND=none
mpirun ./ed_dmft.x > ed.out 2> ed.err
'''.format(int(config['ED']['nprocs']))
    return out


# =========================================================================== 
# =                          File Templates                                 =
# =========================================================================== 

def tpri_dat(config):
    t = config['parameters']['t']
    return "      t=" + str(t) + "d0\n      t1=0.0d0\n      t2=0.0d0"

def init_h(config):
    ns = config['parameters']['ns']
    nmaxx = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (prozessoren={2})\n"
    out = out.format(nmaxx, ns, int(config['ED']['nprocs']))
    return out

def hubb_dat(config):
    out = '''c  U,   hmag
    {0}d0,  0.d0 0.d0
    c beta, w_min, w_max, deltino
    {1}d0, {2}, {3}, 0.01
    c ns,imaxmu,deltamu, # iterations, conv.param.
    {4}, 0, 0.d0, 250,  1.d-14
    c ifix(0,1), <n>,   inew, iauto
    0  , 1.0d0,   1,    1,
    c  th0 , iexp (insignificant)
    1.d-4, 1
    c nmin, nmax
    3 , 7
    c lambda, w0, nph
    0.0, 0.4, 4
    1
    '''
    # TODO: use float(val.replace('e', 'd'))
    out = out.format( config['parameters']['U'],
        config['parameters']['beta'], config['ED']['w_min'],
        config['ED']['w_max'], #config['ED']['conv_param'],
        config['parameters']['ns'] #TODO: important params?
    )
    return out

def hubb_andpar(config):
    #TODO: var length for number of sites
    out = '''           ========================================
               1-band            30-Sep-95 LANCZOS
            ========================================
NSITE     5 IWMAX32768
 {0}d0, -12.0, 12.0, 0.007
c ns,imaxmu,deltamu, # iterations, conv.param.
 {1}, 0, 0.d0, {2},  1.d-14
c ifix(0,1), <n>,   inew, iauto
Eps(k)
'''
    for i in range(config['parameters']['ns']-1):
        out += "  1.000000000000\n"
    out += " tpar(k)\n"
    for i in range(config['parameters']['ns']-1):
        out += "  0.200000000000\n"
    out += "{3}                      #chemical potential\n"
    out = out.format(
        config['parameters']['beta'], #config['ED']['conv_param'],
        config['parameters']['ns'], config['ED']['iterations'],
        config['parameters']['mu']
    )
    return out


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

def run_ed_dmft(cwd, config):
    fp = cwd + "/" + "ed_dmft_run.sh"
    with open(fp, 'w') as f:
        f.write(globals()["job_" + config['general']['cluster']](config))
    process = subprocess.run("sbatch ./ed_dmft_run.sh", cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Compilation did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    return True
    
def copy_from_dmft(subRunDir_ED, subRunDir, files):
    for f in files:
        shutil.copyfile(subCodeDir_ED + "/" + f, subRunDir + "/" + f)
    
def copy_and_edit():
    shutil.copyfile(subCodeDir + "/" + src_file, subRunDir_ED + "/" + src_file)
