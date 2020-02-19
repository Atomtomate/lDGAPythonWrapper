import os
import shutil
import subprocess
#import pathlib
import toml
from scipy.special import comb


# =========================================================================== 
# =                        Job File Templates                               =
# =========================================================================== 


# =========================================================================== 
# =                          File Templates                                 =
# =========================================================================== 

def tpri_dat(config):
    t = config['parameters']['t']
    return "      t=" + str(t) + "d0\n      t1=0.0d0\n      t2=0.0d0"

def init_h(config):
    ns = config['parameters']['ns']
    nmaxx = int(comb(ns, int(ns/2)) ** 2)
    return "      parameter (nmaxx = {})\n".format(nmaxx)

def hubb_dat(config):
    out = '''c  U,   hmag
    {0}d0,  0.d0 0.d0
    c beta, w_min, w_max, deltino
    {1}d0, {2}, {3}, {4}
    c ns,imaxmu,deltamu, # iterations, conv.param.
    {5}, 0, 0.d0, 250,  1.d-13
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
    out.format( config['parameters']['U'],
        config['parameters']['beta'], config['ED']['w_min'],
        config['ED']['w_max'], config['ED']['deltino'],
        config['parameters']['ns'] #TODO: important params?
    )
    return out

def hubb_andpar(config):
    #TODO: var length for number of sites
    out = '''           ========================================
               1-band            30-Sep-95 LANCZOS
            ========================================
NSITE     5 IWMAX32768
 {0}d0, -12.0, 12.0, {1}
c ns,imaxmu,deltamu, # iterations, conv.param.
 {2}, 0, 0.d0, {3},  1.d-14
c ifix(0,1), <n>,   inew, iauto'''
    init = '''Eps(k)
        1.15026497703568
        0.117442660612338
        -1.15026497703568
        -0.117442660612338
        tpar(k)
        0.311680378997208
        0.166899195169290
        0.311680378997208
        0.166899195169290
        {4}                      #chemical potential
    '''
    out.format(
        config['parameters']['beta'], config['ED']['conv_param'],
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
    print("running command:")
    print(command.split())
    process = subprocess.run(command, cwd=cwd, shell=True, capture_output=True)
    if not (process.returncode == 0):
        print("Compilation did not work as expected:")
        print(process.stdout.decode("utf-8"))
        print(process.stderr.decode("utf-8"))
        return False
    return True
