from scipy.special import comb
import numpy as np
import re
from math import ceil
import os

grid_pattern = re.compile(r'-(-?\d+\.\d+)', re.M)
lattice_f90 = {
      "3dsc": 1,
      "fcc":  2,
      "2dsc": 3,
      "bethe": 4,
      "2dmag": 5,
      "p6m": 6,
      "bcc": 7,
      "4dsc": 1
}

def to_fortran_bool(val):
    return ".true." if val else ".false."


# ============================================================================
# =                        Job File Templates                                =
# ============================================================================
def postprocessing_berlingenoa(content, custom, config, jobname=""):
    jn = "#SBATCH -J " + jobname + "\n" if len(jobname) else ""
    cl = "#SBATCH " + custom + "\n" if len(custom) else ""
    out = '''#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH -p {0}
#SBATCH --requeue
{1}
{2}
{3}
'''.format(config['general']['queue'],jn, cl, config['general']['custom_module_load'])
    out += content
    return out

def job_berlingenoa(config, procs, custom, cmd, queue="cpu-clx", copy_from_ed=True,
               custom_lines=True, jobname=""):
    jn = "#SBATCH -J " + jobname + "\n" if len(jobname) else ""
    cl = "#SBATCH " + custom + "\n" if len(custom) else ""
    out = '''#!/bin/bash
#SBATCH -t {0}
#SBATCH --ntasks {1}
#SBATCH -p {2}
{3}
{4}

{5}
'''
    if copy_from_ed:
        out = out + "./get_andpar.sh || true\n"
    out = out + "{6}\n"
    out = out.format(config['general']['global_time_limit'], procs, config['general']['queue'], cl, jn, config['general']['custom_module_load'],
                     cmd)
    return out


def w2dyn_submit_berlingenoa(config, runDir, it):
    out = ""
    fit_str = "julia {0} DMFT_{1}.hdf5 {2} {3} hubb_{1}.andpar\n"
    fit_str = fit_str.format(os.path.abspath(os.path.join(config['general']['codeDir'],"scripts/LadderDGA_utils/fitW2dyn.jl ")),
                             it,config['w2dyn']['NBath'],config['w2dyn']['NFreqFit'])
    if it == len(config['w2dyn']['N_DMFT'])-1:
        fit_str += "cp hubb_"+str(it)+".andpar hubb.andpar\n"
    if it == 0:
        hk_script = os.path.abspath(os.path.join(config['general']['codeDir'],'Dispersions.jl/scripts/w2dyn_kgrid.jl'))
        hk_loc = os.path.abspath(os.path.join(runDir, "ham.hk"))
        out += "julia " + hk_script + " " + config['parameters']['lattice'] + " " + str(config['w2dyn']['Nk']) + " " + hk_loc + "\n"
    out += "mpirun -np "+str(config['w2dyn']['N_procs'][it])+" "+\
            config['w2dyn']['runfile']+" Par_"+str(it)+".in \n"
    out += "NEWEST=$( ls -t current_run*.hdf5 | head -1 ) \n"
    out += "mv \"$NEWEST\" DMFT_"+str(it)+".hdf5\n"
    if it < len(config['w2dyn']['N_DMFT']):
        ncorr_path = os.path.abspath(os.path.join(config['general']['codeDir'],"scripts/LadderDGA_utils/ncorr.jl"))
        hdf5_path = os.path.abspath(os.path.join(runDir, "DMFT_"+str(it)+".hdf5"))
        out += "c=$(julia " + ncorr_path + " " + hdf5_path + ")\n"
        out += 'echo "NCorr = $c" >> Par_'+str(it+1)+'.in\n'
    out += fit_str
    return out
