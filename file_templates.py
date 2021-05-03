from scipy.special import comb
import numpy as np
import re
from math import ceil
import os


# ============================================================================
# =                        Job File Templates                                =
# ============================================================================
def job_berlin(config, procs, custom, cmd, copy_from_ed=True):
    out = '''#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks {0}
#SBATCH -p standard96
#SBATCH {1}
{2}
export SLURM_CPU_BIND=none
'''
    # large96
    if copy_from_ed:
        out = out + "./copy_dmft_files \n"
        out = out + "./copy_data_files || true \n"
    out = out + "{3}\n"
    out = out.format(procs, custom, config['general']['custom_module_load'],
                     cmd)
    return out


def bak_files_script(source_dir, target_dir, files_list, header=False,
                     mode="mv"):
    out = "#!/bin/bash \n" if header else ""
    out = out + mode + " " + os.path.abspath(source_dir) 
    if len(files_list) > 1:
        out += "/{"
        for filename in files_list:
            out = out + filename + ","
        out = out[:-1] + "} " 
    else:
        out += "/" + files_list[0] + " "
    out += os.path.abspath(target_dir)+"\n"
    return out


def bak_dirs_script(source_dir, target_dir, dirs_list, header=False,
                    mode="mv"):
    out = "#!/bin/bash \n" if header else ""
    for d in dirs_list:
        out += mode + " " + os.path.abspath(os.path.join(source_dir, d)) +\
            " " + os.path.abspath(target_dir) + \
            (" -ar \n" if (mode == "cp") else "\n")
    return out


def postprocessing_berlin(content, custom, config):
    out = '''#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --ntasks=1
#SBATCH -p large96:shared
#SBATCH --requeue
#SBATCH {0}
{1}
export SLURM_CPU_BIND=none
'''.format(custom, config['general']['custom_module_load'])
    out += content
    return out


# Either read list form file or generate string for new file
def parse_freq_list(freq_grid):
    nFreq = len(range(*freq_grid[1])) * len(range(*freq_grid[0])) * \
                len(range(*freq_grid[0]))
    file_str = "# === Header Start === ".ljust(30)
    file_str += "\n# Elements".ljust(31)
    file_str += ("\n   " + str(nFreq)).ljust(31)
    file_str += "\n# === Header End === ".ljust(31)
    for bi in range(*freq_grid[1]):
        for nui in range(*freq_grid[0]):
            for nupi in range(*freq_grid[0]):
                file_str += "\n" + str(bi).rjust(10) + str(nui).rjust(10) + \
                        str(nupi).rjust(10)
    return nFreq, file_str

# build fortran input file from frequency grids
def freq_list_h(config, nFreq, max_freq, mode=None):
    max_freq = 2*max_freq
    line_length_counter = 6
    max_line_length = 3500
    out = "      real(dp), parameter :: beta="+str(config['parameters']['beta'])+"\n"
    out += "      real(dp), parameter :: uhub="+str(config['parameters']['U'])+"\n"
    out += "      integer(id), parameter :: nFreq = "+str(nFreq) + "\n"
    out += "      integer(id), parameter :: maxFreq = "+str(max_freq) + "\n"
#    if max_freq < 3500:
#        out += "      complex*16,parameter,dimension("+str(-max_freq)+":"+str(max_freq)+") :: mf = (/ & \n        "
#        for elf in range(-max_freq,max_freq+1):
#            el = (1j*elf*np.pi/config['parameters']['beta'])
#            els = "({:0.1f},{:0.12f}),".format(el.real,el.imag)
#            if line_length_counter+len(els) < max_line_length:
#                out += els
#                line_length_counter += len(els)
#            else:
#                out += " & \n        "+els
#                line_length_counter = 8+len(els)
#        out = out[:-1]
#        out += "/)\n"
#    else:
#        raise NotImplementedError("Frequency range too large for static alloc")
    return out

# ============================================================================
# =                          File Templates                                  =
# ============================================================================
def ladderDGA_in(config, mode=None):
    out = '''c AIM parameters: U, mu, beta, nden
{0}d0      {1}d0       {2}d0       1.0d0
c Iwbox    Iwbox_bose    shift
{3}       {4}       0
c LQ    Nint     k_number
{5}       {6}       {7}
c Should only chis_omega and chich_omega be calculated?
{8}
c Should a lambda-correction be performed only in the spin-channel?
{9}
c Should the summation over the bosonic frequency in the charge-/spin-channel be done for all bosonic Matsubara frequencies?
{10}     {11}\n'''

    k_number = config['lDGA']['k_range'] + 1
    raise NotImplementedError("No longer working on a grid! Cannot start FortranLDGA")
    nBoseFreq = 0
    out = out.format(
        config['parameters']['U'],
        config['parameters']['mu'],
        config['parameters']['beta'],
        config['Vertex']['nFermiFreq'],
        nBoseFreq,
        config['lDGA']['LQ'],
        config['lDGA']['Nint'],
        int(k_number * ( k_number + 1 ) * ( k_number + 2 ) / 6),
        ".TRUE." if config['lDGA']['only_chisp_ch'] else ".FALSE.",
        ".TRUE." if config['lDGA']['only_lambda_sp'] else ".FALSE.",
        ".FALSE." if config['lDGA']['only_positive_ch'] else ".TRUE.",
        ".FALSE." if config['lDGA']['only_positive_sp'] else ".TRUE."
    )
    if config['lDGA']['kInt'].lower() == "fft":
        out += "c fft_bubble, fft_real(not implemented)\n.TRUE.     .FALSE.\n"
    return out

#TODO: some fixed parameters
def lDGA_julia(config, dataDir):
    out = """[Model]
U    = {0}
mu   = {1}
beta = {2}
nden = 1.0
Dimensions = {3}

[Simulation]
Nk = {4}
tail_correction = "{5}"                # "Richardson" # Nothing, Richardson, Shanks
lambda_correction = "{6}"              # nothing, spin, spin_charge
bosonic_sum = "{7}"                    # common (intersection of individual ranges), individual (max range in each channel; fortran def    ault), full (all frequencies), fixed:N:M (always sum from N to (including) M, indexing starts at 0)
force_full_bosonic_chi = {8}           # compute all omega frequencies for chi and trilex
chi_unusable_fill_value = "{9}"        # can be "0", "chi_lambda" or "chi". sets either 0, lambda corrected or non lambda corrected     values outside usable omega range
chi_only = {10}                        # Should only chis_omega and chich_omega be calculated?
rhs  = "{11}"                           # native (fixed for tc, error_comp for naive), fixed (n/2 (1 - n/2) - sum(chi_ch)), error_comp (chi    _loc_ch + chi_loc_sp - chi_ch)
fermionic_tail_coeffs = {12}
bosonic_tail_coeffs = {13}
usable_prct_reduction = {14}

[Environment]
inputDataType = "jld2"      # jld2, text, parquet, TODO: implement hdf5
writeFortran = false
loadAsymptotics = false
inputDir = "{15}"
freqFile = "{16}"
inputVars = "ED_out.jld2"
asymptVars = "vars_asympt_sums.jld"
cast_to_real = false             # TODO: not implemented. cast all arrays with vanishing imaginary part to real
loglevel = "debug"        # error, warn, info, debug
logfile = "lDGA.log"
progressbar = false

[legacy]

[Debug]

"""
    out = out.format(
        config['parameters']['U'],
        config['parameters']['mu'],
        config['parameters']['beta'],
        config['parameters']['Dimensions'],
        config['lDGAJulia']['Nk'],
        config['lDGAJulia']['tail_correction'],
        config['lDGAJulia']['lambda_correction'],
        config['lDGAJulia']['bosonic_sum'],
        str(config['lDGAJulia']['force_full_bosonic_chi']).lower(),
        config['lDGAJulia']['chi_unusable_fill_value'],
        str(config['lDGAJulia']['chi_only']).lower(),
        config['lDGAJulia']['rhs'],
        config['lDGAJulia']['fermionic_tail_coeffs'],
        config['lDGAJulia']['bosonic_tail_coeffs'],
        config['lDGAJulia']['usable_prct_reduction'],
        dataDir,
        os.path.abspath(config['Vertex']['freqList'][:-3] + "jld2")
    )
    return out



def tpri_dat(config, mode=None):
    t = 0.5/np.sqrt(2*config['parameters']['Dimensions'])
    return "      t=" + str(t) + "d0\n      t1=0.0d0\n      t2=0d0"

def init_h(config, mode=None):
    ns = config['parameters']['ns']
    nmax = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (prozessoren={2})\n"
    out = out.format(nmax, ns, (ns+1)**2)
    return out

#TODO: unify init files using modes
def init_vertex_h(config, mode=None):
    ns = config['parameters']['ns']
    nmax = int(comb(ns, int(ns/2)) ** 2)
    out =  "      integer, parameter :: nmax = {0}\n"
    out += "      integer, parameter :: ns={1}\n"
    out = out.format(nmax, ns)
    return out

def init_2_h(config, mode=None):
    out =  "      logical, parameter :: bethe={0}\n"
    out += "      logical, parameter :: twodim={1}\n"
    out += "      logical, parameter :: symm={2}\n"
    bethe = ".true." if config['parameters']['bethe']  else ".false."
    twodim = ".true." if config['parameters']['Dimensions'] == 2 else ".false."
    symm = ".true." if config['parameters']['symm'] else ".false."
    out = out.format(bethe, twodim, symm)
    return out


def init_susc_h(config, mode=None):
    ns = config['parameters']['ns']
    nmax = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (Iwmax={2})\n"
    out += "      parameter (nmpara={3})\n"
    out = out.format(nmax, ns, int(config['Susc']['nBoseFreq']),
                                int(config['Susc']['nmpara']))
    return out

def init_trilex_h(config, mode=None):
    ns = config['parameters']['ns']
    nmax = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (Iwmax={2})\n"
    out += "      parameter (Iwbox_bose={3})\n"
    out += "      parameter (nmpara={4})\n"
    out = out.format(nmax, ns, int(config['Trilex']['nFermiFreq']),
        int(config['Trilex']['nBoseFreq']), int(config['Trilex']['nmpara']))
    return out

def q_sum_h(config, mode=None):
    cnf_str = "".join(config['lDGA']['kInt'].lower().split()) # remove whitespaces and convert to lower case
    if cnf_str == "naive" or cnf_str == "gl-1" or cnf_str == "fft":
        out = """INTEGER, PARAMETER :: ng=1
!points for the Gauss-Legendre integration
REAL(KIND=8), PARAMETER, DIMENSION(ng) :: tstep=(/1.0d0/)
!weights for the Gauss-Legendre integration
REAL(KIND=8), PARAMETER, DIMENSION(ng) :: ws=(/2.0d0/)"""
    elif "gl" in cnf_str:
        order = cnf_str[3:]
        tsteps, ws = np.polynomial.legendre.leggauss(int(order))
        out = "INTEGER, PARAMETER :: ng="+order+"\n"
        out += "!points for the Gauss-Legendre integration\n"
        out += "REAL(KIND=8), PARAMETER, DIMENSION(ng) :: tstep= &\n(/"
        for i,tstep in enumerate(tsteps):
            out += str(tstep) + ","
            if i%2 == 0 and i > 0 and i < len(tsteps) - 1:
                out += " &\n"
        out = out[0:-1] + "/)\n"
        out += "!weights for the Gauss-Legendre integration\n"
        out += "REAL(KIND=8), PARAMETER, DIMENSION(ng) :: ws= &\n(/"
        for i,w in enumerate(ws):
            out += str(w) + ","
            if i%2 == 0 and i > 0 and i < len(ws) - 1:
                out += " &\n"
        out = out[0:-1] + "/)"
    else:
         raise NotImplementedError("Q-Integration method not recognized")
    return out

def hubb_dat(config, mode=None):
    out = '''c  U,   hmag
{0}d0,  0.d0 0.d0
c beta, w_min, w_max, deltino
{1}d0, {2}, {3}, 0.01
c ns,imaxmu,deltamu, # iterations, conv.param.
{4}, 0, 0.d0, {5},  {6}
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
        config['parameters']['ns'], #TODO: important params?
        config['ED']['iterations'], config['ED']['conv_param']
    )
    return out

def hubb_andpar(config, eps_k="", tpar_k=""):
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
    if eps_k:
        ns_eps = len(eps_k.splitlines())
        if not (ns_eps == config['parameters']['ns']-1):
            raise InputError("Number of sites in Eps(k) ({0}) does not \
            correspond to ns ({1})".format(
                ns_eps, config['parameters']['ns']-1
            ))
        out += eps_k
    else:
        for i in range(config['parameters']['ns']-1):
            out += "  0."+str(3*(i+1))+"00000000000\n"
    out += " tpar(k)\n"
    if tpar_k:
        tp_eps = len(tpar_k.splitlines())
        if not (tp_eps == config['parameters']['ns']-1):
            raise InputError("Number of sites in tpar(k) ({0}) does not \
            correspond to ns ({1})".format(
                tp_eps, config['parameters']['ns']-1
            ))
        out += tpar_k
    else:
        for i in range(config['parameters']['ns']-1):
            out += "  0.250000000000\n"
    out += "  {3}                      #chemical potential\n"
    out = out.format(
        config['parameters']['beta'], #config['ED']['conv_param'],
        config['parameters']['ns'], config['ED']['iterations'],
        config['parameters']['mu']
    )
    return out

