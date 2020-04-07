from scipy.special import comb
import os

# =========================================================================== 
# =                        Job File Templates                               =
# =========================================================================== 

def job_berlin(config, procs, custom, cmd, copy_from_ed=True):
    out = '''#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks {0}
#SBATCH -p standard96
#SBATCH {1}
{2}
export SLURM_CPU_BIND=none
'''
    #large96
    if copy_from_ed:
        out = out + "./copy_ed_files \n"
    out = out + "{3}\n"
    out = out.format(procs, custom, config['general']['custom_module_load'], cmd)
    return out

def bak_files_script(source_dir, target_dir, files_list, header=False, mode="mv"):
    out = "#!/bin/bash \n" if header else ""
    out = out + mode +" " + os.path.abspath(source_dir) + "/{"
    for filename in files_list:
        out = out + filename + ","
    out = out[:-1] + "} " + os.path.abspath(target_dir)
    out += "\n"
    return out

def bak_dirs_script(source_dir, target_dir, dirs_list, header=False, mode="mv"):
    out = "#!/bin/bash \n" if header else ""
    for d in dirs_list:
        out += mode +" " + os.path.abspath(os.path.join(source_dir,d))\
            + " " + os.path.abspath(target_dir) + (" -ar \n" if (mode=="cp") else "\n")
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


# =========================================================================== 
# =                          File Templates                                 =
# =========================================================================== 

def ladderDGA_in(config):
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
{10}     {11}'''

    k_number = config['lDGAFortran']['k_range'] + 1
    out = out.format(
        config['parameters']['U'],
        config['parameters']['mu'],
        config['parameters']['beta'],
        config['Vertex']['nFermiFreq'],
        config['Vertex']['nBoseFreq'],
        config['lDGAFortran']['LQ'],
        config['lDGAFortran']['Nint'],
        int(k_number * ( k_number + 1 ) * ( k_number + 2 ) / 6),
        ".TRUE." if config['lDGAFortran']['only_chisp_ch'] else ".FALSE.",
        ".TRUE." if config['lDGAFortran']['only_lambda_sp'] else ".FALSE.",
        ".TRUE." if config['lDGAFortran']['only_positive_ch'] else ".FALSE.",
        ".TRUE." if config['lDGAFortran']['only_positive_sp'] else ".FALSE."
    )
    return out

def split_files(config):
    out = '''#!/bin/bash
cwd=$(pwd)
cd "$(dirname "$0")"
mkdir -p gamma_dir
cd gamma_dir
split --suffix-length=3 -d --lines={0} ../GAMMA_DM_FULLRANGE gamma
cd ..
mkdir -p chi_dir
cd chi_dir
split --suffix-length=3 -d --lines={1} ../vert_chi chi
cd $cwd
'''
    lines = (2*(config['Vertex']['nBoseFreq']+1))**2
    out = out.format(lines, lines)
    return out

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

def init_vertex_h(config):
    ns = config['parameters']['ns']
    nmaxx = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (Iwmax={2})\n"
    out += "      parameter (Iwbox_bose={3})\n"
    out += "      parameter (nmpara={4})\n"
    out = out.format(nmaxx, ns, int(config['Vertex']['nFermiFreq']),
        int(config['Vertex']['nBoseFreq']), int(config['Vertex']['nmpara']))
    return out

def init_psc_h(config):
    ns = config['parameters']['ns']
    nmaxx = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (Iwmax={2})\n"
    out += "      parameter (nmpara={3})\n"
    out = out.format(nmaxx, ns, int(config['Susc']['nBoseFreq']),
                                int(config['Vertex']['nmpara']))
    return out

def init_trilex_h(config):
    ns = config['parameters']['ns']
    nmaxx = int(comb(ns, int(ns/2)) ** 2)
    out = "      parameter (nmaxx = {0})\n"
    out += "      parameter (nss={1})\n"
    out += "      parameter (Iwmax={2})\n"
    out += "      parameter (Iwbox_bose={3})\n"
    out += "      parameter (nmpara={4})\n"
    out = out.format(nmaxx, ns, int(config['Trilex']['nFermiFreq']),
        int(config['Trilex']['nBoseFreq']), int(config['Trilex']['nmpara']))
    return out

def init_sumt_h(config):
    out = "      parameter (Iwmax={0})\n"
    out += "      parameter (Iwbox_fermi={1})\n"
    out += "      parameter (Iwmax_bose={2})\n"
    out += "      parameter (nprocs={3})\n"
    out = out.format(int(config['Vertex']['nFermiFreq']), \
                     int(config['Vertex']['nFermiFreq']), \
                     int(config['Vertex']['nBoseFreq']),  \
                     2*int(config['Vertex']['nBoseFreq']) + 1)
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
            out += "  1.000000000000\n"
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
            out += "  0.200000000000\n"
    out += "  {3}                      #chemical potential\n"
    out = out.format(
        config['parameters']['beta'], #config['ED']['conv_param'],
        config['parameters']['ns'], config['ED']['iterations'],
        config['parameters']['mu']
    )
    return out

def call_script(config):
    out = '''
#!/bin/bash
i=1
sed '1s/^.*$/        1/' idw.dat >hilfe
mv hilfe idw.dat
beta={0}d0
uhub={1}d0
{2}

while [ $i -le 8 ]
do
    name=ver_tpri_run_U$uhub\_beta$beta\_$i.x
    echo $name
    cp run.x ./$name
    mpirun -np {3} ./$name > vertex_out$i.dat 2> vertex_error$i.dat
    wait
    sleep 5
    i=$((i+1))
done
'''
    #pids[${{i}}]=$!
    #for pid in ${{pids[*]}}; do
    #    wait $pid
    #done
    out = out.format(config['parameters']['beta'], config['parameters']['U'],\
                     config['general']['custom_module_load'],\
                     2*int(config['Vertex']['nBoseFreq']) - 1)
    return out

def parameterts_dat(config):
    out = '''c Iwbox_bose_ph   Iwbox_fermi_ph   Iwbox_bose_pp   Iwbox_fermi_pp   Iwbox_bose_gamma   Iwbox_fermi_gamma    Iwbox_bose_lambda   Iwbox_fermi_lambda   Iwbox_green_function
  {0}                 {1}              0                0                0                 0                   0                   0                    {2}
c Frequencies for up_down particle-particle vertex: Iwbox_bose_up_down   Iwbox_fermi_up_down   Iwbox_bose_up_down_backshift   Iwbox_fermi_up_down_backshift
   0     0     0     0
c beta     U
  {3}d0     {4}d0
c vert_pp-> .FALSE.:file vert_chi_pp does not exist, .TRUE.: vert_chi_pp exists; calc_gamma->.TRUE. Gamma's are calculated; calc_lambda ->.TRUE. Lambda's are calculated; calc_up_down->up_down particle-particle vertex is calculated separately; calc_F_only->only F is calculated
   .FALSE.     .TRUE.     .FALSE.     .FALSE.   .FALSE.
c eps: threshold for the smallest eigenvalue of the matrices that are inverted
  1.d-15


COMMENTS about the frequency ranges (or to be more precise about the ranges for the indices for the frequencies):

The frequencies are defined in the following way:
Bosonic frequency: omege=pi*T*(2*i), -Iwbox_bose <= i <= +Iwbox_bose
Fermionic frequency: nu=pi*T*(2*j+1), -Iwbox_fermi <= j <= +Iwbox_fermi-1
Ferminoic frequency: nu_prime=pi*T*(2*k+1), -Iwbox_fermi<= k <= +Iwbox_fermi-1

-) Iwbox_bose_ph, Iwbox_fermi_ph: The number of frequencies for which chi_up_up and chi_up_down are given in the file 'vert_chi'


-) Iwbox_bose_pp, Iwbox_fermi_pp
   a) 'vert_chi_pp' exists: The number of frequencies for which chi_up_up and chi_up_down are given in the file 'vert_chi_pp' 
   b) 'vert_chi_pp' does not exit: in that case one has to calculate the chi's in the pp-notation from the ph-notation by a frequency
      shift: chi_pp(omega,nu,nu_prime)=chi(omega-n-nu_prime,nu,nu_prime).
      omega-nu-nu_prime=pi*T*(2*(i-k-j-1)), so the shifted bosonic frequency corresponds to an index shift i-j-k-1 of the bosonic
      index i.
      Now -Iwbox_bose_ph <= i-j-k-1 <= +Iwbox_bose_ph. The largest value for (i-j-k-1) is reached if i takes the largest value
      Iwbox_bose_pp and j and k take the smallest value -Iwbox_fermi_pp. This leads to the condition: 

      (Iwbox_bose_pp+2*Iwbox_fermi_pp-1) <= Iwbox_bose_ph.

      The smallest value for (i-j-k-1) you get for i=-Iwbox_bose_pp and j,k=Iwbox_fermi_pp-1. This leads to the condition:
      -Iwbox_bose_ph >= (-Iwbox_bose_pp-2*Iwbox_fermi_pp+1) which is - after multiplying with -1 - the same condition as above. 
      So this condition has to  be fullfilled for Iwbox_bose_pp and Iwbox_fermi_pp. In addition you have:

      Iwbox_fermi_pp <= Iwbox_ferm_ph.
      One possible choice for Iwbox_bose_pp and Iwbox_fermi_pp (where the number of bosonic frequencies is nearly the same

      as the number of fermionic frequencies) is:
      Iwbox_fermi_pp=int((Iwbox_bose_ph-1)/3)
      Iwbox_bose_pp=Iwbox_bose_ph-1-2*Iwbox_fermi_pp
   c) Iwbox_bose_gamma and Iwbox_fermi_gamma set the frequency ranges for the output of the Gamma's. The output is given
      in the ph-notation, so the singlet- and the triplet-gamma have to be shifted. 
      Therefore the relation of Iwbox_bose_gamma and Iwbox_fermi_gamma to Iwbox_bose_pp is the same as for Iwbox_bose_pp  and Iwbox_fermi_pp to 
      Iwbox_bose_ph in paragraph b). 

      (Iwbox_bose_gamma+2*Iwbox_fermi_gamm-1)<=Iwbox_bose_pp
      
      This is due to the fact that the frequency shift is simply reversed, in order to get all gammas 
      in the ph-notation: i -> i+j+k+1
      In addition you have:

      Iwbox_bose_gamma<=Iwbox_bose_ph
      Iwbox_fermi_gamma<=Iwbox_fermi_ph

   d) In the lambdas frequency shift of the form: i -> k-j and k -> i+j appear - but only in the ph-channels. That means that:
      2*Iwbox_fermi_lambda-1 <= Iwbox_bose_ph
      Iwbox_bose_lambda+Iwbox_fermi_lambda <= Iwbox_fermi_ph
      In addition it is clear that one has the restriction that 
      Iwbox_fermi_lambda <= Iwbox_fermi_gamma
      Iwbox_bose_lambda <= Iwbox_bose_gamma
      since the Gamma's are only defined on this small intervall when one uses the ph-notation. 
   e) For the up-down particle-particle vertex the shift nu_prime->omega-nu_prime has to be performed.
      Therefore one has as condition for the frequency ranges:
      (Iwbox_bose_up_down+Iwbox_fermi_up_down)<=Iwbox_fermi_pp
      (Iwbox_bose_up_down_backshift+Iwbox_fermi_up_down_backshift)<=Iwbox_fermi_up_down
   f) Iwbox_green_function is the range of the Green function. Since it appear in chi_0 in the combination i+j or i-j-1 it has to fullfill:
      Iwbox_green_function >= Iwbox_fermi_ph+Iwbox_bose_ph   (if Iwbox_ph >= Iwbox_pp, otherwise the same condition hold for Iwbox_..._pp)
'''
    out = out.format(config['Vertex']['nBoseFreq'], config['Vertex']['nFermiFreq'],
                     config['Vertex']['nFermiFreq']+config['Vertex']['nBoseFreq'],
                     config['parameters']['beta'], config['parameters']['U'])
    return out
