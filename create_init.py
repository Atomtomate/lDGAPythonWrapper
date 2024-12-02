import numpy as np
from scipy.special import comb
import toml

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
    return config

def read_andpar(ns):
    with open("hubb.andpar", 'r') as f:
        lines = f.readlines()
    eps = np.zeros(ns-1)
    tpar = np.zeros(ns-1)
    for i in range(ns-1):
        eps[i] = float(lines[9+i])
    for i in range(ns-1):
        tpar[i] = float(lines[ns+9+i])
    mu = float(lines[ns+ns+9-1])
    return mu, eps,tpar

def gen_init_h(config_path):
    config = read_preprocess_config(config_path)
    ns = config['ED']['ns']
    nmax = int(comb(ns, int(ns/2)) ** 2)
    mu, eps, tpar = read_andpar(ns)

    out = """module globals
    use types
        integer(id), parameter :: nmax = {0}
        integer(id), parameter :: ns={1}
        real(dp), parameter :: Traw={2}
        real(dp), parameter :: small={3}
        real(dp), parameter :: approx={4}
        real(dp), parameter,dimension({1})::epsk=(/ &
""".format(nmax, ns, config['ED']['Traw'],\
         config['ED']['small'],config['ED']['approx'])
    out += f"    {-mu}, &\n"
    for i in range(ns-2):
        out += f"    {eps[i]}, &\n"
    out += f"    {eps[ns-2]}/)\n"
    out += f"    real(dp), parameter,dimension({ns-1})::tpar=(/&\n"
    for i in range(ns-2):
        out += f"    {tpar[i]}, &\n"
    out += f"    {tpar[ns-2]}/)\n"
    out += "end module globals"
    with open("init.h", 'w') as f:
        f.write(out)

gen_init_h("../config.toml")
