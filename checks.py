import toml
import numpy as np

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
    for i in range(ns-1):
        eps[i] = float(andpar_lines[9+i])
    for i in range(ns-1):
        tpar[i] = float(andpar_lines[ns+9+i])
    eps_ssum = np.sum(eps**2)
    tpar_ssum = np.sum(tpar**2)
    checks_success = [True, True, True, True]
    checks_success[0] = abs(eps_ssum*0.25 - tpar_ssum) <= config['ED']['square_sum_diff']
    check_val = (1.0/config['parameters']['beta'])*config['ED']['bathsite_cancel_V']
    for i in range(ns-1):
        for j in range(i+1,ns-1):
            if abs(eps[i] - eps[j]) <= config['ED']['bathsite_cancel_eps']:
                checks_success[1] = False
        if abs(tpar[i]) <= config['ED']['bathsite_cancel_V']:
            checks_success[2] = False
            print(tpar[i])
        if abs(eps[i]) <= check_val:
            checks_success[3] = False
    return checks_success


if __name__ == "__main__":
    with open("../config.toml", 'r') as f:
        config_in = f.read()
    config = toml.loads(config_in)
    with open("hubb.andpar", 'r') as f:
        lines = f.readlines()
    hubb_res = check_andpar_result(config, lines)
    print("Anderson Parameter Checks:\n")
    print("0.5*eps_k^2 ~ tpar^2", hubb_res[0])
    print("No bath site cancel eps: ", hubb_res[1])
    print("No bath site cancel V: ", hubb_res[2])
    print("No small onsite e: ", hubb_res[3])