import toml
import numpy as np
import sys

def check_config_consistency(config):
    mu = config['parameters']['mu']
    U = config['parameters']['U']
    if not isclose(mu, U / 2.0):
        print("Not Calculating at half filling! mu = {0} and U = {1} Make sure"
              " that bath is not forced to be symmetric.".format(
                  mu, U / 2.0
              ))

def andpar_check_values(eps, tpar, ns):
    min_epsk_diff = 10e10
    min_tpar = min(abs(tpar))
    min_eps = min(abs(eps))
    for i in range(ns-1):
        for j in range(i+1,ns-1):
            if abs(eps[i] - eps[j]) < min_epsk_diff:
                min_epsk_diff = abs(eps[i] - eps[j])
    return min_epsk_diff, min_tpar, min_eps

def check_andpar_result(config, andpar_lines):
    ns = config['ED']['ns']
    eps = np.zeros(ns-1)
    tpar = np.zeros(ns-1)
    for i in range(ns-1):
        eps[i] = float(andpar_lines[9+i])
    for i in range(ns-1):
        tpar[i] = float(andpar_lines[ns+9+i])
    checks_success = [True, True, True]
    check_val = (1.0/config['parameters']['beta'])*config['ED']['bathsite_cancel_V']
    tpar_norm = np.sum(tpar ** 2)
    min_epsk_diff, min_tpar, min_eps = andpar_check_values(eps, tpar, ns)
    print("    Anderson Parameter Checks: ")
    print("   ============================   ")
    print("1. min(|V_k|)       = " + str(min_tpar))
    print("2. sum(V^2_k)       = " + str(tpar_norm))
    print("3. min(|e_k|)       = " + str(min_eps))
    print("4. min(|e_i - e_j|) = " + str(min_epsk_diff))
    print("   ============================   \n\n")

    for i in range(ns-1):
        for j in range(i+1,ns-1):
            if abs(eps[i] - eps[j]) <= config['ED']['bathsite_cancel_eps']:
                checks_success[0] = False
        if abs(tpar[i]) <= config['ED']['bathsite_cancel_V']:
            checks_success[1] = False
        if abs(eps[i]) <= check_val:
            checks_success[2] = False
    return checks_success


if __name__ == "__main__":
    with open("../config.toml", 'r') as f:
        config_in = f.read()
    config = toml.loads(config_in)
    with open("hubb.andpar", 'r') as f:
        lines = f.readlines()
    hubb_res = check_andpar_result(config, lines)
    print("Anderson Parameter Checks:\n")
    #print(" sum_k V^2_k = ", np.sum())
    print("No bath site cancel eps: ", hubb_res[0], "\n")
    print("No bath site cancel V: ", hubb_res[1], "\n")
    print("No small onsite e: ", hubb_res[2], "\n")
    sys.exit(0 if all(hubb_res) else 1)
