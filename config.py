import os
import re
import shutil
import subprocess
from datetime import datetime


# TODO: implement config class as wrapper around config.toml.
#       this class should provide all cluster and config dependent strings

queue_system = {"hamburg": "slurm", "berlin": "sge"}
qacct_jn = re.compile(r'jobname(\s+)(.*)')
qacct_failed = re.compile(r'failed(\s+)(.*)')
qacct_exit_stat = re.compile(r'exit_status(\s+)(.*)')

def get_submit_cmd(config, dependency_id = None):
    jid_str = ""
    if not dependency_id is None:
        if isinstance(dependency_id, list):
            for el in dependency_id:
                if not el is None:
                    jid_str += el + ":"
            jid_str = jid_str[:-1]
        else:
            jid_str = dependency_id
    if config['general']['cluster'].lower() == "hamburg":
        res = "qsub "
        if not dependency_id is None:
            res += "-hold_jid "+jid_str+" "
    elif config['general']['cluster'].lower() == "berlin": 
        res = "sbatch "+jid_str+" "
    else:
        raise ValueError("Unkown cluster `" +config['general']['cluster']+ "` !")
    return res


def format_log(fn, jobid, loc, config):
    out = """
jobid = {0}
result_dir = {1}
last_check_stamp = {2}
last_status = {3}
run_time = {4}
job_name = {5}
    """

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    status = ""

    if config['general']['cluster'].lower() == "berlin":
        job_cmd = "sacct -j " + str(jobid) + " --format=User,JobID,Jobname,"\
              "partition,state,elapsed,nnodes,ncpus,nodelist"
        process = subprocess.run(job_cmd, shell=True, capture_output=True)
        stdout = process.stdout.decode("utf-8")
        stderr = process.stderr.decode("utf-8")

        if not (process.returncode == 0):
            res = stderr
            status = "sacct not accessible"
            run_time = "sacct not accessible"
            job_name = "sacct not accessible"
            print("Warning: could not run sacct in order to check job completion! "
                  "Got: \n" + res)
        else:
            if len(stdout.splitlines()) < 3:
                out = out.format(jobid, os.path.abspath(loc), timestamp,
                                 "Job not found", "Job not found", "Job not found")
                status = ""
            else:
                res = list(map(str.strip, stdout.splitlines()[2].split()))
                status = res[4]
                run_time = res[5]
                job_name = res[2]
    elif config['general']['cluster'].lower() == "hamburg":
        job_cmd = "qstat -j " + str(jobid)
        process = subprocess.run(job_cmd, shell=True, capture_output=True)
        if process.returncode == 0:
            stdout = process.stdout.decode("utf-8")
            stderr = process.stderr.decode("utf-8")
            t = stdout[stdout.find("job_name:"):]
            tt = t[:t.find("\n")]
            status = "RUNNING"
            run_time = "???"
            job_name = tt.split(":")[1].strip()
        else:
            job_cmd = "qacct -j " + str(jobid)
            process = subprocess.run(job_cmd, shell=True, capture_output=True)
            if process.returncode == 0:
                stdout = process.stdout.decode("utf-8")
                m = qacct_jn.match(stdout)
                job_name = m.group().split()[1]
                failed = qacct_failed.match(stdout).group().split()[1]
                exit_stat = qacct_exit_stat.match(stdout).group().split()[1]
                run_time = "???"
                if failed == 0 and exit_stat == 0:
                    status = "COMPLETED"
                else:
                    status = "FAILED"
            else: 
                status = "qacct not accessible"
                run_time = "qacct not accessible"
                job_name = "qacct not accessible"

        #raise ValueError("SGE job info not implemented yet!")
    else:
        raise ValueError("Unkown cluster `" +config['general']['cluster']+ "` !")

    out = out.format(jobid, os.path.abspath(loc), timestamp,
                     status, run_time, job_name)
    with open(fn, 'w') as f:
        f.write(out)
    return out, status
