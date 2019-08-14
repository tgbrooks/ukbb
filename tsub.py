#!/usr/bin/env python
import argparse
import subprocess
import re
import time

parser = argparse.ArgumentParser(description="Runs bsub but watches to make sure the job is successfully submitted and quits if it doesn't start soon enough.")
parser.add_argument("--timeout", help="time in seconds to wait until giving up watching for the job to enter RUN state", default=30, type=int)
parser.add_argument("command", help="command to pass to bsub to run", nargs=argparse.REMAINDER)

args = parser.parse_args()

# Run the command
command = "bsub " + ' '.join(args.command)
result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)

m = re.match("Job <([\d]+)> is submitted to", result.stdout.decode())
if m:
    job_id = m.groups()[0]
    print(f"Job {job_id} successfully submitted. Watching for job to start running.")
    
    for i in range(args.timeout):
        time.sleep(1)
        bjobs_result = subprocess.run(f"bjobs {job_id}", shell=True, check=True, stdout=subprocess.PIPE)
        if bjobs_result.stdout.decode().endswith("is not found"):
            print("Not yet in bjobs. Waiting")
        else:
            lines = bjobs_result.stdout.decode().splitlines()
            if len(lines) < 2:
                print("Not yet in bjobs. Waiting")
            _job_id, user, stat, queue, *rest = lines[1].split()
            if _job_id == job_id:
                print(f"Job {job_id} in queue with status {stat}")
                if stat == "RUN":
                    print(f"Job started running successfully")
                    exit(0)
    print("Never saw the job in the RUN state in queue")

    # Can't actually do the following since it bhist needs to be on node
    ## Check bhist in case it did run but finished very quickly
    #bhist_result = subprocess.run(f"bhist {job_id}", shell=True, check=True, stdout=subprocess.PIPE)
    #if bhist_result.stdout.decode().startswith("No matching job found"):
    #    print("No trace of the job to be found in bhist either. Declaring failure to submit")
    #    exit(1)
    #else:
    #    lines = bhist_result.stdout.decode().splitlines()
    #    if len(lines) < 2:
    #        print("No trace of the job to be found in bhist either. Declaring failure to submit")
    #        exit(1)
    #    else:
    #        _job_id, user, name, *rest = lines[1].split()
    #        if _job_id == job_id:
    #            print("Found job in bhist, assuming it successfully started running")
    #            exit(0)
    #        else:
    #            print("No trace of the job to be found in bhist either. Declaring failure to submit")
    #            exit(1)

    exit(1)

else:
    print("Failed submission of job")
    print(result.stdout.decode())
