#!/usr/bin/env python
import argparse
import subprocess
import re
import time
import pathlib

parser = argparse.ArgumentParser(description="Runs bsub but watches to make sure the job is successfully submitted and quits if it doesn't start soon enough.")
parser.add_argument("--timeout", help="time in seconds to wait until giving up watching for the job to enter RUN state", default=30, type=int)
parser.add_argument("command", help="command to pass to bsub to run", nargs=argparse.REMAINDER)

args = parser.parse_args()

import random
temp_num = random.randint(0, 1_000_000)
temp_file = pathlib.Path(f"log/tsub/tmp.{temp_num}.started")
temp_file.touch()

# Run the command
with open(temp_file, "w") as temp:
    command = "bsub " + ' '.join(args.command)
    temp.write("Running:\n")
    temp.write(command)
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
    except CalledProcessError:
        temp.write("CALLED BSUB COMMAND FAILED")
    temp.write(result.stdout.decode())

m = re.match("Job <([\d]+)> is submitted to", result.stdout.decode())
if m:
    job_id = m.groups()[0]
    print(job_id)
    temp_file.unlink()
    exit(0)
    
    for i in range(args.timeout):
        time.sleep(1)
        bjobs_result = subprocess.run(f"bjobs {job_id}", shell=True, check=True, stdout=subprocess.PIPE)
        if bjobs_result.stdout.decode().endswith("is not found"):
            continue
        else:
            lines = bjobs_result.stdout.decode().splitlines()
            if len(lines) < 2:
                continue

            _job_id, user, stat, queue, *rest = lines[1].split()
            if _job_id == job_id:
                if stat == "RUN":
                    print(job_id)
                    #print(f"Job {job_id} started running successfully after {i} seconds")
                    temp_file.unlink()
                    exit(0)
                elif stat == "PEND":
                    print(job_id)
                    #print(f"Job {job_id} queued successfully after {i} seconds")
                    temp_file.unlink()
                    exit(0)
                else:
                    #print(f"Job {job_id} in queue with status {stat}")
                    continue
    print("Never saw the job in the RUN or PEND state in queue")

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
    exit(1)
