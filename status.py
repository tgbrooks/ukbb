#!/usr/bin/env python
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Check status of job via bjobs")
parser.add_argument("job_id", help="id of job to check")
args = parser.parse_args()

bjobs_output = subprocess.run(f"bjobs -w {args.job_id}", shell=True, check=True, stdout=subprocess.PIPE)
out = bjobs_output.stdout.decode()

def fail():
    import pathlib
    pathlib.Path(f"job.{args.job_id}.fail").write_text(f"FAILED JOB\n{out}")
    print('failed')

if "is not found" in out:
    fail()
    #print("failed")
elif any(status in out for status in ["EXIT", "USUSP", "SSUSP", "PSUSP", "ZOMBI", "UNKWN"]):
    fail()
    #print("failed")
elif "DONE" in out:
    print("success")
elif any(status in out for status in ["PEND", "RUN", "PROV", "WAIT"]):
    print("running")
else:
    #print("failed")
    fail()
