import subprocess
import pathlib

target_eids = [line.strip()  for line in open("../target_eids.txt").readlines()]

rule all:
    input:
        expand("../processed/acc_analysis/{id}_90001_0_0-{files}",
                    id=target_eids,
                    files=["nonWearBouts.csv.gz", "timeSeries.csv.gz", "summary.json"]),

rule ukbfetch_download_raw:
    output:
        "../data/raw_actigraphy/{id}_90001_0_0.cwa"
    log:
        "log/{id}.ukbfetch.log"
    resources:
        download=1
    run:
        working_path = (pathlib.Path.cwd() / "../data/raw_actigraphy/").resolve()
        ukbfetch_path = (pathlib.Path.cwd() / "../biobank_utils/ukbfetch").resolve()
        key_path = (pathlib.Path.cwd() / "../k50398.key").resolve()
        command = f"{ukbfetch_path} -e{wildcards.id} -d90001_0_0 -a{key_path}"
        log_file = open(log[0], "w")
        log_file.write(f"Running:\n{command}\n")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

rule process_accelerometery:
    input:
        "../data/raw_actigraphy/{id}_90001_0_0.cwa"
    output:
        expand("../processed/acc_analysis/{{id}}_90001_0_0-{files}", files=["nonWearBouts.csv.gz", "timeSeries.csv.gz", "summary.json"])
    log:
        "log/{id}.accelerometer.analysis.log"
    run:
        working_path = (pathlib.Path.cwd() / "../biobankAccelerometerAnalysis/").resolve()
        input_path = (pathlib.Path.cwd() / input[0]).resolve()
        output_path = (pathlib.Path.cwd() / output[0]).parent.resolve()
        command = f"python accProcess.py {input_path} --outputFolder {output_path}/ --timeSeriesDateColumn True --modifyForDaylightSavings False"
        log_file = open(log[0], "w")
        log_file.write(f"Running:\n{command}\n")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)
