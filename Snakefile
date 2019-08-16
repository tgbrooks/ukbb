import subprocess
import pathlib

target_eids = [line.strip()  for line in open("../all_eids_with_actigraphy.txt").readlines()]
#target_eids = [line.strip()  for line in open("../target_eids.txt").readlines()]

unprocessed_eids = [eid for eid in target_eids if not pathlib.Path(f"../processed/acc_analysis/{eid}_90001_0_0-timeSeries.csv.gz").exists()]
set_unprocessed_eids = set(unprocessed_eids)
for_batching = unprocessed_eids + [eid for eid in target_eids if eid not in set_unprocessed_eids]

BATCH_SIZE = 20
batched_eids = [for_batching[i:i+BATCH_SIZE] for i in range(0, len(for_batching), BATCH_SIZE)]
eids_to_batches = {eid: batch_num for batch_num, batch in enumerate(batched_eids) for eid in batch}

rule all_done:
    input:
        "done.txt"

rule ukbfetch_download_raw:
    output:
        temp(directory("../data/raw_actigraphy/batch{batch_num}/"))
    log:
        "log/ukbfetch.batch.{batch_num}.log"
    resources:
        download=1
    priority:
        10 # Downloads tend to be the bottleneck, so let's make sure we do them
    run:
        working_path = pathlib.Path("../data/raw_actigraphy/batch{batch_num}/").resolve()
        working_path.mkdir(exist_ok=True)
        batch_file = working_path / "batch_list.txt"
        batch_file.write_text('\n'.join(batched_eids[int(wildcards.batch_num)])) # Output list of eids we want to download

        ukbfetch_path = pathlib.Path("../biobank_utils/ukbfetch").resolve()
        key_path = pathlib.Path("../k50398.key").resolve()
        command = f"{ukbfetch_path} -b{batch_file} -d90001_0_0 -a{key_path}"

        log_file = open(log[0], "w")
        log_file.write(f"Running:\n{command}\n")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

rule process_accelerometery:
    input:
        lambda wildcards: f"../data/raw_actigraphy/batch{eids_to_batches[wildcards.id]}/"
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


rule all:
    input:
        expand(rules.process_accelerometery.output,
                    id=target_eids,
                    files=["timeSeries.csv.gz"])
    output:
        "done.txt"
    shell:
        "touch done.txt"
