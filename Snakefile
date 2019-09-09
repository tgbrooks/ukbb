import subprocess
import pathlib
import sys

print("Starting pipeline")
target_eids = [line.strip()  for line in open("../eids_ordered_for_batching.txt").readlines()]
#target_eids = [line.strip()  for line in open("../all_eids_with_actigraphy.txt").readlines()]
#target_eids = [line.strip()  for line in open("../target_eids.txt").readlines()]

BATCH_SIZE = 20
batched_eids = [target_eids[i:i+BATCH_SIZE] for i in range(0, len(target_eids), BATCH_SIZE)]
eids_to_batches = {eid: batch_num for batch_num, batch in enumerate(batched_eids) for eid in batch}

ACTIVITY_FEATURES_BATCH_SIZE = 1000
activity_features_batches = [target_eids[i:i+ACTIVITY_FEATURES_BATCH_SIZE] for i in range(0, len(target_eids), ACTIVITY_FEATURES_BATCH_SIZE)]

print(f"Found {len(batched_eids)} batches to consider")


rule all:
    input:
        "../processed/activity_features_aggregate.txt"

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
        working_path = pathlib.Path(f"../data/raw_actigraphy/batch{wildcards.batch_num}/").resolve()
        working_path.mkdir(exist_ok=True)
        batch_file = working_path / "batch_list.txt"
        batch_file.write_text('\n'.join(eid + " 90001_0_0" for eid in batched_eids[int(wildcards.batch_num)])) # Output list of eids we want to download

        ukbfetch_path = pathlib.Path("../biobank_utils/ukbfetch").resolve()
        key_path = pathlib.Path("../k50398.key").resolve()
        command = f"{ukbfetch_path} -bbatch_list.txt -a{key_path}"

        log_file = open(log[0], "w")
        print(f"Running:\n{command}\n")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

rule process_accelerometery:
    input:
        lambda wildcards: f"../data/raw_actigraphy/batch{eids_to_batches[wildcards.id]}/"
    output:
        protected(expand("../processed/acc_analysis/{{id}}_90001_0_0-{files}", files=["nonWearBouts.csv.gz", "timeSeries.csv.gz", "summary.json"]))
    log:
        "log/{id}.accelerometer.analysis.log"
    run:
        working_path = pathlib.Path("../biobankAccelerometerAnalysis/").resolve()
        input_path = (pathlib.Path(input[0]) / f"{wildcards.id}_90001_0_0.cwa").resolve()
        output_path = (pathlib.Path.cwd() / output[0]).parent.resolve()
        command = f"python accProcess.py {input_path} --outputFolder {output_path}/ --timeSeriesDateColumn True --modifyForDaylightSavings False"
        print(f"Running:\n{command}\n")
        log_file = open(log[0], "w")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

rule activity_features_batch:
    input:
        lambda wildcards: expand("../processed/acc_analysis/{id}_90001_0_0-timeSeries.csv.gz", id=activity_features_batches[int(wildcards.batch)])
    output:
        touch("../processed/activity_features/batch{batch}")
    run:
        import activity_features
        for file_path, id in zip(input, activity_features_batches[int(wildcards.batch)]):
            activity_features.run(file_path, f"../processed/activity_features/{id}.json")

rule aggregate:
    input:
        expand(rules.activity_features_batch.output, batch=range(len(activity_features_batches)))
    output:
        protected("../processed/activity_features_aggregate.txt")
    shell:
        "./aggregate.py ../processed/activity_features/ {output} --file_suffix .json"
