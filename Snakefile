import subprocess
import pathlib
import sys

print("Starting pipeline")
target_eids = [line.strip()  for line in open("../eids_for_actigraphy.txt").readlines()]

BATCH_SIZE = 20
batched_eids = [target_eids[i:i+BATCH_SIZE] for i in range(0, len(target_eids), BATCH_SIZE)]
eids_to_batches = {eid: batch_num for batch_num, batch in enumerate(batched_eids) for eid in batch}

ACTIVITY_FEATURES_BATCH_SIZE = 1000
activity_features_batches = [target_eids[i:i+ACTIVITY_FEATURES_BATCH_SIZE] for i in range(0, len(target_eids), ACTIVITY_FEATURES_BATCH_SIZE)]

print(f"Found {len(batched_eids)} batches to consider")
print(f"Found {len(activity_features_batches)} activity feature batches to consider")

rule all:
    input:
        "../processed/activity_features_aggregate.txt", #NOTE: this one takes a long time, even if there is no work to be done, comment if already finished
        "../processed/activity_features_aggregate_seasonal.txt", #NOTE: slow, as above
        "../processed/ukbb_mental_health.h5",
        "../processed/ukbb_employment.h5",
        "../processed/ukbb_data_table.h5",
        "../processed/ukbb_icd10_entries.txt",
        "../processed/ukbb_icd9_entries.txt",
        "../processed/ukbb_employment_history.txt",
        "../processed/ukbb_self_reported_conditions.txt",
        "../processed/ukbb_medications.txt",

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
        def eid_to_target(eid):
            id, instance = eid.split('\t')
            return f"{id} 90001_{instance}_0"
        batch_file.write_text('\n'.join(eid_to_target(eid) for eid in batched_eids[int(wildcards.batch_num)])) # Output list of eids we want to download

        ukbfetch_path = pathlib.Path("../biobank_utils/ukbfetch").resolve()
        key_path = pathlib.Path("../k50398.key").resolve()
        command = f"{ukbfetch_path} -bbatch_list.txt -a{key_path}"

        log_file = open(log[0], "w")
        print(f"Running:\n{command}\n")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

tab = '\t'
rule process_accelerometery:
    input:
        lambda wildcards: f"../data/raw_actigraphy/batch{eids_to_batches[wildcards.id + tab + wildcards.instance]}/"
    output:
        protected(expand("../processed/acc_analysis/{{id}}_90001_{{instance}}_0-{files}", files=["nonWearBouts.csv.gz", "timeSeries.csv.gz", "summary.json"]))
    log:
        "log/{id}.{instance}.accelerometer.analysis.log"
    run:
        working_path = pathlib.Path("../biobankAccelerometerAnalysis/").resolve()
        input_path = (pathlib.Path(input[0]) / f"{wildcards.id}_90001_{wildcards.instance}_0.cwa").resolve()
        output_path = (pathlib.Path.cwd() / output[0]).parent.resolve()
        command = f"python accProcess.py {input_path} --outputFolder {output_path}/ --timeSeriesDateColumn True --modifyForDaylightSavings False"
        print(f"Running:\n{command}\n")
        log_file = open(log[0], "w")
        subprocess.run(command, stdout=log_file, stderr=log_file, shell=True, cwd=working_path, check=True)

def eid_to_filename(eid):
    id, instance = eid.split("\t")
    return f"{id}_90001_{instance}_0"
rule activity_features_batch:
    input:
        lambda wildcards: expand("../processed/acc_analysis/{filename}-timeSeries.csv.gz",
                    filename=[eid_to_filename(eid) for eid in activity_features_batches[int(wildcards.batch)]])
    output:
        touch("../processed/activity_features/batch{batch}")
    run:
        import activity_features
        for file_path, eid in zip(input, activity_features_batches[int(wildcards.batch)]):
            id, instance = eid.split("\t")
            try:
                if id == '0':
                    activity_features.run(file_path,
                        f"../processed/activity_features/{id}.json",
                        f"../processed/activity_features/{id}.by_day.txt")
                else:
                    activity_features.run(file_path,
                            f"../processed/activity_features/seasonal/{id}.{instance}.json",
                            f"../processed/activity_features/seasonal/{id}.{instance}.by_day.txt")
            except Exception as e:
                print(f"An Exception occured processed {file_path}")
                print(e)
                raise

rule aggregate:
    input:
        expand(rules.activity_features_batch.output, batch=range(len(activity_features_batches)))
    output:
        activity_features = protected("../processed/activity_features_aggregate.txt"),
        activity_summary = protected("../processed/activity_summary_aggregate.txt"),
        activity_by_day = protected("../processed/activity_by_day.txt")
    shell:
        "./aggregate.py ../processed/acc_analysis/ {output.activity_summary} && "
        "./aggregate.py ../processed/activity_features/ {output.activity_features} --file_suffix .json && "
        "./aggregate.py ../processed/activity_features/ {output.activity_by_day} --file_suffix .by_day.txt"

rule aggregate_seasonal:
    input:
        expand(rules.activity_features_batch.output, batch=range(len(activity_features_batches)))
    output:
        activity_features = protected("../processed/activity_features_aggregate_seasonal.txt"),
        activity_summary = protected("../processed/activity_summary_aggregate_seasonal.txt"),
        activity_by_day = protected("../processed/activity_by_day_seasonal.txt")
    shell:
        "./aggregate.py ../processed/acc_analysis/ {output.activity_summary}  --file_suffix _90001_1_0-summary.json  _90001_2_0-summary.json _90001_3_0-summary.json _90001_4_0-summary.json  --seasonal && "
        "./aggregate.py ../processed/activity_features/seasonal/ {output.activity_features} --file_suffix .json && "
        "./aggregate.py ../processed/activity_features/seasonal/ {output.activity_by_day} --file_suffix .by_day.txt"

rule process_ukbb_table:
    input:
        "../data/ukb34939.tab",
        "../data/ukb41264.tab",
        "../data/ukb44535.tab",
        "../Data_Dictionary_Showcase.csv",
        "../Codings_Showcase.csv",
    output:
        mental_health_table = "../processed/ukbb_mental_health.h5",
        employment_table = "../processed/ukbb_employment.h5",
    resources:
        mem_mb = 180000
    shell:
        "./process_ukbb_table.py -t {input[0]} {input[1]} {input[2]} -o {output.mental_health_table} -s mental_health_fields &&"
        "./process_ukbb_table.py -t {input[0]} {input[1]} {input[2]} -o {output.employment_table} -s employment_fields"

rule process_ukbb_table_general:
    input:
        "../data/ukb34939.tab",
        "../data/ukb41264.tab",
        "../data/ukb44535.tab",
        "../Data_Dictionary_Showcase.csv",
        "../Codings_Showcase.csv",
    output:
        general_table = "../processed/ukbb_data_table.h5",
    resources:
        mem_mb = 180000
    shell:
        "./process_ukbb_table.py -t {input[0]} {input[1]} {input[2]} -o {output.general_table} -s all_general_fields"

rule process_icd10_codes:
    input:
        "../data/ukb41264.tab",
    output:
        "../processed/ukbb_icd10_entries.txt"
    resources:
        mem_mb = 20000
    shell:
        "./process_icd_codes.py -t {input} -o {output}"

rule process_icd9_codes:
    input:
        "../data/ukb41264.tab",
    output:
        "../processed/ukbb_icd9_entries.txt"
    resources:
        mem_mb = 20000
    shell:
        "./process_icd_codes.py -t {input} -o {output} -v 9"

rule process_self_reported_conditions:
    input:
        "../data/ukb41264.tab",
    output:
        "../processed/ukbb_self_reported_conditions.txt"
    resources:
        mem_mb = 20000
    shell:
        "./process_self_reported_conditions.py -t {input} -o {output}"

rule process_medications:
    input:
        "../data/ukb41264.tab",
    output:
        "../processed/ukbb_medications.txt"
    resources:
        mem_mb = 20000
    shell:
        "./process_medications.py -t {input} -o {output}"

rule process_employment_history:
    input:
        "../data/ukb41264.tab",
    output:
        "../processed/ukbb_employment_history.txt"
    resources:
        mem_mb = 20000
    shell:
        "./process_employment.py -t {input} -o {output}"
