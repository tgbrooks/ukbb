'''
Snakefile for performing just the Daylight saving time analysis
Must be run after the pipeline in `Snakefile`

Separate pipeline so that:
1) The first pipeline can be run independently and therefore shared between multiple analyses
2) This pipeline will be easier to run and debug without having such a large DAG to build
'''

import pandas
import subprocess

# Gather the EIDs used for this pipeline
target_eids = [line.strip()  for line in open("../eids_ordered_for_batching.txt").readlines()]

ACTIVITY_FEATURES_BATCH_SIZE = 1000
activity_features_batches = [target_eids[i:i+ACTIVITY_FEATURES_BATCH_SIZE] for i in range(0, len(target_eids), ACTIVITY_FEATURES_BATCH_SIZE)]

# Table of the Sundays where the DST transition occured at 1 am
dates_of_transitions = ['2013-03-31',
                        '2013-10-27',
                        '2014-03-30',
                        '2014-10-26',
                        '2015-03-29',
                        '2015-10-25',
                        '2016-03-27',
                        '2016-10-30']

rule all:
    input:
        expand("../processed/dst/week_of_{date}.txt", date=dates_of_transitions)

rule gather_near_transitions:
    input:
        expand("../processed/activity_analysis/batch{batch}", batch in range(len(activity_features_batches)))
    output:
        "../processed/dst/week_of_{date}.txt"
    run:
        transition = pandas.to_datetime(wildcards.date)
        start_time = (transition - pandas.to_timedelta("1W")).strftime("%Y-%m-%d")
        end_time = (transition + pandas.to_timedelta("1W")).strftime("%Y-%m-%d")
        command = f"./averages_over_timespan.py {start_time} {end_time} ../processed/activity_analysis/ ../processed/dst/week_of{wildcards.date}.txt ../processed/activity_summary_aggregate.txt"
        print("Running:")
        print(command)
        subprocess.run(command, shell=True)
