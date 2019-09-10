#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description="Take activity analyses and average them by date to get day-level sleep and activity level measures")
parser.add_argument("start_date", help="First date (in YYYY-MM-DD format) to process")
parser.add_argument("end_date", help="Last date (in YYYY-MM-DD format) to process")
parser.add_argument("activity_analysis_directory", help="folder containing the activity analysis timeseries data")
parser.add_argument("output_file", help="file to output the by-date summary")
parser.add_argument("summary_file", help="tab-separated file with the agregated summary output of the activity analysis, contains dates of subjects")

args = parser.parse_args()

import pathlib
import pandas
import json
import sys


summary = pandas.read_csv(args.summary_file, sep="\t", index_col=0, parse_dates=["file-startTime", "file-endTime"])
start_time = pandas.to_datetime(args.start_date)
end_time = pandas.to_datetime(args.end_date) + pandas.to_timedelta("1D")
in_range = (summary['file-startTime'] < end_time) & (summary['file-endTime'] >= start_time)
eids = summary.index[in_range]

columns = "moderate,sedentary,sleep,tasks-light,walking,MET,temp,light".split(",")

directory = pathlib.Path(args.activity_analysis_directory)
index = pandas.date_range(start_time, end_time, freq="30s", tz="Europe/London")
activity = pandas.DataFrame(0, columns=columns + ['count'], index=index)
activity_dfs = []
for eid in eids:
    try:
        summary_data = json.load((directory/ f"{eid}_90001_0_0-summary.json").open())
        if not summary_data.get("quality-calibratedOnOwnData", False):
            print(f"Skipping eid {eid} due to missing calibration")
            continue
        if not summary_data.get("quality-goodWearTime", False):
            print(f"Skipping eid {eid} due to bad wear time")
            continue

        # Load the activity data
        activity_data = pandas.read_csv(directory / f"{eid}_90001_0_0-timeSeries.csv.gz", parse_dates=[0], index_col=0)

        # Align time of all the samples by nearest 30s
        offset = activity_data.index[0] - activity_data.index[0].round('30s')
        activity_data.index = activity_data.index - offset
        activity_data.index = activity_data.index.localize("Europe/London")

        # Make multi-index so that it can be concatenated together with the other dataframes
        # for processing into means, etc
        activity_data['eid'] = eid
        activity_data = activity_data.set_index(activity_data.index, activity_data.eid)

        # Add activity observed, but drop the imputed data since it's not good
        activity_data.loc[activity_data.imputed == 1,columns] = float("NaN")
        activity_dfs.append(activity_data.loc[start_time:end_time, columns])
    except:
        print(f"Exception during processing of EID {eid}", file=sys.stderr)
        raise

if activity_dfs:
    all_activity = pandas.concat(activity_dfs, sort=True)
    del activity_dfs # This can use a large amount of memory and it's now redundant
    grouped_activity = all_activity.groupby(level=0)
    mean_activity = grouped_activity.mean()
    std_activity = grouped_activity.std()
    counts = grouped_activity.count()
    mean_activity['counts'] = counts.sleep # IDK which collumn to use, they should all be the same

    mean_activity.to_csv(args.output_file, sep="\t")
else:
    print("Failed to find any files in the time range. Output blank file", file=sys.stderr)
    pathlib.Path(args.output_file).touch()

