#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description="Take activity analyses and average them by date to get day-level sleep and activity level measures")
parser.add_argument("start_date", help="First date (in YYYY-MM-DD format) to process")
parser.add_argument("end_date", help="Last date (in YYYY-MM-DD format) to process")
parser.add_argument("activity_analysis_directory", help="folder containing the activity analysis timeseries data")
parser.add_argument("output_file", help="file to output the by-date summary")
parser.add_argument("--summary_file", help="tab-separated file with the agregated summary output of the activity analysis, contains dates of subjects; if none provided, looks through all in directory (slow)", default=None)
parser.add_argument("--activation_date", help="select only those whose start date of activity data is this date (optional)", default=None)
parser.add_argument("--quantile", help="quantile to take (eg: 0.9 is 90th percentile). By default compute the mean instead", default="mean")
parser.add_argument("--seasonal", help="use the seasonal repeats (90001_1_0, 90001_2_0, 90001_3_0, and 9000_1_4_0 files)", default=False, action="store_const", const=True)
args = parser.parse_args()

import pathlib
import pandas
import json
import sys

start_time = pandas.to_datetime(args.start_date)
end_time = pandas.to_datetime(args.end_date) + pandas.to_timedelta("1D")

directory = pathlib.Path(args.activity_analysis_directory)

if args.activation_date is not None:
    activation_date = pandas.to_datetime(args.activation_date).date()

if args.summary_file is not None:
    # Find which eids to use
    summary = pandas.read_csv(args.summary_file, sep="\t", index_col=0, parse_dates=["file-startTime", "file-endTime"])
    in_range = (summary['file-startTime'] < end_time) & (summary['file-endTime'] >= start_time)
    if args.activation_date is not None:
        in_range &= summary['file-startTime'].dt.date == activation_date
    eids = summary.index[in_range]
else:
    # No aggregated summary data, therefore we will use all eids
    eids = None

columns = "acceleration,moderate,sedentary,sleep,tasks-light,walking,MET,temp,light".split(",")

# Collect and build the data
index = pandas.date_range(start_time, end_time, freq="30s", tz="Europe/London")
activity = pandas.DataFrame(0, columns=columns + ['count'], index=index)
activity_dfs = []

if args.seasonal:
    files = [f for i in [1,2,3,4] for f in directory.glob(f"*_90001_{i}_0-summary.json")]
else:
    files = directory.glob("*_90001_0_0-summary.json")

for filename in files:
    eid = str(filename.name).split("_")[0]
    instance = str(filename.name).split("_")[2]
    if eids is not None:
        if eid not in eids:
            # Skip all files not in our eid list
            continue

    try:
        summary_data = json.load((directory/ f"{eid}_90001_{instance}_0-summary.json").open())
        # Check date ranges
        file_start = pandas.to_datetime(summary_data['file-startTime'])
        file_end = pandas.to_datetime(summary_data['file-endTime'])
        if not ((file_start <= end_time) and (file_end >= start_time)):
            # skipping since not in range
            continue

        # Check start date
        if args.activation_date is not None:
            if file_start.date() != activation_date:
                continue

        # Quality control
        if not summary_data.get("quality-calibratedOnOwnData", False):
            print(f"Skipping eid {eid} due to missing calibration")
            continue
        if not summary_data.get("quality-goodWearTime", False):
            print(f"Skipping eid {eid} due to bad wear time")
            continue

        # Load the activity data
        activity_data = pandas.read_csv(directory / f"{eid}_90001_{instance}_0-timeSeries.csv.gz", parse_dates=[0], index_col=0)
        activity_data.rename(columns={activity_data.columns[0]:"acceleration"}, inplace=True)

        # Align time of all the samples by nearest 30s
        offset = activity_data.index[0] - activity_data.index[0].round('30s')
        activity_data.index = activity_data.index - offset

        # Make multi-index so that it can be concatenated together with the other dataframes
        # for processing into means, etc
        activity_data['eid'] = eid
        activity_data = activity_data.set_index(activity_data.index, activity_data.eid)

        # Add activity observed, but drop the imputed data since it's not good
        activity_data.loc[activity_data.imputed == 1,columns] = float("NaN")
        activity_dfs.append(activity_data.loc[start_time:end_time])
    except Exception as e:
        print(f"Exception during processing of EID {eid} {instance}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(f"Skipping EID {eid} {instance}", file=sys.stderr)
        continue

if activity_dfs:
    all_activity = pandas.concat(activity_dfs, sort=True)
    del activity_dfs # This can use a large amount of memory and it's now redundant
    grouped_activity = all_activity.groupby(level=0)
    if args.quantile == 'mean':
        summary_activity = grouped_activity.mean()
    else:
        try:
            quantile = float(args.quantile)
        except ValueError:
            print(f"Invalid 'quantile' parameter, need float 0-1, got `{args.quantile}`")
            exit(0)
        summary_activity = grouped_activity.quantile(quantile)


    counts = grouped_activity.count()
    summary_activity['counts'] = counts.sleep # Pick one collumn to use, they should all be the same

    summary_activity.to_csv(args.output_file, sep="\t")
else:
    print("Failed to find any files in the time range. Output blank file", file=sys.stderr)
    pathlib.Path(args.output_file).touch()
