#!/usr/bin/env python
import pandas
import numpy

import pathlib

import argparse

parser = argparse.ArgumentParser(description="Compute RA, L5, M10, L5_time, M10_time, IS and IV stats for each subject given")
parser.add_argument("subject_directory", help="directory which contains subject actigraphy files")
parser.add_argument("output_path", help="file to write the summary stats out to as a tsv file")
parser.add_argument("--subject_ids", help="file with list of subject IDs to run, ignore other subjects in the directory", default=None)

args = parser.parse_args()
subject_ids = None if args.subject_ids is None else [ID.strip() for ID in open(args.subject_ids).readlines()]

subject_dir = pathlib.Path(args.subject_directory)

PERIODS = ["1Min", "10Min", "30Min", "60Min", "120Min"]

if subject_ids is None:
    files = subject_dir.glob("*_90004_0_0.csv")
else:
    files = [subject_dir / f"{id}_90004_0_0.csv" for id in subject_ids]
    missing = [id for id, file in zip(files, subject_ids) if not files.exists()]
    if missing:
        print(f"Missing data for following {len(missing)} subjects:\n{', '.join(missing)}")
        print("Skipping those ids")

    files = [id for id, file in zip(files, subject_ids) if files.exists()]

output = pandas.DataFrame([], columns=["L5", "M10", "RA", "L5_time", "M10_time"]
                                    + [f"IS_{period}" for period in PERIODS]
                                    + [f"IV_{period}" for period in PERIODS]
                                    + ["hours_of_data", "largest_contiguous_stretch", "percent_valid_best_72hrs"])
for file in files:
    subject_id = file.name.split("_")[0]

    data = pandas.read_csv(file, sep=",")

    # Data is every 5s from a start time given at the top of the file in the header
    header = data.columns[0]
    _, start_time, end_time, _ = header.split(' - ')
    start_time = pandas.to_datetime(start_time)
    end_time = pandas.to_datetime(end_time)
    runtime = (end_time - start_time).total_seconds()
    index = start_time + pandas.to_timedelta(numpy.arange(0, runtime+1, 5, dtype=int), unit="s")

    data = data.set_index(index)
    data.columns = ["acceleration", "imputed"]
    data.imputed = data.imputed.astype('bool')

    data_H = data.resample("1H").mean()
    data_H["time"] = [t.hour for t in data_H.index.time]

    #mask = util.mask_inactivity(data_H)
    #mask = numpy.array([True for i in range(len(data_H.index))]) # For now, nothing is masked. Assumed everyone wears it all the time (!)
    mask = ~(data_H["imputed"] == 1.0)
    movement = data_H['acceleration']

    mean = movement[mask].mean()
    variance = ((movement[mask] - mean)**2).sum()  / len(movement[mask])

    # IS and IV
    # See https://www.sciencedirect.com/science/article/pii/0006322390905235
    # for definitions of all of these stats
    movement_with_nans = movement.copy()
    movement_with_nans[~mask] = float("NaN")
    hourly_diffs = (movement_with_nans.diff()**2).dropna()
    IV = hourly_diffs.sum() / len(hourly_diffs) / variance
    def IV_IS(period):
        data_resampled = data.resample(period).mean()

        # TODO: get a better movement mask
        #mask = util.mask_inactivity(data_resampled)
        mask = numpy.array([True for i in range(len(data_resampled.index))]) # For now, nothing is masked. Assumed everyone wears it all the time (!)

        movement = data_resampled['acceleration']
        mean = movement[mask].mean()
        variance = ((movement[mask] - mean)**2).sum()  / len(movement[mask])

        movement_with_nans = movement.copy()
        movement_with_nans[~mask] = float("NaN")
        hourly_diffs = (movement_with_nans.diff()**2).dropna()
        IV = hourly_diffs.sum() / len(hourly_diffs) / variance

        start = data_resampled.index[0]
        periods_per_day = pandas.to_timedelta("24H") / period
        data_resampled["time"] = [((t - start) / period) % (periods_per_day)  for t in data_resampled.index]
        by_time = data_resampled[mask].groupby("time")
        IS = ((by_time['acceleration'].mean() - mean)**2).sum() / len(by_time.mean().index) / variance
        return IV, IS

    results = {}
    for period in PERIODS:
        IV, IS = IV_IS(period)
        results[f"IV_{period}"] = IV
        results[f"IS_{period}"] = IS

    doubled = data_H.groupby("time").mean()["acceleration"].values
    doubled = numpy.concatenate([doubled, doubled])
    cumulative = numpy.concatenate([[0], numpy.cumsum(doubled)])

    # Minimum activity over any 5 hour period, taken from averaging over all days
    L5_time = numpy.argmin(cumulative[5:] - cumulative[:-5])
    L5 = numpy.min(cumulative[5:] - cumulative[:-5])
    # Maximum activity over any 10 hour period, taken from averaging over all days
    M10_time = numpy.argmax(cumulative[10:] - cumulative[:-10])
    M10 = numpy.max(cumulative[10:] - cumulative[:-10])

    # convert to hour of the day, taking the midpoint
    L5_time = (L5_time + 2.5) % 24
    M10_time = (M10_time + 5) % 24

    RA = (M10 - L5)/(M10 + L5)

    ## Assess the data quality
    hours_of_data = numpy.sum(~data.imputed) * 5 / 60 / 60

    if len(data.index) > 1:
        padded = numpy.concatenate([[False], data.imputed, [False]])
        diff = numpy.diff(padded.astype('int'))
        starts, = numpy.where(diff == 1)
        ends, = numpy.where(diff == -1)
        if len(starts) > 0:
            largest_contiguous_stretch = numpy.max(ends - starts) * 5 / 60 / 60
        else:
            largest_contiguous_stretch = 0
    else:
        largest_contiugous_stretch = 0

    three_days = 72 * 60*60 // 5 # Number of readings in this time
    if len(data.index) > three_days:
        # Find the 72hrs with the most data availability
        cumulative_good = numpy.concatenate([[0],numpy.cumsum(~data.imputed)])
        percent_valid_best_72hrs = numpy.max(cumulative_good[three_days:] - cumulative_good[:-three_days]) / three_days
    else:
        percent_valid_best_72hrs = hours_of_data / 72 # Don't actually have 72 hours of data

    results.update( {"L5": L5, "M10": M10, "RA": RA, "L5_time": L5_time, "M10_time": M10_time,
                    "hours_of_data": hours_of_data,
                    "largest_contiguous_stretch": largest_contiguous_stretch,
                    "percent_valid_best_72hrs": percent_valid_best_72hrs} )
    output.loc[subject_id] = results

output.to_csv(args.output_path, sep="\t")
