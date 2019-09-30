#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description="Aggregate the summary output of each participant into a tab-separated spreadsheet")
parser.add_argument("input_dir", help="path to input directory of participant accelerometer analyses")
parser.add_argument("output_path", help="path to the target tab-separated output spreadsheet of aggregated summary data.")
parser.add_argument("--file_suffix", help="suffix of all files you want to aggregate", default="_90001_0_0-summary.json")

args = parser.parse_args()

import pandas
import pathlib
import json

# Gather all the files together into a dict ID -> data
if args.file_suffix.endswith(".json"):
    data = {}
    for input_path in pathlib.Path(args.input_dir).glob("*" + args.file_suffix):
        ID = input_path.name[:-len(args.file_suffix)]
        with open(input_path) as input_file:
            data[ID] = json.load(input_file)
    print(f"Loaded {len(data)} accelerometer summaries")

    # Aggregate the data from many dicts into one dataframe
    # filling in NaNs for any values that are in one dict and not another
    aggregate = pandas.DataFrame.from_dict(data, orient="index")

else:
    # Aggregate from tab-separated files instead
    data = []
    for input_path in pathlib.Path(args.input_dir).glob("*" + args.file_suffix):
        ID = input_path.name[:-len(args.file_suffix)]
        id_data = pandas.read_csv(input_path, sep="\t", index_col=0)
        # Add the ID as a column
        id_data["ID"] = ID
        data.append(id_data)

    # Now concatenate all the ID-specific dataframe together
    aggregate = pandas.concat(data)

# Replace dashes with underscores for python convenience
aggregate.rename(columns=lambda s: s.replace('-', '_'))

# Output the aggregated data
aggregate.to_csv(args.output_path, sep="\t")
