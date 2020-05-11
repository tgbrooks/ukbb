#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(description="Aggregate the summary output of each participant into a tab-separated spreadsheet")
parser.add_argument("input_dir", help="path to input directory of participant accelerometer analyses")
parser.add_argument("output_path", help="path to the target tab-separated output spreadsheet of aggregated summary data.")
parser.add_argument("--file_suffix", help="suffix of all files you want to aggregate", default=["_90001_0_0-summary.json"], nargs="+")
parser.add_argument("--seasonal", help="use seasonal repeats",  action="store_const", const=True, default=False)

args = parser.parse_args()

import pandas
import pathlib
import json

# Gather all the files together into a dict ID -> data
suffix = args.file_suffix[0]
if suffix.endswith(".json"):
    data = {}
    for file_suffix in args.file_suffix:
        for input_path in pathlib.Path(args.input_dir).glob("*" + file_suffix):
            if args.seasonal:
                ID = input_path.name.split("_")[0]
                instance =  input_path.name.split("_")[2]
                ID = f"{ID}.{instance}"
            else:
                ID = input_path.name[:-len(file_suffix)]
            try:
                with open(input_path) as input_file:
                    data[ID] = json.load(input_file)
            except Exception as e:
                print(f"Error with file {input_path}. SKIPPING")
                print(e)
                continue
    print(f"Loaded {len(data)} accelerometer summaries")

    # Aggregate the data from many dicts into one dataframe
    # filling in NaNs for any values that are in one dict and not another
    aggregate = pandas.DataFrame.from_dict(data, orient="index")

else:
    # Aggregate from tab-separated files instead
    data = []
    for file_suffix in args.file_suffix:
        for input_path in pathlib.Path(args.input_dir).glob("*" + file_suffix):
            ID = input_path.name[:-len(file_suffix)]
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
