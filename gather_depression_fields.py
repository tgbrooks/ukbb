#!/usr/bin/env python

import argparse

import pandas

parser = argparse.ArgumentParser(description="Generate a smaller table of the variables we care about")
parser.add_argument("-t", "--tsv", help="path to file of tab-separated data on the subjects", nargs="+")
parser.add_argument("-o", "--output", help="path to output to")

args = parser.parse_args()

data_tables = [pandas.read_csv(path, sep="\t", index_col=0)
                for path in args.tsv]
data = pandas.concat(data_tables, axis=1)

columns = {
    "ever_prolonged_depression": "f.20446.0.0",
    "professional_informed_of_depression": "f.20448.0.0",
    "age_at_first_episode": "f.20433.0.0",
    "recent_feelings_of_depression": "f.20510.0.0",
    "recent_low_energy": "f.20519.0.0",
    "sleep_change_worst_epsidoe": "f.20532.0.0",
    "sleep_too_much_worst_episode": "f.20534.0.0",
    "trouble_falling_asleep_worst_episode": "f.20533.0.0",
    "waking_early_worst_episode": "f.20535.0.0",
    "recent_sleep_troubles": "f.20517.0.0",
    "actigraphy_file": "f.90004.0.0",
    "birth_year": "f.34.0.0",
    "sex": "f.31.0.0",
    }

invert_columns = {val:key for key, val in columns.items()}
data = data.rename(columns=invert_columns)

small_data = data[list(columns.keys())]
small_data.to_csv(args.output, sep="\t")
