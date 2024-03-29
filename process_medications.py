#!/usr/bin/env python
import argparse

description = '''
Gather self-reported meidcations

The result is a table of entries, one for each reported non-cancer condition (field 20002)
ID medication_code
So a participant may have multiple entries, or none if they reported no conditions
'''

parser = argparse.ArgumentParser(description)
parser.add_argument("-t", "--table", help="UKBB table file to read in", required=True)
parser.add_argument("-o", "--output", help="output tab-separated file to write to", required=True)

args = parser.parse_args()


MEDICATION_FIELD = 20003
MEDICATOIN_CODING = 4 # Coded by coding number 4

import pandas

# These two file contain the information describing the fields in the UKBB
# they can be downloaded from 
# http://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv 
fields = pandas.read_csv("metadata/Data_Dictionary_Showcase.csv", index_col=2)

num_entries = fields.loc[MEDICATION_FIELD].Array

processed_data = []

for data in pandas.read_csv(args.table, sep="\t", index_col=0, chunksize=10_000, low_memory=False):
    data.index.rename("ID", inplace=True)
    for i in range(num_entries):
        condition_field = f"f.{MEDICATION_FIELD}.0.{i}"
        code = data[condition_field]

        valid = ~code.isna()
        code = code[valid]
        entries = pandas.DataFrame({"medication_code": code})
        processed_data.append(entries)

all_data = pandas.concat(processed_data)
all_data.sort_index(inplace=True, kind="mergesort")
all_data.to_csv(args.output, sep="\t")
