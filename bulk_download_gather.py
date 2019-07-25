#!/usr/bin/env python

# Reads in the given tsv file and generates the list of participant IDs and Data files to download
import pandas

DATA_FILE = "data/ukb32828.tab"
OUTPUT_FILE = "data/bulk_download_list.txt"

# List of columns to extract the raw data locations from 
DESIRED_FILE_COLUMNS = ["f.90004.0.0"]

data = pandas.read_csv(DATA_FILE, sep="\t", index_col=0)


file_columns = [data[column].dropna() for column in DESIRED_FILE_COLUMNS]
[print(f"Found {len(extracted)} entries in {col}") for extracted, col in zip(file_columns, DESIRED_FILE_COLUMNS)]

output = pandas.concat(file_columns)
output.to_csv(OUTPUT_FILE, sep=" ", header=False)
