#!/usr/bin/env python
import argparse

description = '''
Gather ICD10 codes from the datatable

The result is a table of entries, one for each occurance of an ICD10 code:
ID  ICD10_code  first_date

So a participant may have multiple entries
'''

parser = argparse.ArgumentParser(description)
parser.add_argument("-t", "--table", help="UKBB table file to read in", required=True)
parser.add_argument("-o", "--output", help="output tab-separated file to write to", required=True)
parser.add_argument("-ids", help="list of IDs to gather", required=False)

args = parser.parse_args()

ICD10_FIELD = 41270
ICD10_FIRST_DATE_FIELD = 41280 # NOTE: we did not request this field


import pandas

# These two files contain the information describing the fields in the UKBB
# they are downloaded from:
# http://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=19
# http://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv 
codings = pandas.read_csv("../icd10_coding.txt", index_col=0)
fields = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col=2)

data = pandas.read_csv(args.table, sep="\t", index_col=0)
data.index.rename("ID", inplace=True)

num_entries = fields.loc[ICD10_FIELD].Array


processed_data = []

icd10code_fields = [f"f.{ICD10_FIELD}.0.{i}" for i in range(num_entries)]
first_date_fields = [f"f.{ICD10_FIRST_DATE_FIELD}.0.{i}" for i in range(num_entries)]
for i, (code_field, date_field) in enumerate(zip(icd10code_fields, first_date_fields)):
    print(f"Processing {i} of {num_entries}")
    code = data[code_field]
    #date = data[date_field]

    valid = ~code.isna()
    code = code[valid]
    #date = date[valid]
    #together = pandas.DataFrame({"ICD10_code":code, "first_date": date})
    all = pandas.DataFrame({"ICD10": code})
    processed_data.append(all)

all_data = pandas.concat(processed_data)
all_data.sort_index(inplace=True, kind="mergesort")
all_data.to_csv(args.output, sep="\t")
