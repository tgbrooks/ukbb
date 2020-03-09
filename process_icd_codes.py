#!/usr/bin/env python
import argparse

description = '''
Gather ICD10/9 codes from the datatable

The result is a table of entries, one for each occurance of an ICD10/9 code:
ID  ICD10_code  first_date
or (if -v 9 is passed)
ID  ICD9_code  first_date

So a participant may have multiple entries, or none
'''

parser = argparse.ArgumentParser(description)
parser.add_argument("-t", "--table", help="UKBB table file to read in", required=True)
parser.add_argument("-o", "--output", help="output tab-separated file to write to", required=True)
parser.add_argument("-v", "--version", help="whether to use ICD 9 or ICD10 codes", default="10", choices=["10", "9"])

args = parser.parse_args()

ICD10_FIELD = 41270
ICD10_FIRST_DATE_FIELD = 41280
ICD9_FIELD = 41270
ICD9_FIRST_DATE_FIELD = 41281

if args.version == "10":
    CODE_FIELD = ICD10_FIELD
    FIRST_DATE_FIELD = ICD10_FIRST_DATE_FIELD
    CODE_NAME = "ICD10_code"
else:
    CODE_FIELD = ICD9_FIELD
    FIRST_DATE_FIELD = ICD9_FIRST_DATE_FIELD
    CODE_NAME = "ICD9_code"

import pandas

# These two files contain the information describing the fields in the UKBB
# they are downloaded from:
# http://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=19
# http://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=87
# http://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv 
if args.version == "10":
    codings = pandas.read_csv("../icd10_coding.txt", index_col=0)
else:
    codings = pandas.read_csv("../icd9_coding.txt", index_col=0)
fields = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col=2)

data = pandas.read_csv(args.table, sep="\t", index_col=0)
data.index.rename("ID", inplace=True)

num_entries = fields.loc[CODE_FIELD].Array


processed_data = []

icd10code_fields = [f"f.{CODE_FIELD}.0.{i}" for i in range(num_entries)]
first_date_fields = [f"f.{FIRST_DATE_FIELD}.0.{i}" for i in range(num_entries)]
for i, (code_field, date_field) in enumerate(zip(icd10code_fields, first_date_fields)):
    print(f"Processing {i} of {num_entries}")
    code = data[code_field]
    date = data[date_field]

    valid = ~code.isna()
    code = code[valid]
    date = date[valid]
    entries = pandas.DataFrame(CODE_NAME: code, "first_date": date})
    processed_data.append(entries)

all_data = pandas.concat(processed_data)
all_data.sort_index(inplace=True, kind="mergesort")
all_data.to_csv(args.output, sep="\t")
