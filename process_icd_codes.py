#!/usr/bin/env python
import argparse

description = '''
Gather ICD10/9 codes from the HESIN/HESIN_DIAG tables

The result is a table of entries, one for each occurance of an ICD10/9 code:
ID  ICD10_code  first_date
or (if -v 9 is passed)
ID  ICD9_code  first_date

So a participant may have multiple entries, or none
'''

parser = argparse.ArgumentParser(description)
parser.add_argument("--hesin", help="UKBB HESIN table file to read in", required=True)
parser.add_argument("--hesin_diag", help="UKBB HESIN diagnosis table file to read in", required=True)
parser.add_argument("-o", "--output", help="output tab-separated file to write to", required=True)
parser.add_argument("-v", "--version", help="whether to use ICD 9 or ICD10 codes", default="10", choices=["10", "9"])

args = parser.parse_args()

ICD10_FIELD = "diag_icd10"
ICD9_FIELD = "diag_icd9"

if args.version == "10":
    CODE_FIELD = ICD10_FIELD
    CODE_NAME = "ICD10_code"
else:
    CODE_FIELD = ICD9_FIELD
    CODE_NAME = "ICD9_code"

import pandas

events = pandas.read_csv("../data/patient_records/hesin.txt", sep="\t")
events['epistart'] = pandas.to_datetime(events.epistart, format="%d/%m/%Y")
events['disdate'] = pandas.to_datetime(events.disdate, format="%d/%m/%Y")
events['date'] = events.epistart
events.loc[events.date.isna(), 'date'] = events.loc[events.date.isna(), "disdate"] # Some have only disdate, so use that when epistart is not available
diags = pandas.read_csv("../data/patient_records/hesin_diag.txt", sep="\t")

data = pandas.merge(diags, events[["eid", "ins_index", "date"]], on=["eid", "ins_index"])

selected = data[~data[CODE_FIELD].isna()] #Only use the requested codes

# Take just the first date of diagnosis per each individaul
first_diags = selected.groupby(["eid", CODE_FIELD]).date.first().reset_index()
first_diags = first_diags.rename(columns={"eid": "ID", CODE_FIELD: CODE_NAME, "date": "first_date"})

first_diags.to_csv(args.output, sep="\t", index=False)
