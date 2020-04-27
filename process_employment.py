#!/usr/bin/env python
import argparse

description = '''
Gather employment history data for each person

The result is a table of entries, one for each occurance of an employment history entry
Columns:
ID    year_job_started year_job_ended job_code night_shifts_worked consecutive_night_shifts_during_mixed_shifts consecutive_night_shifts_during_night_sihfts day_shifts_worked job_involved_shift_work mixture_of_day_and_night_shifts night_shifts_worked number_night_shifts_monthly_during_mixed_shifts number_night_shifts_monthly_during_night_shifts period_spent_working_day_shifts period_spent_working_mixed_shifts period_spent_working_night_shifts rest_days_during_mixed_shift_periods rest_days_during_night_shift_periods length_of_night_shift_during_mixed_shifts length_of_night_shift_during_night_shifts

So a participant may have multiple entries, one for each job worked
Note: many participants have no information about their job history
'''

parser = argparse.ArgumentParser(description)
parser.add_argument("-t", "--table", help="UKBB table file to read in", required=True)
parser.add_argument("-o", "--output", help="output tab-separated file to write to", required=True)

args = parser.parse_args()

import pandas

from fields_of_interest import employment_fields

# This file contain the information describing the fields in the UKBB
# http://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv 
field_data = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col=2)

num_entries = field_data.loc[employment_fields['year_job_started']].Array

processed_data = []


for data in pandas.read_csv(args.table, sep="\t", index_col=0, chunksize=10_000, low_memory=False):
    data.index.rename("ID", inplace=True)
    for i in range(num_entries):
        fields = {name:f"f.{fnum}.0.{i}" for name,fnum in employment_fields.items()}
        # Note: some fields are not present in later entries!
        # this is because the column is removed if all their entries for all participants are blank
        fields = {name:field for name,field in fields.items()
                    if field in data.columns}

        fields_reversed = {field:name for name,field in fields.items()}

        entry = data[fields.values()].rename(columns=fields_reversed)

        valid = ~(entry.year_job_started.isna())
        entry = entry[valid]
        processed_data.append(entry)

all_data = pandas.concat(processed_data, sort=True)
all_data.sort_index(inplace=True, kind="mergesort")
all_data.to_csv(args.output, sep="\t")
