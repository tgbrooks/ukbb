#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description="Extract from the raw tsv UKBB tables the desired columns and apply appropriate codings")
parser.add_argument("-t", "--tables", help="table of UKBB data", nargs="+", required=True)
parser.add_argument("-o", "--output", help="file to output table to", required=True)
parser.add_argument("-f", "--filter", help="Field number to filter by: we only output participants who do not have a missing value in this field", default=None)
parser.add_argument("-s", "--set", help="Set of fields to extract, as defined in fields_of_interest.py", default="all_fields")

args = parser.parse_args()

import numpy
import pandas

import field_processing
def process_field(field, col):
    return field_processing.by_field.get(field, lambda x:x)(col)

NAN_VALUES = [
        "Prefer not to answer",
        "Do not know",
        "Do not remember",
        "Not applicable",
        "Prefer not to answer (group A)",
        "Prefer not to answer (group B)",
        "Measure not cleanly recoverable from data",
]

# These two files contain the information describing the fields in the UKBB
# they are downloaded from:
# http://biobank.ndph.ox.ac.uk/~bbdatan/Codings_Showcase.csv
# http://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv 
codings = pandas.read_csv("../Codings_Showcase.csv", index_col=0)
fields = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col=2)

data_tables = [pandas.read_csv(table, index_col=0, sep="\t") for table in args.tables]
data = pandas.concat(data_tables, axis=1)
del data_tables

if args.filter is not None:
    # Drop rows
    data.dropna(subset=[f"f.{args.filter}.0.0"], inplace=True)

import fields_of_interest
try:
    field_group =  fields_of_interest.__dict__[args.set]
except KeyError:
    print(f"Failed to find field set {args.set} in 'fields_of_interest.py'")
    raise SystemExit(1)

output = {}
for field_name, field in field_group.items():

    field_description = fields.loc[field]

    if numpy.isfinite(field_description.Coding):
        coding = codings.loc[[field_description.Coding]]
    else:
        coding = None

    entries = range(field_description.Array)
    instances = range(field_description.Instances)

    # TODO: handle instances
    if len(instances) > 1:
        print(f"WARNING: field {field_name}:{field} has {instances} instances but we are only extracting the first.")

    type = field_description.ValueType
    if type in ["Integer", "Date", "Text", "Continuous", "Time"]:
        dtype = {"Integer": float, "Date":str, "Text":str, "Continuous":float, "Time":str}[type]

        start = 0
        if f"f.{field}.0.0" not in data:
            # Sometimes the UKBB starts indexing at 1 instead of 0!!
            # So we detect that here
            start = 1

        for index in entries:

            try:
                col = data[f"f.{field}.0.{index + start}"].copy()
            except KeyError:
                print(f"WARNING: failed to find field {field_name}:{field}.{index} in data. Skipping.")
                continue

            if coding is not None:
                categories = coding.Value
                labels = coding.Meaning
                for category, label in zip(categories, labels):
                    if label in NAN_VALUES:
                        col[col == dtype(category)] = float("NaN")
                    else:
                        if index == 0:
                            print(f"WARNING: Unknown coding '{category}:{label}' in {field_name}:{field} of type {type}")

            col = process_field(field, col)

            if len(entries) > 1:
                output[f"{field_name}_V{index}"]  = col
            else:
                output[f"{field_name}"]  = col

    elif type == "Compound":
        raise NotImplementedError(f"field {field_name}:{field} has ValueType Compound which is not supported")
    elif type == "Categorical single":
        try:
            col = data[f"f.{field}.0.0"]
        except KeyError:
            print(f"WARNING: failed to find field {field_name}:{field} in data. Skipping.")
            continue
        col = process_field(field, col)
        categories = coding.Value.astype(int)
        labels = coding.Meaning
        column = pandas.Categorical(data[f"f.{field}.0.0"], categories)
        output[field_name] =  column.rename_categories({cat:label for cat, label in zip(categories, labels)})
    elif type == "Categorical multiple":
        # The UKBB stores "Category Multiple" variables as multiple columns and each participant filling out
        # the first several of those columns with the values of the category that they have
        # Eg you might have
        # 1 2 Nan
        # 2 3 4
        # 5 Nan Nan
        # to indicate that first person has 1,2, the second had 2,3,4, and third 5.
        # But this is inconvenient so we swap to making one column per possible response
        # and giving True or False values to each person for each possibble category
        # But this is bad for very large categories like the job selection
        categories = coding.Value
        if len(categories) > 30:
            print(f"WARNING: field {field_name}:{field} has more than 30 categories. Skipping")
            continue
        labels = coding.Meaning
        labels = [label.replace(" ", "_") for label in labels]
        columns = {f"{field_name}_{label}": pandas.Series(float("NaN"),index=data.index)
                    for label in labels}

        start = 0
        if f"f.{field}.0.0" not in data:
            # Sometimes the UKBB starts indexing at 1 instead of 0!!
            # So we detect that here
            start = 1

        for index in entries:
            try:
                col = data[f"f.{field}.0.{index+start}"]
            except KeyError:
                print(f"WARNING: failed to find field {field_name}:{field}.{index} in data. Skipping.")
                continue

            col = process_field(field, col)

            for value, label in zip(categories, labels):
                if index == 0:
                    valid = ~col.isna()
                    columns[f"{field_name}_{label}"][valid] = 0

                columns[f"{field_name}_{label}"][col == int(value)] = 1
        output.update(columns)
    else:
        raise NotImplementedError(f"field {field_name}:{field} has unown type {type}")

output = pandas.DataFrame(output)
if args.output.endswith("txt"):
    output.to_csv(args.output, sep="\t")
elif args.output.endswith("h5"):
    output.to_hdf(args.output, key="data", sep="\t", mode="w", format="table")
