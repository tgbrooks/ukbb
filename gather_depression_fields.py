#!/usr/bin/env python

import argparse

import pandas

import fields_of_interest
import datafield_codings

parser = argparse.ArgumentParser(description="Generate a smaller table of the variables we care about")
parser.add_argument("-t", "--tsv", help="path to file of tab-separated data on the subjects", nargs="+")
parser.add_argument("-o", "--output", help="path to output to")

args = parser.parse_args()

data_tables = [pandas.read_csv(path, sep="\t", index_col=0)
                for path in args.tsv]
data = pandas.concat(data_tables, axis=1)

all_fields = {key:f"f.{num}.0.0" for key, num in fields_of_interest.all_fields.items()}
invert_fields = {val:key for key, val in all_fields.items()}
data = data.rename(fields=invert_fields)

# Grab just those fields and output
small_data = data[list(fields.keys())]


def process_fields(data, codings):
    ''' Operates on 'data' dataframe to apply the codings of the fields in it
    As well as to do the best we can to fill in missing values, etc.
    '''

    # Go through every coding and examine the fields that it affects
    # Then process those by filling in NaN values, processing with the type of the field,
    # and filling in values for questions that weren't asked
    processed_fields = []
    for coding_num, coding in datafield_codings.codings.items():
        for field_name, field in all_fields:
            if field in coding['fields']:
                print(f"Applying coding {coding_num} to field {field_name}:{field_num}")
                processed_fields.append(field_name)

                type = coding.get("type", None)
                if type == "list":
                    print(f"Found a `list` type and its values are:")
                    print(set(data[field_name]))
                    # TODO: remove this and instead process the list (into multiple entries?)
                elif type == "ordinal":
                    # TODO: do we want to process or mark ordinal columns in some manner?
                    pass
                elif type == None:
                    pass
                else:
                    # Might want categorical data codings later, processed as one-hot or similar
                    raise NotImplementedError

                # Set nan-like values to NaN
                for value in coding.get('to_nan', []):
                    data.loc[data[field_name] == value, field_name] = float("NaN")

                # Process field dependencies
                fillvalue = coding.get('fillvalue', None)
                if fillvalue is not None:
                    if field in fields_of_interest.anxiety_dependent_fields:
                        # Anxiety dependent fields asked only if "ever worried much more" or longest period of worrying was at least 6 months
                        dependency = data[ever_worried_much_more] | (data[longest_period_worried] >  6)
                        data.loc[~dependency, field_name] = fillvalue

                    if field in fields_of_interest.depression_dependent_fields:
                        # depression dependent fields asked only if "ever prolonged depression" or "ever prolonged loss of interest"
                        dependency = data[ever_prolonged_depression] | data[ever_prolonged_loss_of_interest]
                        data.loc[~dependency, field_name] = fillvalue

                    if field in fields_of_interest.sleep_change_type_fields:
                        # Only asked how it changed if sleep change was indicated
                        dependency = data[sleep_change_worst_episode]
                        data.loc[~dependency, field_name] = fillvalue

                    if field in fields_of_interset.mania_dependent_fields:
                        dependency = data[ever_extreme_irritability] | data[ever_mania]
                        data.loc[~dependency, field_name] = fillvalue

    for field_name, field in all_fields.items():
        if field_name not in processed_fields:
            print(f"WARNING: no coding provided for field {field_name}:{field}")

small_data = process_fields(small_data)
small_data.to_csv(args.output, sep="\t")
