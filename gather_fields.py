#!/usr/bin/env python
'''
Generates a smaller table of the variables we care about
'''
import argparse

import pandas

import fields_of_interest
from fields_of_interest import all_fields
import datafield_codings

parser = argparse.ArgumentParser(description="Generate a smaller table of the variables we care about")
parser.add_argument("-t", "--tsv", help="path to file of tab-separated data on the subjects", nargs="+")
parser.add_argument("-o", "--output", help="path to output to")

args = parser.parse_args()

data_tables = [pandas.read_csv(path, sep="\t", index_col=0)
                for path in args.tsv]
data = pandas.concat(data_tables, axis=1)
del data_tables


def process_fields(data, fields):
    ''' Operates on 'data' dataframe to apply the codings of the fields in it
    As well as to do the best we can to fill in missing values, etc.
    '''

    output = pandas.DataFrame([], index=data.index)

    # Go through every coding and examine the fields that it affects
    # Then process those by filling in NaN values, processing with the type of the field,
    # and filling in values for questions that weren't asked
    processed_fields = []
    for field_name, field in fields.items():

        coding = datafield_codings.fields_to_coding.get(field, None)

        if coding:
            print(f"Applying coding {coding_num} to field {field_name}:{field}")
            processed_fields.append(field_name)

            type = coding.get("type", None)
            if type == "array":
                cols = [c for c in data.columns if c.startswith("f.{field_name}.")]
                columns = {f"{field_name}_{i}":c for i,c in enumerate(cols)}
            elif type == "ordinal":
                columns = {field_name: f"f.{field}.0.0"}
                # TODO: do we want to process or mark ordinal columns in some manner?
            elif type == None:
                columns = {field_name: f"f.{field}.0.0"}
            else:
                # Might want categorical data codings later, processed as one-hot or similar
                raise NotImplementedError

            for column_name, column in columns.items():

                output[column_name] = data[column]

                # Set nan-like values to NaN
                for value in coding.get('to_nan', []):
                    output.loc[output[column_name] == value, column_name] = float("NaN")

                # Process field dependencies
                # I.e. we can sometimes fill in a reasonable value for a question
                #      that was never asked due to the value of another question
                #      If you say no to ever having a problem, we'll put you down as
                #      saying no for having each specific type of that problem, for example
                fillvalue = coding.get('fillvalue', None)
                if fillvalue is not None:
                    not_asked = pandas.Series(False, index=data.index)
                    if column_name in fields_of_interest.anxiety_dependent_fields:
                        # Anxiety dependent fields asked only if "ever worried much more" or longest period of worrying was at least 6 months
                        not_asked |= (data.ever_worried_much_more == 0) & ((data.ever_felt_worried_more_than_month == 0) | (data.longest_period_worried < 6) | (data.longest_period_worried != -999))

                    if (column_name in fields_of_interest.depression_dependent_fields
                            or column_name in fields_of_interest.sleep_change_type_fields):
                        # depression dependent fields asked only if "ever prolonged depression" or "ever prolonged loss of interest"
                        not_asked |= (data.ever_prolonged_depression == 0) & (data.ever_prolonged_loss_of_interest == 0)

                    if column_name in fields_of_interest.sleep_change_type_fields:
                        # Only asked how it changed if sleep change was indicated
                        not_asked |= (data.sleep_change_worst_episode == 0)

                    if column_name in fields_of_interest.mania_dependent_fields:
                        not_asked |= (data.ever_extreme_irritability == 0) & (data.ever_mania == 0)

                    if column_name == "longest_period_worried":
                        not_asked |= (data.ever_felt_worried_more_than_month == 0)

                    output.loc[not_asked, column_name] = fillvalue

                    print(f"Filling {field_name} with {fillvalue} in {not_asked.sum()} cases")
        else:
            print(f"WARNING: no coding provided for field {field_name}:{field}")
            output[field_name] = data[f"f.{field}.0.0"]

process_fields(small_data)
small_data.to_csv(args.output, sep="\t")
