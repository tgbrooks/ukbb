#!/usr/bin/env python

import argparse

import pandas

import fields_of_interest
from fields_of_interest import all_fields
import datafield_codings

parser = argparse.ArgumentParser(description="Generate a smaller table of the variables we care about")
parser.add_argument("-t", "--tsv", help="path to file of tab-separated data on the subjects", nargs="+")
parser.add_argument("-o", "--output", help="path to output to")

args = parser.parse_args()

processed_all_fields = {key:f"f.{num}.0.0" for key, num in all_fields.items()}
invert_fields = {val:key for key, val in processed_all_fields.items()}

data_tables = [pandas.read_csv(path, sep="\t", index_col=0)
                for path in args.tsv]
data = pandas.concat(data_tables, axis=1)
del data_tables

small_data = data[list(processed_all_fields.values())].copy()
del data

small_data.rename(columns=invert_fields, inplace=True)

def process_fields(data):
    ''' Operates on 'data' dataframe to apply the codings of the fields in it
    As well as to do the best we can to fill in missing values, etc.
    '''

    # Those with a questionnaire filled out
    # we don't want to use fillvalues on people who haven't answered any questionnaires
    has_questionnaire = ~data.date_of_mental_health_questionnaire.isna()

    # Go through every coding and examine the fields that it affects
    # Then process those by filling in NaN values, processing with the type of the field,
    # and filling in values for questions that weren't asked
    processed_fields = []
    for coding_num, coding in datafield_codings.codings.items():
        for field_name, field in all_fields.items():
            if field in coding['fields']:
                print(f"Applying coding {coding_num} to field {field_name}:{field}")
                processed_fields.append(field_name)

                type = coding.get("type", None)
                if type == "list":
                    raise NotImplementedError
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
                # I.e. we can sometimes fill in a reasonable value for a question
                #      that was never asked due to the value of another question
                #      If you say no to ever having a problem, we'll put you down as
                #      saying no for having each specific type of that problem, for example
                fillvalue = coding.get('fillvalue', None)
                if fillvalue is not None:
                    not_asked = pandas.Series(False, index=data.index)
                    if field_name in fields_of_interest.anxiety_dependent_fields:
                        # Anxiety dependent fields asked only if "ever worried much more" or longest period of worrying was at least 6 months
                        not_asked |= (data.ever_worried_much_more == 0) & ((data.ever_felt_worried_more_than_month == 0) | (data.longest_period_worried < 6) | (data.longest_period_worried != -999))

                    if (field_name in fields_of_interest.depression_dependent_fields
                            or field_name in fields_of_interest.sleep_change_type_fields):
                        # depression dependent fields asked only if "ever prolonged depression" or "ever prolonged loss of interest"
                        not_asked |= (data.ever_prolonged_depression == 0) & (data.ever_prolonged_loss_of_interest == 0)

                    if field_name in fields_of_interest.sleep_change_type_fields:
                        # Only asked how it changed if sleep change was indicated
                        not_asked |= (data.sleep_change_worst_episode == 0)

                    if field_name in fields_of_interest.mania_dependent_fields:
                        not_asked |= (data.ever_extreme_irritability == 0) & (data.ever_mania == 0)

                    if field_name == "longest_period_worried":
                        not_asked |= (data.ever_felt_worried_more_than_month == 0)

                    data.loc[not_asked, field_name] = fillvalue
                    print(f"Filling {field_name} with {fillvalue} in {not_asked.sum()} cases")

    for field_name, field in all_fields.items():
        if field_name not in processed_fields:
            print(f"WARNING: no coding provided for field {field_name}:{field}")

process_fields(small_data)
small_data.to_csv(args.output, sep="\t")
