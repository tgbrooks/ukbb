import numpy
import pandas

## Add self-reported variables to activity document
# they need to be converted to 0,1 and NaNs
self_report_circadian_variables = {
    "daytime_dozing": {
        "name": "daytime_dozing",
        "zeros": ["Never/rarely"],
        "ones": ["All of the time", "Often"],
    },
    "getting_up_in_morning": {
        "name": "getting_up_in_morning",
        "zeros": ["Very easy"],
        "ones": ["Not at all easy"],
    },
    "morning_evening_person": {
        "name": "chronotype",
        "zeros": ["Definitely a 'morning' person"],
        "ones": ["Definitely an 'evening' person"],
    },
    "nap_during_day": {
        "name": "nap_during_day",
        "zeros": ["Never/rarely"],
        "ones": ["Usually"],
    },
    "sleep_duration": {
        "name": "sleep_duration",
        "zeros": None, # Sleep duration is quantative
        "ones": None,
    },
    "sleeplessness": {
        "name": "sleeplessness",
        "zeros": ["Never/rarely"],
        "ones": ["Usually"],
    },
    "snoring": {
        "name": "snoring",
        "zeros": ["No"],
        "ones": ["Yes"],
    },
    "IPAQ_activity_group": {
        "name": "IPAQ_activity_group",
        "zeros": ['low'],
        "ones": ['high'],
    },
}


def load_ukbb():
    ''' Overall UK Biobank data table '''
    ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")
    ukbb.columns = ukbb.columns.str.replace("[,:/]","_") # Can't use special characters easily

    return ukbb

def load_activity(ukbb):
    ''' Subject-level activity data '''
    full_activity = pandas.read_csv("../processed/activity_features_aggregate_seasonal.txt", sep="\t", dtype={'Unnamed: 0': str})
    activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
    activity_summary_seasonal = pandas.read_csv("../processed/activity_summary_aggregate_seasonal.txt", index_col=0, sep="\t")
    activity_summary_seasonal["ID"] = activity_summary_seasonal.index.astype(int)

    # Separate out the user ID from the run number (0 = original, 1-4 are seasonal repeats)
    full_activity.rename(columns={"Unnamed: 0": "run_id"}, inplace=True)
    full_activity['id'] = full_activity.run_id.apply(lambda x: int(x.split('.')[0]))
    full_activity['run'] = full_activity.run_id.apply(lambda x: int(x.split('.')[1]))
    activity = full_activity[full_activity.run == 0]
    activity.set_index('id', inplace=True)


    ## Select the activity variables that have between-person variance greater than their within-person variance
    # and for the summary variables, use only those that are overall summary variables
    activity_variance = pandas.read_csv("../processed/inter_intra_personal_variance.seasonal_correction.txt", sep="\t", index_col=0)
    activity_variance['summary_var'] = activity_variance.index.isin(activity_summary.columns)
    activity_variance['use'] = (~activity_variance.summary_var) | activity_variance.index.str.contains("overall-")
    good_variance = (activity_variance.corrected_intra_personal_normalized < 1)
    activity_variables = activity_variance.index[good_variance & activity_variance.use]
    activity_variables = activity_variables.intersection(activity.columns)

    print(f"Started with {len(activity.columns.intersection(activity_variance[activity_variance.use].index))} activity variables")
    activity = activity[activity.columns[activity.columns.isin(activity_variables)]]
    print(f"Selected {len(activity.columns)} after discarding those with poor intra-personal variance")

    # Load descriptions + categorization of activity variables
    activity_variable_descriptions = pandas.read_excel("../table_header.xlsx", index_col="Activity Variable", sheet_name="Variables")

    # drop activity for people who fail basic QC
    okay = (activity_summary['quality-goodCalibration'].astype(bool)
                & (~activity_summary['quality-daylightSavingsCrossover'].astype(bool))
                & (activity_summary['quality-goodWearTime'].astype(bool))
           )
    activity = activity[okay]
    activity.columns = activity.columns.str.replace("-","_") # Can't use special characters easily
    print(f"Dropping {(~okay).sum()} entries out of {len(okay)} due to bad quality or wear-time")

    activity_variance.index = activity_variance.index.str.replace("-","_") # Can't use special characters easily

    ## Process activity variables that need cleaning
    activity.phase = activity.phase % 24

    ## Correct activity measures based off of seasonality
    # compute the 'fraction of year' value (0 = January 1st, 1 = December 31st)
    actigraphy_start_date = activity.index.map(pandas.to_datetime(activity_summary['file-startTime']))
    year_start = pandas.to_datetime(actigraphy_start_date.year.astype(str) + "-01-01")
    year_fraction = (actigraphy_start_date - year_start) / (pandas.to_timedelta("1Y"))
    cos_year_fraction = numpy.cos(year_fraction*2*numpy.pi)
    sin_year_fraction = numpy.sin(year_fraction*2*numpy.pi)

    for var in activity_variables:
        if var.startswith("self_report"):
            continue
        activity[var] = activity[var] - cos_year_fraction * activity_variance.loc[var].cos - sin_year_fraction * activity_variance.loc[var].sin


    self_report_data = {}
    for variable, var_data in self_report_circadian_variables.items():
        if var_data['zeros'] is not None:
            # Binarize
            zeros = ukbb[variable].isin(var_data['zeros'])
            ones = ukbb[variable].isin(var_data['ones'])
            values = ones.astype(int)
            values[(~zeros) & (~ones)] = float("NaN")
        else:
            values = ukbb[variable]
        self_report_data["self_report_"+var_data['name']] = values
    self_report_data = pandas.DataFrame(self_report_data)
    activity = activity.join(self_report_data)

    # Create absolute deviation variables from phase variables
    # since both extreme low and high phase are expected to be correlated with outcomes
    # and we do the same for sleep duration measures
    for var in activity_variable_descriptions.index:
        if var.endswith('_abs_dev'):
            base_var = var[:-8]
            activity[var] = (activity[base_var] - activity[base_var].mean(axis=0)).abs()

    # List the activity variables
    activity_variables = activity.columns
    return activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance

def load_phecode(selected_ids):
    # Load the PheCode mappings
    # Downloaded from https://phewascatalog.org/phecodes_icd10
    # Has columns:
    # ICD10 | PHECODE | Exl. Phecodes | Excl. Phenotypes
    phecode_info = pandas.read_csv("../phecode_definitions1.2.csv", index_col=0)
    phecode_map = pandas.read_csv("../Phecode_map_v1_2_icd10_beta.csv")
    phecode_map.set_index(phecode_map.ICD10.str.replace(".",""), inplace=True) # Remove '.' to match UKBB-style ICD10 codes

    # and convert to phecodes
    # v1.2 Downloaded from https://phewascatalog.org/phecodes
    phecode_map_icd9 = pandas.read_csv("../phecode_icd9_map_unrolled.csv")
    phecode_map_icd9.rename(columns={"icd9":"ICD9", "phecode":"PHECODE"}, inplace=True)
    phecode_map_icd9.set_index( phecode_map_icd9['ICD9'].str.replace(".",""), inplace=True) # Remove dots to match UKBB-style ICD9s

    # ## Load the ICD10/9 code data
    icd10_entries_all = pandas.read_csv("../processed/ukbb_icd10_entries.txt", sep="\t")
    # Select our cohort from all the entries
    icd10_entries = icd10_entries_all[icd10_entries_all.ID.isin(selected_ids)].copy()
    icd10_entries.rename(columns={"ICD10_code": "ICD10"}, inplace=True)
    icd10_entries = icd10_entries.join(phecode_map.PHECODE, on="ICD10")

    ### and the ICD9 data
    icd9_entries_all = pandas.read_csv("../processed/ukbb_icd9_entries.txt", sep="\t")
    # Select our cohort from all the entries
    icd9_entries = icd9_entries_all[icd9_entries_all.ID.isin(selected_ids)].copy()
    icd9_entries.rename(columns={"ICD9_code": "ICD9"}, inplace=True)
    icd9_entries = icd9_entries.join(phecode_map_icd9.PHECODE, on="ICD9")

    # Self-reported conditions from the interview stage of the UK Biobank
    self_reported_all = pandas.read_csv("../processed/ukbb_self_reported_conditions.txt", sep="\t", dtype={"condition_code":int})
    self_reported = self_reported_all[self_reported_all.ID.isin(selected_ids)].copy()
    data_fields = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col="FieldID")
    codings = pandas.read_csv("../Codings_Showcase.csv", dtype={"Coding": int})
    condition_code_to_meaning = codings[codings.Coding  == data_fields.loc[20002].Coding].drop_duplicates(subset=["Value"], keep=False).set_index("Value")
    self_reported["condition"] = self_reported.condition_code.astype(str).map(condition_code_to_meaning.Meaning)

    # Convert self-reported conditions to phecodes

    # Load Manaully mapped self-reports to phecodes
    self_report_phecode_map = pandas.read_csv("../self_report_conditions_meanings.txt", sep="\t", index_col=0)
    self_reported["phecode"] = self_reported.condition_code.map(self_report_phecode_map['PheCODE'])


    # Gather whether each person has a diagnosis from a given PheCode group

    # Group phecodes together that differ only after the '.'
    # i.e. if they convert to the same integer
    phecode_groups = phecode_info.index.astype(int).unique()

    phecode_data_icd10 = {}
    phecode_data_icd9 = {}
    phecode_data_self_report = {}
    for group in phecode_groups:
        group_data = phecode_info[phecode_info.index.astype(int) == group]
        icd10_codes = phecode_map[phecode_map.PHECODE.isin(group_data.index)].index
        icd9_codes = phecode_map_icd9[phecode_map_icd9.PHECODE.isin(group_data.index)].index
        in_block = icd10_entries.ICD10.isin(icd10_codes)
        in_block_icd9 = icd9_entries.ICD9.isin(icd9_codes)
        
        diagnosed = in_block.groupby(icd10_entries.ID).any()
        phecode_data_icd10[group] = diagnosed
        phecode_data_icd9[group] =  in_block_icd9.groupby(icd9_entries.ID).any()
        phecode_data_self_report[group] = self_reported.phecode.isin(group_data.index)

    phecode_data_icd10 = pandas.DataFrame(phecode_data_icd10)
    phecode_data_icd9 = pandas.DataFrame(phecode_data_icd9)
    phecode_data_self_report = pandas.DataFrame(phecode_data_self_report).set_index(self_reported.ID)
    phecode_data = pandas.concat([phecode_data_icd10, phecode_data_icd9, phecode_data_self_report]).reset_index().groupby(by="ID").any()

    # ### Display which sources the cases come from for the top codes

    phecode_counts = pandas.DataFrame({"counts": phecode_data.sum(axis=0)})
    for name, d in {"icd10": phecode_data_icd10, "icd9": phecode_data_icd9, "self_report": phecode_data_self_report}.items():
        cases = d.reset_index().groupby(by="ID").any()
        phecode_counts[name + "_cases"] = cases.sum(axis=0)
    phecode_counts["phecode_meaning"] = phecode_counts.index.map(phecode_info.phenotype)
    print("Most frequent phecodes by source")
    print(phecode_counts.sort_values(by="counts", ascending=False).head(20))


    return phecode_data, phecode_groups, phecode_info, phecode_map, icd10_entries, icd10_entries_all

def phecode_count_summary():
    # TODO: do we want to use this anymore?
    ## Tally the number of occurances of each phecode (at the lowest level, not just groups)
    phecode_count_details = pandas.concat([
        icd10_entries[['ID', 'PHECODE']],
        icd9_entries[['ID', 'PHECODE']],
        self_reported[['ID', 'phecode']].rename(columns={"phecode":"PHECODE"})
    ]).groupby('PHECODE').ID.nunique()
    phecode_count_details = pandas.DataFrame({"count": phecode_count_details})
    phecode_count_details['phecode_meaning'] = phecode_count_details.index.map(phecode_info.phenotype)
    phecode_count_details['phecode_category'] = phecode_count_details.index.map(phecode_info.category)
    #phecode_count_details.to_csv("phecode_counts.txt", sep="\t", header=True)
    return phecode_count_details

def load_data(cohort):
    ukbb = load_ukbb()
    activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance = load_activity(ukbb)

    # Gather all the data
    data_full = activity.join(ukbb, how="inner")
    print(f"Data starting size: {data_full.shape}")


    # Down sample for testing
    numpy.random.seed(0)
    # Note: total 92331, half is 46164
    cohort_id_ranges = {1: slice(0, 25000),
               2: slice(25000,50000)}
    selected_ids = numpy.random.choice(data_full.index, size=data_full.shape[0], replace=False)[cohort_id_ranges[cohort]]

    data = data_full.loc[selected_ids].copy()
    print(f"Data size after selecting test set: {data.shape}")

    # Finish processing
    # Age/birth year processing
    data['birth_year_category'] = pandas.cut(data.birth_year, bins=[1930, 1940, 1950, 1960, 1970])
    data['actigraphy_start_date'] = data.index.map(pandas.to_datetime(activity_summary['file-startTime']))
    def year_to_jan_first(year):
        if year != year:
            return float("NaN")
        else:
            return str(int(year)) + "-01-01"
    data['birth_year_dt'] = pandas.to_datetime(data.birth_year.apply(year_to_jan_first)) # As datetime
    data['age_at_actigraphy'] = (data.actigraphy_start_date - data.birth_year_dt) / pandas.to_timedelta("1Y")

    # Create simplified versions of the categorical covarites
    # This is necessary for convergence of the logistic models
    data['ethnicity_white'] = data.ethnicity.isin(["British", "any other white background", "Irish", "White"])
    data['overall_health_good'] = data.overall_health.isin(["Good", "Excellent"])
    data.loc[data.overall_health.isin(['Do not know', 'Prefer not to answer']), 'overall_health_good'] = float("NaN")
    data['smoking_ever'] = data.smoking.isin(['Previous', 'Current'])
    data.loc[data.smoking == 'Prefer not to answer', 'smoking_ever'] = float("NaN")
    data['high_income'] = data.household_income.isin(['52,000 to 100,000', 'Greater than 100,000'])
    data.loc[data.high_income == 'Do not know', 'high_income'] = float("NaN")
    data['college_education'] = data['education_College_or_University_degree']

    # Process daeath data, appropriate for Cox proportional hazards model
    data['date_of_death_censored'] = pandas.to_datetime(data.date_of_death)
    data.date_of_death_censored.fillna(data.date_of_death_censored.max(), inplace=True)
    data['date_of_death_censored_number'] = (data.date_of_death_censored - data.date_of_death_censored.min()).dt.total_seconds()
    data['uncensored'] = (~data.date_of_death.isna()).astype(int)
    data['age_at_death_censored'] = (pandas.to_datetime(data.date_of_death) - data.birth_year_dt) / pandas.to_timedelta("1Y")
    data['entry_age'] = (data.actigraphy_start_date - data.birth_year_dt) / pandas.to_timedelta("1Y")
    data.age_at_death_censored.fillna(data.age_at_death_censored.max(), inplace=True)


    # Load phecode data
    phecode_data, phecode_groups, phecode_info, phecode_map, icd10_entries, icd10_entries_all = load_phecode(selected_ids)

    # Gather phecode diagnosis information for each subject
    for group in phecode_groups:
        # Note that those without any ICD10 entries at all should be marked as non-case, hence the fillna()
        data[group] = data.index.map(phecode_data[group].astype(int)).fillna(0)
    
    return data, ukbb, activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, phecode_data, phecode_groups, phecode_info, phecode_map, icd10_entries, icd10_entries_all

if __name__ == "__main__":
    data, ukbb, activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, phecode_data, phecode_gorups, phecode_info, phecode_map, icd10_entries, icd10_entries_all = load_data(1)