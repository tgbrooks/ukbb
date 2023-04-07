'''
This script uses the real data to generate a similar, synthetic dataset for testing and demonstration purposes

This demonstration data can be shared and contains no real patient data, but mimics some aspects of the real data.
'''

#%
import pathlib
import pandas
import numpy as np

import phewas_preprocess


# Number of subjects to simulate
# The 'cases' will be given a diagnosis of diabetes
# and will be simulated to match those with diabetes in the real datasets
# while 'controls' will be given no diagnoses and will be simulated to match
# the overall dataset
N_CONTROLS = 1000
N_CASES = 1000
# Only one in SEASONAL_RATIO subjects have seasonal repeat data
SEASONAL_RATIO = 10

OUTDIR = pathlib.Path("synthetic_data/")
OUTDIR.mkdir(exist_ok=True)
PATIENT_DIR = OUTDIR / "patient_records"
PATIENT_DIR.mkdir(exist_ok=True)

#%
#### Load the data
# All data here is kept 'raw' without pre-processing
ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")

deaths = pandas.read_csv("../data/patient_records/death.txt", sep='\t', parse_dates=["date_of_death"], dayfirst=True).drop_duplicates(['eid', 'date_of_death']).set_index('eid')

full_activity = pandas.read_csv("../processed/activity_features_aggregate_seasonal.txt", sep="\t", dtype={'Unnamed: 0': str})
activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
activity_summary_seasonal = pandas.read_csv("../processed/activity_summary_aggregate_seasonal.txt", index_col=0, sep="\t")

icd10_entries_all = pandas.read_csv("../processed/ukbb_icd10_entries.txt", sep="\t")
icd9_entries_all = pandas.read_csv("../processed/ukbb_icd9_entries.txt", sep="\t")

variance = pandas.read_csv("../processed/inter_intra_personal_variance.seasonal_correction.txt", sep="\t")

self_report = pandas.read_csv("../processed/ukbb_self_reported_conditions.txt", sep="\t")

#%
main_data_columns = [
    'ethnicity',
    'overall_health',
    'BMI',
    'sex',
    'birth_year',
    'smoking',
    'alcohol_frequency',
    'townsend_deprivation_index',
    'education_College_or_University_degree',
    *list(phewas_preprocess.self_report_circadian_variables.keys()),
    'date_of_death',
    'blood_sample_time_collected_V0',
]
activity_columns = [
    'temp_amplitude',
    'temp_amplitude',
    'temp_RA',
    'temp_IV',
    'temp_cosinor_rsquared',
    'temp_within_day_SD',
    'temp_mean_mean',
    'phase',
    'temp_phase',
]

activity_summary_columns = [
    'file-startTime',
    'file-deviceID',
    'quality-goodCalibration',
    'quality-goodWearTime',
    'quality-daylightSavingsCrossover',
]

#%

# Seed is 0
rnd = np.random.default_rng(0)

# Generate the synthetic 'controls'
# By grabbing a random sample of each column, done independently so that there is no
# patient information retained
control_ids = np.random.choice(ukbb.index, size=N_CONTROLS, replace=False)
control_ukbb = pandas.DataFrame({
        col: ukbb[col].sample(n=N_CONTROLS).to_numpy()
        for col in main_data_columns
    },
    index = control_ids)

control_activity = pandas.DataFrame({
        col: full_activity[col].sample(n=N_CONTROLS).to_numpy()
        for col in activity_columns
    },
    index = [f'{id}.0' for id in control_ids])

control_activity_seasonal = pandas.DataFrame({
        col: full_activity[col].sample(n=4*N_CONTROLS//SEASONAL_RATIO).to_numpy()
        for col in activity_columns
    },
    index = [f'{id}.{run}' for id in control_ids[::SEASONAL_RATIO] for run in [1,2,3,4]])

control_act_summary = pandas.DataFrame({
        col: activity_summary[col].sample(n=N_CONTROLS).to_numpy()
        for col in activity_summary_columns
    },
    index=control_ids)

control_act_summary_seasonal = pandas.DataFrame({
        col: activity_summary_seasonal[col].sample(n=4*N_CONTROLS//SEASONAL_RATIO).to_numpy()
        for col in activity_summary_columns
    },
    index=[f'{id}.{run}' for id in control_ids[::SEASONAL_RATIO] for run in [1,2,3,4]])

control_deaths = ukbb.iloc[:,:0].join(deaths['date_of_death']).sample(N_CONTROLS).dropna()
control_deaths.index = np.random.choice(control_ids, size=len(control_deaths), replace=False)
act_start = control_deaths.join(control_act_summary['file-startTime'])
# Since the actual date of death may be after the actigraphy start, we generate random deaths
date_of_death = pandas.to_datetime(act_start['file-startTime']) + np.array([np.random.uniform(0,6) * pandas.to_timedelta("1d") * 365.25 for _ in range(control_deaths.shape[0])])
date_of_death = date_of_death.dt.date
control_deaths['date_of_death'] = control_deaths.index.map(date_of_death)
control_deaths.index.name = "eid"

control_self_reports = self_report[self_report.ID.isin(control_ids)][['ID', 'condition_code']]

#% ## Generate the 'cases'

# take the type 2 diabetes
is_diabetes = icd10_entries_all.ICD10_code.str.startswith("E11")
real_cases = icd10_entries_all[is_diabetes].ID.unique()

case_ids = np.random.choice(ukbb.index.difference(control_ids), size=N_CASES, replace=False)

case_ukbb = pandas.DataFrame({
        col: ukbb.loc[real_cases, col].sample(n=N_CASES).to_numpy()
        for col in main_data_columns
    },
    index = case_ids)

activity = full_activity.set_index(full_activity['Unnamed: 0'].astype(float).astype(int))
diabetes_activity = activity[activity.index.isin(real_cases)]
case_activity = pandas.DataFrame({
        col: diabetes_activity[col].sample(n=N_CASES).to_numpy()
        for col in activity_columns
    },
    index = [f'{id}.0' for id in case_ids])

case_act_summary = pandas.DataFrame({
        col: activity_summary[col].sample(n=N_CASES).to_numpy()
        for col in activity_summary_columns
    },
    index=case_ids)

# No case_act_summary_seasonl
# we'll just assume none of them have seasonal activity readings
# since we don't use these with case/control status

case_deaths = ukbb.loc[real_cases].iloc[:,:0].join(deaths['date_of_death']).sample(N_CASES).dropna()
case_deaths.index = np.random.choice(case_ids, size=len(case_deaths), replace=False)
act_start = case_deaths.join(case_act_summary['file-startTime'])
# Since the actual date of death may be after the actigraphy start, we generate random deaths
date_of_death = pandas.to_datetime(act_start['file-startTime']) + np.array([np.random.uniform(0,6) * pandas.to_timedelta("1d") * 365.25 for _ in range(case_deaths.shape[0])])
date_of_death = date_of_death.dt.date
case_deaths['date_of_death'] = case_deaths.index.map(date_of_death)
case_deaths.index.name = "eid"

diag_date = pandas.to_datetime(case_act_summary['file-startTime']) + np.array([np.random.uniform(0,6) * pandas.to_timedelta("1d") * 365.25 for _ in range(N_CASES)])
diag_date = diag_date.dt.date
case_icd10_codes = pandas.DataFrame({
    "ID": case_ids,
    "ICD10_code": "E119",
    "first_date": diag_date,
})

#### Join all the data together
joined_ukbb = pandas.concat([case_ukbb, control_ukbb])
joined_ukbb['actigraphy_file'] = '90004_0_0' # everyone has actigraphy in this synthetic data
joined_deaths = pandas.concat([case_deaths, control_deaths])
joined_activity = pandas.concat([case_activity, control_activity, control_activity_seasonal])
joined_act_summary = pandas.concat([case_act_summary, control_act_summary])
joined_act_summary_seasonal = control_act_summary_seasonal # no case_act_summary_seasonal made
joined_icd10_codes = case_icd10_codes # no diagnoses for controls
joined_icd9_codes = icd9_entries_all.iloc[:0] # Empty icd9 codes... we don't care
joined_self_reports = control_self_reports # No self reports for the cases since we don't want them excluded

##### Write out the files
joined_ukbb.to_hdf(OUTDIR / "ukbb_data_table.h5", key="data", mode="w", format="table")
joined_deaths.to_csv(PATIENT_DIR / "death.txt", sep="\t", date_format="%d/%m/%Y")
joined_activity.to_csv(OUTDIR / "activity_features_aggregate_seasonal.txt", sep="\t")
joined_act_summary.to_csv(OUTDIR / "activity_summary_aggregate.txt", sep="\t")
joined_act_summary_seasonal.to_csv(OUTDIR / "activity_summary_aggregate_seasonal.txt", sep="\t")
joined_icd10_codes.to_csv(OUTDIR / "ukbb_icd10_entries.txt", sep="\t", index=False)
joined_icd9_codes.to_csv(OUTDIR / "ukbb_icd9_entries.txt", sep="\t")
variance.to_csv(OUTDIR / "inter_intra_personal_variance.seasonal_correction.txt", sep="\t", index=False) # Just copy this file, no PI
joined_self_reports.to_csv(OUTDIR / "ukbb_self_reported_conditions.txt", sep="\t")