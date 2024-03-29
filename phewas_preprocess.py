import numpy
import pandas
import statsmodels.formula.api as smf


MAX_TEMP_AMPLITUDE = 10 # CUTOFF for values that are implausible or driven by extreme environments

YEAR = 365.25 * pandas.to_timedelta("1D")
# We will reject all individual measurements beyond this zscore cutoff value
ZSCORE_OUTLIER_CUTOFF = 7

# Force these activity variables to be included in the analysis even though they do meet the
# requirements for repeateability from the seasonal data
EXTRA_ACTIVITY_VARIABLES = ["temp_L1_time", "temp_phase", "weekend_temp_amplitude", "weekday_temp_amplitude"]

# Bins to group by age (at the time of actigraphy) in years
AGE_BINS  = [ 40, 55, 60, 65, 70, 80 ]
#AGE_BINS = [40, 65, 80]

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


def load_ukbb(input_dir):
    ''' Overall UK Biobank data table '''
    ukbb = pandas.read_hdf(input_dir / "ukbb_data_table.h5")
    ukbb.columns = ukbb.columns.str.replace("[,:/]","_", regex=True) # Can't use special characters easily

    # Update death information
    deaths = pandas.read_csv(input_dir / "patient_records/death.txt", sep='\t', parse_dates=["date_of_death"], dayfirst=True).drop_duplicates(['eid', 'date_of_death']).set_index('eid')
    ukbb['date_of_death'] = ukbb.index.map(deaths.date_of_death)

    return ukbb

def load_activity(ukbb, input_dir):
    ''' Subject-level activity data '''
    full_activity = pandas.read_csv(input_dir / "activity_features_aggregate_seasonal.txt", sep="\t", dtype={'Unnamed: 0': str})
    activity_summary = pandas.read_csv(input_dir / "activity_summary_aggregate.txt", index_col=0, sep="\t")
    activity_summary_seasonal = pandas.read_csv(input_dir / "activity_summary_aggregate_seasonal.txt", index_col=0, sep="\t")
    activity_summary_seasonal["ID"] = activity_summary_seasonal.index.astype(int)

    # Separate out the user ID from the run number (0 = original, 1-4 are seasonal repeats)
    full_activity.rename(columns={"Unnamed: 0": "run_id"}, inplace=True)
    full_activity['id'] = full_activity.run_id.apply(lambda x: int(x.split('.')[0]))
    full_activity['run'] = full_activity.run_id.apply(lambda x: int(x.split('.')[1]))

    # Compute the QC checks for everyone
    full_summary = pandas.concat([activity_summary, activity_summary_seasonal])
    calibrated = full_summary['quality-goodCalibration'].astype(bool)
    weartime = full_summary['quality-goodWearTime'].astype(bool)
    no_DST = ~full_summary['quality-daylightSavingsCrossover'].astype(bool)
    okay = calibrated & weartime & no_DST
    full_activity['qc_pass'] = full_activity.run_id.astype(float).map(okay)

    # Extract just the first runs
    activity = full_activity[full_activity.run == 0].copy()
    activity.set_index('id', inplace=True)

    # Some participants will not be present in the ukbb or have no actual data in the table
    # we exclude those now
    activity = activity[activity.index.map(~ukbb['sex'].isna())].copy()


    ## Select the activity variables that have between-person variance greater than their within-person variance
    # and for the summary variables, use only those that are overall summary variables
    activity_variance = pandas.read_csv(input_dir / "inter_intra_personal_variance.seasonal_correction.txt", sep="\t", index_col=0)
    activity_variance['summary_var'] = activity_variance.index.isin(activity_summary.columns)
    activity_variance['use'] = (~activity_variance.summary_var) | activity_variance.index.str.contains("overall-")
    good_variance = (activity_variance.corrected_intra_personal_normalized < 1)
    good_variance = good_variance | activity_variance.index.isin(EXTRA_ACTIVITY_VARIABLES) # Force inclusion of some variables despite variance
    activity_variables = activity_variance.index[good_variance & activity_variance.use]
    activity_variables = activity_variables.intersection(activity.columns)

    print(f"Started with {len(activity.columns.intersection(activity_variance[activity_variance.use].index))} activity variables")
    activity = activity[activity.columns[activity.columns.isin(activity_variables)]]
    print(f"Selected {len(activity.columns)} after discarding those with poor intra-personal variance")

    # drop and print activity for people who fail basic QC
    # Repeated from above for just the first measurements and so that
    # we can print out each step on its own for the purposes of the flowchart
    calibrated = activity_summary['quality-goodCalibration'].astype(bool)
    weartime = activity_summary['quality-goodWearTime'].astype(bool)
    no_DST = ~activity_summary['quality-daylightSavingsCrossover'].astype(bool)
    print(f"Dropping {(~calibrated).sum()} for bad calibration")
    print(f"Dropping {((~weartime) & calibrated).sum()} for bad weartime")
    print(f"Dropping {((~no_DST)  & weartime & calibrated).sum()} for daylight savings transition crossover")
    okay = (calibrated & weartime & no_DST)
    activity = activity[activity.index.map(okay)].copy()
    print(f"Dropped total {(~okay).sum()} entries out of {len(okay)} due to bad quality, wear-time, or DST crossover")

    activity.columns = activity.columns.str.replace("-","_", regex=False) # Can't use special characters easily

    activity_variance.index = activity_variance.index.str.replace("-","_", regex=False) # Can't use special characjers easily

    ## Process activity variables that need cleaning
    activity.phase = activity.phase % 24
    activity.temp_phase = activity.temp_phase % 24

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

    # List the activity variables
    activity_variables = activity.columns

    # Drop out the extreme outliers in the dataset
    # First remove all infinities as NaNs - shouldn't be many anyway
    infs = ~numpy.isfinite(activity.values)
    activity.iloc[infs] = float("NaN")
    stds = activity.std()
    means = activity.mean()
    zscores = ((activity - means) / stds).abs()
    outlier = zscores > ZSCORE_OUTLIER_CUTOFF
    num_outliers = outlier.sum().sum()
    pct_outliers = num_outliers / len(zscores.values.flatten())
    print(f"Identified and removed {num_outliers} ({pct_outliers:0.3%}) outlier measurements")
    activity[outlier] = float("Nan")

    return activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity

def correct_for_device_cluster(data, variable, multiplicative=True):
    d = data[variable]
    grand_median = d.median()
    category_medians = d.groupby(data.device_cluster).median()
    if multiplicative:
        data[variable] = d / data.device_cluster.map(category_medians).astype(float) * grand_median
    else:
        data[variable] = d - data.device_cluster.map(category_medians).astype(float) + grand_median

def load_medications(cohort_ids, input_dir):
    medications = pandas.read_csv(input_dir / "ukbb_medications.txt", sep="\t",
         dtype=dict(medication_code=int))
    data_fields = pandas.read_csv("metadata/Data_Dictionary_Showcase.csv", index_col="FieldID")
    codings = pandas.read_csv("metadata/Codings_Showcase.csv", dtype={"Coding": int})
    medication_code_to_meaning = codings[codings.Coding  == data_fields.loc[20003].Coding].drop_duplicates(subset=["Value"], keep=False)
    medication_code_to_meaning.Value = medication_code_to_meaning.Value.astype(int)
    medication_code_to_meaning.set_index("Value", inplace=True)
    medications['medication'] = medications.medication_code.map(medication_code_to_meaning.Meaning)

    medications = medications[medications.ID.isin(cohort_ids)]
    return medications

def load_data(cohort, input_dir):
    ukbb = load_ukbb(input_dir)
    activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity = load_activity(ukbb, input_dir)

    # Gather all the data
    data_full = activity.join(ukbb, how="inner")
    print(f"Data starting size: {data_full.shape}")

    # Actigraphy device metadata
    # Device id's cluster into three groups with large gaps between them and significant
    # differences in some measurements, particularly light and temperature
    # We give name them clusters A/B/C
    data_full['device_id'] = activity_summary['file-deviceID']
    data_full['device_cluster'] = pandas.cut( data_full.device_id, [0, 7_500, 12_500, 20_000]).cat.rename_categories(["A", "B", "C"])


    # Down sample for testing
    numpy.random.seed(0)
    cohort_id_ranges = {
        0: slice(0, None),  # Everyone
        1: slice(0, 25000), # RANDOM Exploratory cohort1
        2: slice(25000,None), # RANDOM Everyone not in cohort1
        10: data_full.device_cluster == "A",
        11: data_full.device_cluster.isin(['B', 'C']),
    }
    if cohort in [0, 1,2]:
        # Our randomized cohorts
        selected_ids = numpy.random.choice(data_full.index, size=data_full.shape[0], replace=False)[cohort_id_ranges[cohort]]
    else:
        # Cohorts that are selected based off of device-id
        selected_ids = data_full.index[cohort_id_ranges[cohort]]

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
    data['age_at_actigraphy'] = (data.actigraphy_start_date - data.birth_year_dt) / YEAR
    data['age_at_actigraphy_cat'] = pandas.cut(
        data.age_at_actigraphy,
        AGE_BINS
    ).astype("category").cat.rename_categories([f'{a}-{b}' for a,b in zip(AGE_BINS[:-1], AGE_BINS[1:])])

    # Use device clusters to correct some actigraphy values
    correct_for_device_cluster(data, 'temp_amplitude', multiplicative=True)
    correct_for_device_cluster(data, 'temp_RA', multiplicative=True)
    correct_for_device_cluster(data, 'temp_IV', multiplicative=True)
    correct_for_device_cluster(data, 'temp_cosinor_rsquared', multiplicative=True)
    correct_for_device_cluster(data, 'temp_within_day_SD', multiplicative=True)

    # Create simplified versions of the categorical covarites
    # so that there aren't too many covariate factors included in the model
    data['ethnicity_white'] = data.ethnicity.isin(["British", "Any other white background", "Irish", "White"])
    data['overall_health_good'] = data.overall_health.isin(["Good", "Excellent"])
    data.loc[data.overall_health.isin(['Do not know', 'Prefer not to answer']), 'overall_health_good'] = float("NaN")
    #data['smoking_ever'] = data.smoking.isin(['Previous', 'Current'])
    #data.loc[data.smoking == 'Prefer not to answer', 'smoking_ever'] = float("NaN")
    #data['high_income'] = data.household_income.isin(['52,000 to 100,000', 'Greater than 100,000'])
    #data.loc[data.high_income == 'Do not know', 'high_income'] = float("NaN")
    data['college_education'] = data['education_College_or_University_degree']
    #data['alcohol'] = data.alcohol_frequency.map({
    #    "Prefer not to answer": float("NaN"),
    #    "Never": "never",
    #    "Special occaisions only": "never",
    #    "One to three times a month": "sometimes",
    #    "Once or twice a week": "sometimes",
    #    "Three or four times a week": "often",
    #    "Daily or almost daily": "often",
    #})


    # Process death data, appropriate for Cox proportional hazards model
    data['date_of_death_censored'] = pandas.to_datetime(data.date_of_death)
    data.date_of_death_censored.fillna(data.date_of_death_censored.max(), inplace=True)
    data['date_of_death_censored_number'] = (data.date_of_death_censored - data.date_of_death_censored.min()).dt.total_seconds()
    data['uncensored'] = (~data.date_of_death.isna()).astype(int)
    data['age_at_death_censored'] = (pandas.to_datetime(data.date_of_death) - data.birth_year_dt) / YEAR
    data['entry_age'] = (data.actigraphy_start_date - data.birth_year_dt) / YEAR
    data.age_at_death_censored.fillna(data.age_at_death_censored.max(), inplace=True)

    # Bin BMI by the standard thresholds
    def BMI_type(bmi):
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "healthy weight"
        elif bmi < 30:
            return "overweight"
        elif bmi >= 30:
            return "obese"
        else:
            return float("NaN")
    data['BMI type'] = data.BMI.map(BMI_type).astype('category').cat.reorder_categories(['underweight', 'healthy weight', 'overweight', 'obese'])

    return data, ukbb, activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity
    
def correct_for_seasonality_and_cluster(data, full_activity, activity_summary, activity_summary_seasonal):
    full_summary = pandas.concat([activity_summary, activity_summary_seasonal])
    good = full_activity.temp_amplitude < MAX_TEMP_AMPLITUDE

    # Seasonal correction for full (i.e. with seasonal repeated measurements) activity data
    starts = pandas.to_datetime(full_summary['file-startTime'])
    full_activity['start_date'] = full_activity.run_id.astype(float).map(starts)
    year_start = pandas.to_datetime(full_activity['start_date'].dt.year.astype(str) + "-01-01")
    year_fraction = (full_activity['start_date'] - year_start) / (YEAR)
    full_activity['cos_year_fraction'] = numpy.cos(year_fraction*2*numpy.pi)
    full_activity['sin_year_fraction'] = numpy.sin(year_fraction*2*numpy.pi)

    # A discrete season variable
    def season(month):
         if month in [1, 2, 3]:
             return 'winter'
         elif month in [4,5,6]:
             return 'spring'
         elif month in [7,8,9]:
             return 'summer'
         elif month in [10,11,12]:
             return 'fall'
    full_activity['season'] = full_activity['start_date'].dt.month.map(season)

    # Compute the cosinor fit of log(temp_amplitude) over the duration of the year
    full_activity['log_temp_amplitude'] = numpy.log(full_activity.temp_amplitude)
    full_activity.loc[full_activity.log_temp_amplitude == float('-Inf'), 'log_temp_amplitude'] = float("NaN") # 2 subjects get log(0) but we just drop them
    fit = smf.ols(
        f"log_temp_amplitude ~ cos_year_fraction + sin_year_fraction",
        data = full_activity.loc[good, ['log_temp_amplitude', 'cos_year_fraction', 'sin_year_fraction', 'id']].dropna(),
    ).fit()
    full_activity['corrected_temp_amplitude'] = numpy.exp(
        full_activity.log_temp_amplitude
        - full_activity['cos_year_fraction'] * fit.params['cos_year_fraction']
        - full_activity['sin_year_fraction'] * fit.params['sin_year_fraction']
    )

    # Investigate device id groups
    device = full_summary.loc[full_activity.run_id.astype(float), 'file-deviceID']
    full_activity['device_id'] = device.values
    full_activity['device_cluster'] = pandas.cut( full_activity.device_id, [0, 7_500, 12_500, 20_000, 50_000]).cat.rename_categories(["A", "B", "C", "D"])

    # Correct for device cluster
    cluster_med = full_activity.groupby("device_cluster").corrected_temp_amplitude.median()
    overall_med = full_activity.corrected_temp_amplitude.median()
    full_activity['twice_corrected_temp_amplitude'] = full_activity.corrected_temp_amplitude / full_activity.device_cluster.map(cluster_med).astype(float) * overall_med

    # Zero out implausibly high values
    full_activity.loc[~good, 'twice_corrected_temp_amplitude'] = float("NaN")

    # Set these values in the data
    data['temp_amplitude'] = data.index.map(full_activity.query("run == 0.0").set_index("id").twice_corrected_temp_amplitude)