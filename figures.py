import re

import scipy
import scipy.stats
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.cluster import hierarchy
import pylab
import pandas

COHORT = 1
OUTDIR = f"../global_phewas/cohort{COHORT}/"

### Whether to run all the big computations or attempt to load from disk since already computed
RECOMPUTE = False

full_activity = pandas.read_csv("../processed/activity_features_aggregate_seasonal.txt", sep="\t", dtype={'Unnamed: 0': str})
activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")

ukbb.columns = ukbb.columns.str.replace("[,:/]","_") # Can't use special characters easily

# Separate out the user ID from the run number (0 = original, 1-4 are seasonal repeats)
full_activity.rename(columns={"Unnamed: 0": "run_id"}, inplace=True)
full_activity['id'] = full_activity.run_id.apply(lambda x: int(x.split('.')[0]))
full_activity['run'] = full_activity.run_id.apply(lambda x: int(x.split('.')[1]))
activity = full_activity[full_activity.run == 0]
activity.set_index('id', inplace=True)
activity = activity.join(activity_summary)


## Select the activity variables that have between-person variance greater than their within-person variance
# and for the summary variables, use only those that are overall summary variables
activity_variance = pandas.read_csv("../processed/inter_intra_personal_variance.txt", sep="\t", index_col=0)
activity_variance['summary_var'] = activity_variance.index.isin(activity_summary.columns)
activity_variance['use'] = (~activity_variance.summary_var) | activity_variance.index.str.contains("overall-")
good_variance = (activity_variance.normalized < 1)
activity_variables = activity_variance.index[good_variance & activity_variance.use]
activity_variables = activity_variables.intersection(activity.columns)

print(f"Started with {len(activity.columns.intersection(activity_variance[activity_variance.use].index))} activity variables")
activity = activity[activity.columns[activity.columns.isin(activity_variables)]]
print(f"Selected {len(activity.columns)} after discarding those with poor intra-personal variance")

# Load descriptions + categorization of activity variables
activity_variable_descriptions = pandas.read_excel("../variable_descriptions.xlsx", index_col="Activity Variable")

# drop activity for people who fail basic QC
okay = (activity_summary['quality-goodCalibration'].astype(bool)
            & (~activity_summary['quality-daylightSavingsCrossover'].astype(bool))
            & (activity_summary['quality-goodWearTime'].astype(bool))
       )
activity = activity[okay]
activity.columns = activity.columns.str.replace("-","_") # Can't use special characters easily
activity_variables = activity_variables.str.replace("-","_")
print(f"Dropping {(~okay).sum()} entries out of {len(okay)} due to bad quality or wear-time")

## Process activity variables that need cleaning
activity.phase = activity.phase % 24

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
    "monring_evening_persion": {
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
    }
}

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
activity_variables = activity_variables.append(self_report_data.columns)

# Gather all the data
data_full = activity.join(ukbb, how="inner")
print(f"Data starting size: {data_full.shape}")


# List of covariates we will controll for in the linear model
covariates = [
              "sex", "ethnicity", "overall_health", "household_income", "smoking", "birth_year", "BMI",
               #'education_Prefer_not_to_answer', # This answer causes problems for some reason
               'education_None_of_the_above',
               'education_College_or_University_degree',
               'education_A_levels_AS_levels_or_equivalent', 
               'education_O_levels_GCSEs_or_equivalent',
               'education_CSEs_or_equivalent',
               'education_NVQ_or_HND_or_HNC_or_equivalent',
               'education_Other_professional_qualifications_eg__nursing__teaching',
                ]

# Down sample for testing
numpy.random.seed(0)
# Note: total 92331, half is 46164
cohort_id_ranges = {1: slice(0, 25000),
           2: slice(25000,50000)}
selected_ids = numpy.random.choice(data_full.index, size=data_full.shape[0], replace=False)[cohort_id_ranges[COHORT]]
data = data_full.loc[selected_ids].copy()
print(f"Data size after selecting test set: {data.shape}")

# Q-value utility
def BH_FDR(ps):
    ''' Benjamini-Hochberg FDR control

    Converts p values to q values'''

    # For the purposes of comparison, an implementation of Benjamini Hochberg correction
    sort_order = numpy.argsort(ps)

    adjusted = numpy.zeros(ps.shape)
    adjusted[sort_order] = numpy.array(ps)[sort_order]*len(ps)/numpy.arange(1,len(ps)+1)

    # Make monotone, skipping NaNs
    m = 1
    for i, r in enumerate(sort_order[::-1]):
        if numpy.isfinite(adjusted[r]):
            m = min(adjusted[r], m)
            adjusted[r] = m

    return adjusted # the q-values

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
SELF_REPORTED_CONDITION_FIELD = 20002
condition_code_to_meaning = codings[codings.Coding  == data_fields.loc[20002].Coding].drop_duplicates(subset=["Value"], keep=False).set_index("Value")
self_reported["condition"] = self_reported.condition_code.astype(str).map(condition_code_to_meaning.Meaning)

# # Run a PheCode-based analysis

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


## Tally the number of occurances of each phecode (at the lowest level, not just groups)
phecode_count_details = pandas.concat([
    icd10_entries[['ID', 'PHECODE']],
    icd9_entries[['ID', 'PHECODE']],
    self_reported[['ID', 'phecode']].rename(columns={"phecode":"PHECODE"})
]).groupby('PHECODE').ID.nunique()
phecode_count_details = pandas.DataFrame({"count": phecode_count_details})
phecode_count_details['meaning'] = phecode_count_details.index.map(phecode_info.phenotype)
phecode_count_details['category'] = phecode_count_details.index.map(phecode_info.category)
phecode_count_details.to_csv(OUTDIR+"phecode_counts.txt", sep="\t", header=True)



# Fancy style plot
# Only really works for highly abundant phenotypes like hypertension (401)
def fancy_case_control_plot(data, code, var="acceleration_RA", normalize=False, confidence_interval=False):
    CONTROL_COLOR = "teal"
    CASE_COLOR = "orange"
    UNCERTAIN_COLOR = (0.8, 0.8, 0.8)

    case = data[code] == True

    point_width = 0.01
    xbottom = 0.6
    xtop = 1.0
    eval_x = numpy.linspace(xbottom, xtop, 0.5/point_width + 1)

    case_scaling = (case).sum() * point_width
    control_scaling = (~case).sum() * point_width
    case_avg = data[var][case].median()
    control_avg = data[var][~case].median()
    total_incidence = case.sum()/len(case)
    
    def densities_and_incidence(data):
        case_density = scipy.stats.gaussian_kde(data[var][case], bw_method=0.15)(eval_x) * case_scaling
        control_density = scipy.stats.gaussian_kde(data[var][~case], bw_method=0.15)(eval_x) * control_scaling
        if not normalize:
            #incidence = local_regression(data[var], data[code], eval_x, bw=0.1)
            incidence = case_density / (control_density  + case_density)
        if normalize:
            incidence = case_density / total_incidence / 2 / (control_density + case_density / total_incidence / 2)
        return case_density, control_density, incidence
    
    case_density, control_density, incidence = densities_and_incidence(data)
            
    if confidence_interval:
        N = 40
        incidences = numpy.empty((len(eval_x), N))
        for i in range(N):
            sample = data.sample(len(data), replace=True)
            _, _, incidence = densities_and_incidence(sample)
            incidences[:,i] = incidence
        incidences = numpy.sort(incidences, axis=1)
        lower_bound = incidences[:,0]
        upper_bound = incidences[:,-1]
        middle = incidences[:,incidences.shape[1]//2]

    fig, (ax1,ax3,ax2) = pylab.subplots(nrows=3, sharex=True,
                                        gridspec_kw = {"hspace":0.1,
                                                       "height_ratios":[0.2,0.2,0.6]})

    # Plot the data
    ax1.fill_between(eval_x, 0, control_density, color=CONTROL_COLOR)
    if confidence_interval:
        ax2.fill_between(eval_x, lower_bound, middle, color='k', alpha=0.5)
        ax2.fill_between(eval_x, middle, upper_bound, color='k', alpha=0.5)
    ax3.fill_between(eval_x, 0, case_density, color=CASE_COLOR)

    # Plot avgs
    ax1.axvline(control_avg, c='k', linestyle="--")
    ax3.axvline(case_avg, c='k', linestyle="--")
    ax2.axhline(total_incidence, c='k', linestyle="--")

    # Label plot
    ax1.set_ylabel(f"controls\nN={(~case).sum()}")
    ax2.set_ylabel(f"ratio\n(overall={total_incidence:0.1%})")
    ax3.set_ylabel(f"cases\nN={case.sum()}") 
    ax2.set_xlabel("RA")

    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax3.tick_params(bottom=False)
    ax1.yaxis.set_ticks([])
    #ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax2.yaxis.set_ticklabels(["0%", "25%", "50%", "75%","100%"])
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.set_ticks([])

    # Set axis limits
    ax1.set_xlim(xbottom, xtop)
    ax2.set_ylim(0, 1)
    if not normalize:
        max_density = max(numpy.max(case_density), numpy.max(control_density))
        ax1.set_ylim(0, max_density)
        ax3.set_ylim(0, max_density)
    else:
        ax1.set_ylim(0)
        ax3.set_ylim(0)
    ax3.invert_yaxis()

    try:
        ax1.set_title(phecode_info.loc[code].phenotype + ("\n(normalized)" if normalize else ""))
    except KeyError:
        ax1.set_title(code)
    return fig

fig = fancy_case_control_plot(data, 401, normalize=False, confidence_interval=True)
