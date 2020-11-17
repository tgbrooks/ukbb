# # PheWAS analysis
# 
# Check if there are associations of phenotypes with circadian problems,
# particularly for unusual timing or for lack of consistent rhythm.

import re

import scipy
import scipy.stats
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
import matplotlib.patches as mpatches
from scipy.cluster import hierarchy
import pylab
import pandas
import networkx

pandas.plotting.register_matplotlib_converters()

COHORT = 1
OUTDIR = f"../global_phewas/cohort{COHORT}/"

### Whether to run all the big computations or attempt to load from disk since already computed
RECOMPUTE = False

# Point at which to call FDR q values significant
FDR_CUTOFF_VALUE = 0.05

# OLS method is either 'svd' or 'qr'
# Using 'qr' since intermittent problems wtih SVD convergence
OLS_METHOD = 'qr'
def OLS(*args, **kwargs):
    for i in range(100):
        try:
            return smf.ols(*args, **kwargs).fit()#method=OLS_METHOD)
        except numpy.linalg.LinAlgError:
            print("Failed regression:")
            print(args)
            print(kwargs)
            print("Attempt number {i}")
            continue
    raise numpy.linalg.LinAlgError

full_activity = pandas.read_csv("../processed/activity_features_aggregate_seasonal.txt", sep="\t", dtype={'Unnamed: 0': str})
activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
activity_summary_seasonal = pandas.read_csv("../processed/activity_summary_aggregate_seasonal.txt", index_col=0, sep="\t")
activity_summary_seasonal["ID"] = activity_summary_seasonal.index.astype(int)
ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")

ukbb.columns = ukbb.columns.str.replace("[,:/]","_") # Can't use special characters easily

# Separate out the user ID from the run number (0 = original, 1-4 are seasonal repeats)
full_activity.rename(columns={"Unnamed: 0": "run_id"}, inplace=True)
full_activity['id'] = full_activity.run_id.apply(lambda x: int(x.split('.')[0]))
full_activity['run'] = full_activity.run_id.apply(lambda x: int(x.split('.')[1]))
activity = full_activity[full_activity.run == 0]
activity.set_index('id', inplace=True)


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
activity_variable_descriptions = pandas.read_excel("../table_header.xlsx", index_col="Activity Variable", sheet_name="Variables")

# drop activity for people who fail basic QC
okay = (activity_summary['quality-goodCalibration'].astype(bool)
            & (~activity_summary['quality-daylightSavingsCrossover'].astype(bool))
            & (activity_summary['quality-goodWearTime'].astype(bool))
       )
activity = activity[okay]
activity.columns = activity.columns.str.replace("-","_") # Can't use special characters easily
activity_variables = activity_variables.str.replace("-","_")
print(f"Dropping {(~okay).sum()} entries out of {len(okay)} due to bad quality or wear-time")

activity_variance.index = activity_variance.index.str.replace("-","_") # Can't use special characters easily

## Process activity variables that need cleaning
activity.phase = activity.phase % 24

# Create absolute deviation variables from phase variables
# since both extreme low and high phase are expected to be correlated with outcomes
phase_vars = activity_variable_descriptions.index[activity_variable_descriptions.Subcategory == "Phase"]
for var in phase_vars:
    if '_abs_dev' in var:
        base_var = var[:-8]
        activity[var] = (activity[base_var] - activity[base_var].mean(axis=0)).abs()

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

# Age/birth year processing
data['birth_year_category'] = pandas.cut(data.birth_year, bins=[1930, 1940, 1950, 1960, 1970])
data['actigraphy_start_date'] = data.index.map(pandas.to_datetime(activity_summary['file-startTime']))
birth_year = pandas.to_datetime(data.birth_year.astype(int).astype(str) + "-01-01") # As datetime
data['age_at_actigraphy'] = (data.actigraphy_start_date - birth_year) / pandas.to_timedelta("1Y")

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
phecode_count_details['phecode_meaning'] = phecode_count_details.index.map(phecode_info.phenotype)
phecode_count_details['phecode_category'] = phecode_count_details.index.map(phecode_info.category)
phecode_count_details.to_csv(OUTDIR+"phecode_counts.txt", sep="\t", header=True)


# ### Display which sources the cases come from for the top codes


phecode_counts = pandas.DataFrame({"counts": phecode_data.sum(axis=0)})
for name, d in {"icd10": phecode_data_icd10, "icd9": phecode_data_icd9, "self_report": phecode_data_self_report}.items():
    cases = d.reset_index().groupby(by="ID").any()
    phecode_counts[name + "_cases"] = cases.sum(axis=0)
phecode_counts["phecode_meaning"] = phecode_counts.index.map(phecode_info.phenotype)
print("Most frequent phecodes by source")
print(phecode_counts.sort_values(by="counts", ascending=False).head(20))

# Gather phecode diagnosis information for each subject
for group in phecode_groups:
    # Note that those without any ICD10 entries at all should be marked as non-case, hence the fillna()
    data[group] = data.index.map(phecode_data[group].astype(int)).fillna(0)

# Correlate each block-level code with our activity variable
# Loop through all the activity variables and phecode groups we are interested in
if RECOMPUTE:
    phecode_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
    for group in phecode_groups:
        print(group, )
        N = data[group].sum()
        if N < 50:
            print(f"Skipping {group} - only {N} cases found")
            continue
        
        for activity_variable in activity.columns:
            fit = OLS(f"{activity_variable} ~ Q({group}) + sex * ({covariate_formula})",
                         data=data)
            p = fit.pvalues[f"Q({group})"]
            coeff = fit.params[f"Q({group})"]
            std_effect = coeff / data[activity_variable].std()
            if activity_variable.startswith("self_report"):
                # May contain NaNs, need to drop all NaNs and count accurately
                N_cases = data.loc[~data[activity_variable].isna(), group].sum()
            else:
                N_cases = N
            phecode_tests_list.append({"phecode": group,
                                    "activity_var": activity_variable,
                                    "p": p,
                                    "coeff": coeff,
                                    "std_effect": std_effect,
                                    "N_cases": N_cases,
                                   })
    phecode_tests = pandas.DataFrame(phecode_tests_list)

    phecode_tests['q'] = BH_FDR(phecode_tests.p)
    phecode_tests["phecode_meaning"] = phecode_tests.phecode.map(phecode_info.phenotype)
    phecode_tests["phecode_category"] = phecode_tests.phecode.map(phecode_info.category)

    phecode_tests.to_csv(OUTDIR+f"phecodes.txt", sep="\t", index=False)
else:
    phecode_tests = pandas.read_csv(OUTDIR+"phecodes.txt", sep="\t")
phecode_tests_raw = phecode_tests.copy()

phecode_tests['activity_var_category'] = phecode_tests['activity_var'].map(activity_variable_descriptions.Category)
phecode_tests['q_significant'] = (phecode_tests.q < FDR_CUTOFF_VALUE).astype(int)

# Summarize the phecode test results
num_nonnull = len(phecode_tests) - phecode_tests.p.sum()*2
bonferonni_cutoff = 0.05 / len(phecode_tests)
FDR_cutoff = phecode_tests[phecode_tests.q < 0.05].p.max()
print(f"Of {len(phecode_tests)} tested, approx {int(num_nonnull)} expected non-null")
print(f"and {(phecode_tests.p <= bonferonni_cutoff).sum()} exceed the Bonferonni significance threshold")
print(f"and {(phecode_tests.p <= FDR_cutoff).sum()} exceed the FDR < 0.05 significance threshold")

fig, ax = pylab.subplots()

ax.scatter(phecode_tests.N_cases, -numpy.log10(phecode_tests.p), marker=".")
ax.set_xlabel("Number cases")
ax.set_ylabel("-log10(p-value)")
ax.axhline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
ax.axhline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
ax.set_title("PheCode - Activity associations")
fig.savefig(OUTDIR+"phewas_summary.png")


fig, ax = pylab.subplots()

ax.scatter(phecode_tests.std_effect, -numpy.log10(phecode_tests.p), marker=".")
ax.set_xlabel("Effect size")
ax.set_ylabel("-log10(p-value)")
ax.axhline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
ax.axhline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
ax.set_title("PheCode - Activity associations")
fig.savefig(OUTDIR+"phewas.volcano_plot.png")



### Generate summaries of the phecode test results

## Display the p-values of each actiivty variable
fig, ax = pylab.subplots(figsize=(8,8))
for i, activity_variable in enumerate(activity_variables):
    ps = phecode_tests[phecode_tests['activity_var'] == activity_variable].p
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.random(ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Phecode associations\ngrouped by activity variable")
ax.set_yticks(range(len(activity_variables)))
ax.set_yticklabels(activity_variables)
ax.set_ylim(-1, len(activity_variables))
ax.axvline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
ax.axvline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_activity_variable.png")

## Display p-values by the category of the phecode
fig, ax = pylab.subplots(figsize=(6,8))
phecode_categories = phecode_tests.phecode_category.unique()
for i, category in enumerate(phecode_categories):
    ps = phecode_tests[phecode_tests.phecode_category == category].p
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.uniform(size=ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Phecode associations\ngrouped by phecode category")
ax.set_yticks(range(len(phecode_categories)))
ax.set_yticklabels(phecode_categories)
ax.set_ylim(-1, len(phecode_categories))
ax.axvline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
ax.axvline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_phecode_category.png")

## Display p-values by the inter-intra personal variance ratio
fig, ax = pylab.subplots(figsize=(8,8))
ax.scatter(phecode_tests['activity_var'].map(activity_variance.normalized),
            -numpy.log10(phecode_tests.p))
ax.set_xlabel("Ratio of intra- to inter-personal variance")
ax.set_ylabel("-log10(p-value)")
ax.set_title("p-values by variance ratio")
fig.savefig(OUTDIR+"pvalues_by_variance.png")

## Display effect sizes by the inter-intra personal variance ratio
fig, ax = pylab.subplots(figsize=(8,8))
ax.scatter(phecode_tests['activity_var'].map(activity_variance.normalized),
            phecode_tests.std_effect.abs())
ax.set_xlabel("Ratio of intra- to inter-personal variance")
ax.set_ylabel("Standardized Effect Size")
ax.set_title("Effect sizes by variance ratio")
fig.savefig(OUTDIR+"effect_size_by_variance.png")

## heatmap of phenotype-activity relationships
fig, ax = pylab.subplots(figsize=(9,9))
FDR_CUTOFF_VALUE = 0.05
pvalue_counts = phecode_tests.groupby(["activity_var", "phecode_category"]).q_significant.sum().unstack()
h = ax.imshow(pvalue_counts.values)
ax.set_xticks(range(len(pvalue_counts.columns)))
ax.set_xticklabels(pvalue_counts.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_counts.columns)-0.5)
ax.set_yticks(range(len(pvalue_counts.index)))
ax.set_yticklabels(pvalue_counts.index)
ax.set_ylim(-0.5, len(pvalue_counts.index)-0.5)
ax.set_title(f"Number associations significant (q < {FDR_CUTOFF_VALUE})")
c = fig.colorbar(h)
c.ax.set_ylabel("Number significant in category")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.png")

## same as above but with percent-of-category-significant displayed
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_percent = phecode_tests.groupby(["activity_var", "phecode_category"]).q_significant.mean().unstack()*100
h = ax.imshow(pvalue_percent.values)
ax.set_xticks(range(len(pvalue_percent.columns)))
ax.set_xticklabels(pvalue_percent.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_percent.columns)-0.5)
ax.set_yticks(range(len(pvalue_percent.index)))
ax.set_yticklabels(pvalue_percent.index)
ax.set_ylim(-0.5, len(pvalue_percent.index)-0.5)
ax.set_title(f"Percent phenotypes with significant associations\n(q < {FDR_CUTOFF_VALUE})")
c = fig.colorbar(h)
c.ax.set_ylabel("Percent of category significant")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.percent.png")

## Same as a above showing the hypergeometric test p-value or enrichment
total_significant = phecode_tests.groupby(["activity_var"]).q_significant.sum()
num_tests = phecode_tests.phecode.nunique()
#category_sizes = phecode_tests.groupby(['phecode_category']).phecode.nunique().

def hypergeom_enrichment(data):
    var = data['activity_var'].iloc[0]
    k = data.q_significant.sum()
    M = num_tests
    n = total_significant[var]
    N = len(data)
    p =  scipy.stats.hypergeom.sf(k, M, n, N)
    if n == 0:
        return 1
    return p
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_enrichment_stacked = phecode_tests.groupby(["activity_var", "phecode_category"])[['phecode', 'q_significant', 'activity_var']].apply(hypergeom_enrichment)
pvalue_enrichment = pvalue_enrichment_stacked.unstack()
enrichment_qs = BH_FDR(pvalue_enrichment.values.ravel()).reshape(pvalue_enrichment.shape)
h = ax.imshow(-numpy.log10(enrichment_qs))
ax.set_xticks(range(len(pvalue_enrichment.columns)))
ax.set_xticklabels(pvalue_enrichment.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_enrichment.columns)-0.5)
ax.set_yticks(range(len(pvalue_enrichment.index)))
ax.set_yticklabels(pvalue_enrichment.index)
ax.set_ylim(-0.5, len(pvalue_enrichment.index)-0.5)
ax.set_title("Enrichment of significant phenotypes within a category")
c = fig.colorbar(h)
c.ax.set_ylabel("-log10(enrichment q-value)")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.enrichment.png")

## Heat of activity-category versus phecode-category p-values
fig, ax = pylab.subplots(figsize=(9,5))
pvalue_percent_categories = phecode_tests.groupby(["activity_var_category", "phecode_category"]).q_significant.mean().unstack()*100
h = ax.imshow(pvalue_percent_categories.values)
ax.set_xticks(range(len(pvalue_percent_categories.columns)))
ax.set_xticklabels(pvalue_percent_categories.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_percent_categories.columns)-0.5)
ax.set_yticks(range(len(pvalue_percent_categories.index)))
ax.set_yticklabels(pvalue_percent_categories.index)
ax.set_ylim(-0.5, len(pvalue_percent_categories.index)-0.5)
ax.set_title(f"Percent phenotypes with significant associations\n(q < {FDR_CUTOFF_VALUE})")
c = fig.colorbar(h)
c.ax.set_ylabel("Percent of category significant")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.by_activity_var_category.percent.png")


## PCA of the different phenotypes
# each point is a phenotype and its given the vector of effect sizes relating to the different associations
# is there a pattern/clustering to the phenotypes?
#phecode_effect_vectors = phecode_tests.set_index(["group", "var"]).std_effect.unstack()
#pca = PCA(n_components=2)
#pca_coords = pca.fit_transform(phecode_effect_vectors)
#phecode_pca = pandas.DataFrame({0: pca_coords[:,0], 1:pca_coords[:,1]}, index=phecode_effect_vectors.index)
#phecode_pca['category'] = phecode_pca.index.map(phecode_info.category)
#
#fig, ax = pylab.subplots(figsize=(8,8))
#for category in phecode_info.category.unique():
#    category_points = phecode_pca.loc[phecode_pca.category == category, [0,1]]
#    ax.scatter(category_points[0], category_points[1], label=category)
#ax.legend()
#ax.set_title("PCA of Phenotypes by Activity Effect Sizes")
#fig.savefig(OUTDIR+"phecode_pca.png")

## PCA of the different activity variables
#activity_effect_vectors = phecode_tests.set_index(["var", "group"]).std_effect.unstack()
#pca_coords = pca.fit_transform(activity_effect_vectors)
#activity_pca = pandas.DataFrame({0: pca_coords[:,0], 1:pca_coords[:,1]}, index=activity_effect_vectors.index)
#fig, ax = pylab.subplots(figsize=(8,8))
#ax.scatter(activity_pca[0], activity_pca[1])
#for i, var in enumerate(activity_effect_vectors.index):
#    ax.annotate(var, (activity_pca.loc[var, 0], activity_pca.loc[var,1]))
#ax.legend()
#ax.set_title("PCA of Activity Variables by Phenotype Effect Sizes")
#fig.savefig(OUTDIR+"activity_variable_pca.png")


## Connection analysis
# We say there is a 'connection' between two phenotypes if there is an activity variable
# that associates with both of them

# count the number of connections for each phenotype-pairing
significance_matrix = phecode_tests.set_index(["activity_var", "phecode"]).q_significant.unstack()
connections_phecodes = significance_matrix.T @ significance_matrix
connections_activity = significance_matrix @ significance_matrix.T
def plot_heatmap(data, order=True, label=""):
    fig, ax = pylab.subplots()
    if order:
        dist_x = scipy.spatial.distance.squareform(1/(data @ data.T+1), checks=False)
        linkage_x = scipy.cluster.hierarchy.linkage(dist_x, optimal_ordering=True)
        ordering_x = scipy.cluster.hierarchy.leaves_list(linkage_x)

        dist_y = scipy.spatial.distance.squareform(1/(data.T @ data+1), checks=False)
        linkage_y = scipy.cluster.hierarchy.linkage(dist_y, optimal_ordering=True)
        ordering_y = scipy.cluster.hierarchy.leaves_list(linkage_y)
    else:
        ordering_x = numpy.arange(len(data.index))
        ordering_y = numpy.arange(len(data.columns))

    ax.imshow(data.iloc[ordering_x, ordering_y])

    if "x" in label:
        ax.set_xticks(numpy.arange(len(data.columns)))
        ax.set_xticklabels(data.columns[ordering_x])
    if "y" in label:
        ax.set_yticks(numpy.arange(len(data.index)))
        ax.set_yticklabels(data.index[ordering_y])
    return ax
plot_heatmap(connections_phecodes)
plot_heatmap(connections_activity, label="xy")



# # Test sex differences in RA-diagnosis associations
# 
# We are interested in whether there is a difference between male and female susceptibility to
# loss of circadian rhythm and differences in the impact of loss of circadian rhythm.
# 
# We extract the most significant associations and plot their associations within each sex.

# Correlate each block-level code with our activity variable within each sex

if RECOMPUTE:
    phecode_tests_by_sex_list = []
    sex_covariate_formula = ' + '.join(c for c in covariates if c != 'sex')

    for group in phecode_groups:
        #TODO: skip any sex-specific phecodes
        N = data[group].sum()
        N_male = numpy.sum(data[group].astype(bool) & (data.sex == "Male"))
        N_female = numpy.sum(data[group].astype(bool) & (data.sex == "Female"))
        if N_male <= 50 or N_female < 50:
            print(f"Skipping {group} - only {N_male} M and  {N_female} F cases found")
            continue
            
        if False: #phecode_tests.loc[group, "q"] > 0.01:
            # Skip test, not significant
            print(f"Skipping {group} since q > 0.01")
            continue
        
        for activity_variable in activity.columns:
            fit = OLS(f"{activity_variable} ~ 0 + C(sex, Treatment(reference=-1)) : ({sex_covariate_formula} +  Q({group}))",
                             data=data)


            female_coeff = fit.params[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']
            male_coeff = fit.params[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            p_female = fit.pvalues[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']
            p_male = fit.pvalues[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            diff_test = fit.t_test(f'C(sex, Treatment(reference=-1))[Male]:Q({group}) = C(sex, Treatment(reference=-1))[Female]:Q({group})')
            p_diff = diff_test.pvalue
            conf_ints = fit.conf_int()
            male_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            female_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']

            male_std = data.loc[data.sex == "Male", activity_variable].std()
            female_std = data.loc[data.sex == "Female", activity_variable].std()
            
            phecode_tests_by_sex_list.append({
                "phecode": group,
                "activity_var": activity_variable,
                "std_male_coeff": float(male_coeff) / male_std,
                "std_female_coeff": float(female_coeff) /  female_std,
                "p_male": float(p_male),
                "p_female": float(p_female),
                "p_diff": float(p_diff),
                "N_male": N_male,
                "N_female": N_female,
                "std_male_coeff_low": float(male_conf_int[0]) / male_std,
                "std_male_coeff_high": float(male_conf_int[1]) / male_std,
                "std_female_coeff_low": float(female_conf_int[0]) / female_std,
                "std_female_coeff_high": float(female_conf_int[1]) / female_std,
            })

    phecode_tests_by_sex = pandas.DataFrame(phecode_tests_by_sex_list)

    phecode_tests_by_sex["phecode_meaning"] = phecode_tests_by_sex.phecode.map(phecode_info.phenotype)
    phecode_tests_by_sex["phecode_category"] = phecode_tests_by_sex.phecode.map(phecode_info.category)
    phecode_tests_by_sex = phecode_tests_by_sex.join(phecode_tests.q, how="left")
    phecode_tests_by_sex['q_diff'] = BH_FDR(phecode_tests_by_sex.p_diff)
    phecode_tests_by_sex['differential_std_coeff'] = phecode_tests_by_sex.std_male_coeff - phecode_tests_by_sex.std_female_coeff
    phecode_tests_by_sex.sort_values(by="p_diff", inplace=True)

    phecode_tests_by_sex.to_csv(OUTDIR+"/all_phenotypes.by_sex.txt", sep="\t", index=False)
else:
    phecode_tests_by_sex = pandas.read_csv(OUTDIR+"/all_phenotypes.by_sex.txt", sep="\t")
phecode_tests_by_sex_raw = phecode_tests_by_sex.copy()

### Generate summaries of the phecode test by-sex results

## Display the p-values of each actiivty variable
fig, ax = pylab.subplots(figsize=(8,8))
for i, activity_variable in enumerate(activity_variables):
    ps = phecode_tests_by_sex[phecode_tests_by_sex['activity_var'] == activity_variable].p_diff
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.uniform(size=ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Sex-differences\ngrouped by activity variable")
ax.set_yticks(range(len(activity_variables)))
ax.set_yticklabels(activity_variables)
ax.set_ylim(-1, len(activity_variables))
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_activity_variable.by_sex.png")

## Display p-values by the category of the phecode
fig, ax = pylab.subplots(figsize=(6,8))
phecode_categories = phecode_tests_by_sex.phecode_category.unique()
for i, category in enumerate(phecode_categories):
    ps = phecode_tests_by_sex[phecode_tests_by_sex.phecode_category == category].p_diff
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.uniform(size=ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Sex differences\ngrouped by phecode category")
ax.set_yticks(range(len(phecode_categories)))
ax.set_yticklabels(phecode_categories)
ax.set_ylim(-1, len(phecode_categories))
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_phecode_category.by_sex.png")

# Plot the regression coefficients for each of the phenotypes
num_male = (data.sex == "Male").sum()
num_female = (data.sex == "Female").sum()
color_by_phecode_cat = {cat:color for cat, color in
                            zip(phecode_tests.phecode_category.unique(),
                                [pylab.get_cmap("tab20")(i) for i in range(20)])}
d = phecode_tests_by_sex[True #(phecode_tests_by_sex.q < 0.05 )
                        & (phecode_tests_by_sex.N_male > 300)
                        & (phecode_tests_by_sex.N_female > 300)]
d['male_effect'] = (d["std_male_coeff"]) # / (d["N_male"] / num_male)
d['female_effect'] = (d["std_female_coeff"]) # / (d["N_female"] / num_female)
def sex_difference_plot(d, color_by="phecode_category"):
    if color_by == "phecode_category":
        colormap = color_by_phecode_cat
    else:
        colormap = {cat:color for cat, color in
                            zip(d[color_by].unique(),
                                [pylab.get_cmap("Dark2")(i) for i in range(20)])}
    color = [colormap[c] for c in d[color_by]]
    fig, ax = pylab.subplots(figsize=(9,9))
    # The points
    ax.scatter(
        d.male_effect,
        d.female_effect,
        label="phenotypes",
        #s=-numpy.log10(d.p_diff)*10,
        s=-numpy.log10(numpy.minimum(d.p_male, d.p_female))*4,
        c=color)
    ax.set_title("Effect sizes by sex\nAmong signifcant associations")
    ax.spines['bottom'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.axvline(c="k", lw=1)
    ax.axhline(c="k", lw=1)
    ax.set_xlabel("Effect size in males")
    ax.set_ylabel("Effect size in females")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    # Diagonal y=x line
    bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                numpy.max([ax.get_xlim(), ax.get_ylim()]))
    diag = numpy.array([-bound, bound])
    ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal")
    ax.plot(diag, -diag, linestyle="--", c='k', zorder=-1, label="diagonal")
    bbox = {'facecolor': (1,1,1,0.8), 'edgecolor':(0,0,0,0)}
    ax.annotate("Male Effect Larger", xy=(0.4,0), ha="center", bbox=bbox, zorder=3)
    ax.annotate("Male Effect Larger", xy=(-0.4,0), ha="center", bbox=bbox, zorder=3)
    ax.annotate("Female Effect Larger", xy=(0,0.4), ha="center", bbox=bbox, zorder=3)
    ax.annotate("Female Effect Larger", xy=(0,-0.25), ha="center", bbox=bbox, zorder=3)
    legend_elts = [matplotlib.lines.Line2D(
                            [0],[0],
                            marker="o", markerfacecolor=c, markersize=10,
                            label=cat if not pandas.isna(cat) else "NA",
                            c=c, lw=0)
                        for cat, c in colormap.items()]
    ax.legend(handles=legend_elts, ncol=2, fontsize="small")
    return fig, ax
fig, ax = sex_difference_plot(d)
fig.savefig(f"{OUTDIR}/sex_differences.all_phenotypes.png")

fig, ax = sex_difference_plot(d[d.phecode_category == 'circulatory system'], color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/sex_differences.circulatory.png")
fig, ax = sex_difference_plot(d[d.phecode_category == 'mental disorders'], color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/sex_differences.mental_disorders.png")
fig, ax = sex_difference_plot(d[d.phecode_category == 'endocrine/metabolic'], color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/sex_differences.endocrine.png")
fig, ax = sex_difference_plot(d[d.phecode_category == 'infectious diseases'], color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/sex_differences.infections.png")
fig, ax = sex_difference_plot(d[d.phecode_category == 'respiratory'], color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/sex_differences.respiratory.png")

def local_regression(x,y, out_x, bw=0.05):
    # Preform a local regression y ~ x and evaluate it at the provided points `out_x`
    reg = sm.nonparametric.KernelReg(exog=x, endog=y, var_type='c',
                                     bw=[bw],
                                    )
    fit, mfx = reg.fit(out_x)
    return fit


# Plot disease-incidence rates versus the coefficients, in both male and female
# TODO: should be done selecting only significant associations or all?
fig, ax = pylab.subplots( figsize=(9,9) )
selected = (phecode_tests_by_sex.q < FDR_CUTOFF_VALUE) & (phecode_tests_by_sex.N_male > 100) & (phecode_tests_by_sex.N_female > 100)
d = phecode_tests_by_sex[selected]
ax.scatter(numpy.log10(d.N_male),
            d.std_male_coeff.abs(),
            c="r", label="Male", marker='.')
ax.scatter(numpy.log10(d.N_female),
            d.std_female_coeff.abs(),
            c="b", label="Female", marker='.')
for i in range(1):
    if i > 0:
        d = phecode_tests_by_sex[selected].sample(len(phecode_tests_by_sex),replace=True)
    else:
        d = phecode_tests_by_sex[selected]
    male_smooth = sm.nonparametric.lowess(
                        d.std_male_coeff.abs(),
                        numpy.log10(d.N_male),
                        return_sorted=True,
                        frac=0.4,
                        it=0)
    ax.plot(male_smooth[:,0], male_smooth[:,1], c="r", alpha=1, linewidth=5)
    female_smooth = sm.nonparametric.lowess(
                        d.std_female_coeff.abs(),
                        numpy.log10(d.N_female),
                        return_sorted=True,
                        frac=0.4,
                        it=0)
    ax.plot(female_smooth[:,0], female_smooth[:,1], c="b", alpha=1, linewidth=5)
ax.legend()
ax.set_xlabel("Number of Cases (log10)")
ax.set_ylabel("Standardized Effect Size")
ax.set_title("Phenotype-Rhythmicity association by incidence rate")
#ax.set_ylim(-0.04,0.00)
fig.savefig(OUTDIR+"/all_phenotypes.by_incidence_rate.png")

## Do an "enrichment" study of the set of phenotypes associating in males and females
phecode_tests_by_sex['significant_male'] = BH_FDR(phecode_tests_by_sex.p_male) < 0.1
phecode_tests_by_sex['significant_female'] = BH_FDR(phecode_tests_by_sex.p_female) < 0.1
phecode_tests_by_sex['significant_either'] = phecode_tests_by_sex.significant_male | phecode_tests_by_sex.significant_female
num_significant_male = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"]).significant_male.sum()
num_significant_female = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"]).significant_female.sum()
num_significant_either = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"]).significant_either.sum()

#TODO: is there a meaningful way to test for male/female enrichment by category?
#male_enriched = [scipy.stats.hypergeom(M, n, num_significant_male

# ### Check the overall average of effect size by sex of the RA-phenotype associations

male_weights = 1 / (phecode_tests_by_sex.std_male_coeff_high - phecode_tests_by_sex.std_male_coeff_low)**2 * (phecode_tests_by_sex.std_male_coeff != 0.0)
female_weights = 1 / (phecode_tests_by_sex.std_female_coeff_high - phecode_tests_by_sex.std_female_coeff_low)**2 * (phecode_tests_by_sex.std_female_coeff != 0.0)
rel_male_coeff = numpy.abs(phecode_tests_by_sex.std_male_coeff  * male_weights)
rel_female_coeff = numpy.abs(phecode_tests_by_sex.std_female_coeff * female_weights)

print(f"Weighted mean male effect:   {rel_male_coeff.mean() / male_weights.mean():0.4f}")
print(f"Median male effect:          {phecode_tests_by_sex.std_male_coeff.abs().median():0.4f}")
print(f"Weighted mean female effect: {rel_female_coeff.mean() / male_weights.mean():0.4f}")
print(f"Median female effect:        {phecode_tests_by_sex.std_female_coeff.abs().median():0.4f}")
#print(f"Note: effects are the difference in mean RA values between cases and controls of the phenotype.")
#print(f"   standard deviation of RA:  {data.acceleration_RA.std():0.4f}")

## Heatmap of sex-difference signifiances
fig, ax = pylab.subplots(figsize=(9,9))
FDR_CUTOFF_VALUE = 0.05
phecode_tests_by_sex['q_significant'] = (phecode_tests_by_sex.q_diff < FDR_CUTOFF_VALUE).astype(int)
pvalue_counts = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"]).q_significant.sum().unstack()
h = ax.imshow(pvalue_counts.values)
ax.set_xticks(range(len(pvalue_counts.columns)))
ax.set_xticklabels(pvalue_counts.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_counts.columns)-0.5)
ax.set_yticks(range(len(pvalue_counts.index)))
ax.set_yticklabels(pvalue_counts.index)
ax.set_ylim(-0.5, len(pvalue_counts.index)-0.5)
ax.set_title(f"Number sex-difference associations significant (q < {FDR_CUTOFF_VALUE})")
c = fig.colorbar(h)
c.ax.set_ylabel("Number significant in category")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.by_sex.png")

## same as above but with percent-of-category-significant displayed
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_percent = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"]).q_significant.mean().unstack()*100
h = ax.imshow(pvalue_percent.values)
ax.set_xticks(range(len(pvalue_percent.columns)))
ax.set_xticklabels(pvalue_percent.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_percent.columns)-0.5)
ax.set_yticks(range(len(pvalue_percent.index)))
ax.set_yticklabels(pvalue_percent.index)
ax.set_ylim(-0.5, len(pvalue_percent.index)-0.5)
ax.set_title(f"Percent phenotypes with significant sex-difference associations\n(q < {FDR_CUTOFF_VALUE})")
c = fig.colorbar(h)
c.ax.set_ylabel("Percent of category significant")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.percent.by_sex.png")

## Same as a above showing the hypergeometric test p-value or enrichment
total_significant = phecode_tests_by_sex.groupby(["activity_var"]).q_significant.sum()
num_tests = phecode_tests_by_sex.phecode.nunique()
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_enrichment_stacked = phecode_tests_by_sex.groupby(["activity_var", "phecode_category"])[['phecode', 'q_significant', 'activity_var']].apply(hypergeom_enrichment)
pvalue_enrichment = pvalue_enrichment_stacked.unstack()
enrichment_qs = BH_FDR(pvalue_enrichment.values.ravel()).reshape(pvalue_enrichment.shape)
h = ax.imshow(-numpy.log10(enrichment_qs))
ax.set_xticks(range(len(pvalue_enrichment.columns)))
ax.set_xticklabels(pvalue_enrichment.columns, rotation=90)
ax.set_xlim(-0.5, len(pvalue_enrichment.columns)-0.5)
ax.set_yticks(range(len(pvalue_enrichment.index)))
ax.set_yticklabels(pvalue_enrichment.index)
ax.set_ylim(-0.5, len(pvalue_enrichment.index)-0.5)
ax.set_title("Enrichment of significant sex-difference phenotypes within a category")
c = fig.colorbar(h)
c.ax.set_ylabel("-log10(enrichment q-value)")
fig.tight_layout()
fig.savefig(OUTDIR+"pvalue_significance_heatmap.enrichment.by_sex.png")


## PCA by sex-specific phecodes
#TODO: use the absolute value of the effect sizes here?
#phecode_effect_vectors_male = phecode_tests_by_sex.set_index(["group", "var"])['std_male_coeff'].unstack()
#phecode_effect_vectors_female = phecode_tests_by_sex.set_index(["group", "var"])['std_female_coeff'].unstack()
#pca = PCA(n_components=2)
#pca.fit(pandas.concat([phecode_effect_vectors_male, phecode_effect_vectors_female]))
#pca_coords_male = pca.transform(phecode_effect_vectors_male)
#pca_coords_female = pca.transform(phecode_effect_vectors_female)
#phecode_pca_male = pandas.DataFrame({0: pca_coords_male[:,0], 1: pca_coords_male[:,1]},
#                                        index=phecode_effect_vectors_male.index)
#phecode_pca_female = pandas.DataFrame({0: pca_coords_female[:,0], 1: pca_coords_female[:,1]},
#                                        index=phecode_effect_vectors_female.index)
#fig, ax = pylab.subplots(figsize=(8,8))
#for var in phecode_pca_male.index:
#    ax.plot([phecode_pca_male.loc[var,0], phecode_pca_female.loc[var,0]],
#            [phecode_pca_male.loc[var,1], phecode_pca_female.loc[var,1]],
#            c="k",
#            zorder=-1)
#ax.scatter(phecode_pca_male[0], phecode_pca_male[1], label="male")
#ax.scatter(phecode_pca_female[0], phecode_pca_female[1], label="female")
#ax.legend()
#ax.set_title("PCA of Phenotypes by Activity Effect Sizes by Sex")
#fig.savefig(OUTDIR+"phecode_pca.by_sex.png")


### Associate with quantitiative traits
# Quantitative traits:
import fields_of_interest
quantitative_blocks = [
    fields_of_interest.blood_fields,
    fields_of_interest.urine,
    fields_of_interest.arterial_stiffness,
    fields_of_interest.physical_measures,
]
def find_var(var):
    for v in [var, var+"_V0"]:
        if v in data.columns:
            if pandas.api.types.is_numeric_dtype(data[v].dtype):
                return v
    print(var)
    return None # can't find it
quantitative_vars = [find_var(c) for block in quantitative_blocks
                        for c in block
                        if find_var(c) is not None]

if RECOMPUTE:
    quantitative_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
    for phenotype in quantitative_vars:
        if phenotype in covariates:
            # Can't regress a variable that is also a exogenous variable (namely, BMI)
            continue

        N = data[phenotype].count()
        if N < 50:
            print(f"Skipping {phenotype} - only {N} cases found")
            continue
        
        for activity_variable in activity.columns:
            fit = OLS(f"{phenotype} ~ {activity_variable} + sex * ({covariate_formula})",
                         data=data)
            p = fit.pvalues[activity_variable]
            coeff = fit.params[activity_variable]
            std_effect = coeff * data[activity_variable].std() / data[phenotype].std()
            quantitative_tests_list.append({"phenotype": phenotype,
                                    "activity_var": activity_variable,
                                    "p": p,
                                    "coeff": coeff,
                                    "std_effect": std_effect,
                                    "N": N,
                                   })
    quantitative_tests = pandas.DataFrame(quantitative_tests_list)
    quantitative_tests['q'] = BH_FDR(quantitative_tests.p)
    def base_name(x):
        if "_V" in x:
            return x.split("_V")[0]
        return x
    base_variable_name = quantitative_tests.phenotype.apply(base_name)
    quantitative_tests['ukbb_field'] = base_variable_name.map(fields_of_interest.all_fields)
    quantitative_tests.to_csv(OUTDIR+"/quantitative_traits.txt", sep="\t", index=False)
else:
    quantitative_tests = pandas.read_csv(OUTDIR+"/quantitative_traits.txt", sep="\t")
quantitative_tests_raw = quantitative_tests.copy()

pylab.close('all')


## Summarize the quantitative_trait results
quantitative_bonferonni_cutoff = 0.05 / len(quantitative_tests)
quantitative_FDR_cutoff = quantitative_tests[quantitative_tests.q < 0.05].p.max()

fig, ax = pylab.subplots(figsize=(8,16))
for i, trait in enumerate(quantitative_vars):
    ps = quantitative_tests[quantitative_tests['phenotype'] == trait].p
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.uniform(size=ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Trait associations")
ax.set_yticks(range(len(quantitative_vars)))
ax.set_yticklabels(quantitative_vars)
ax.set_ylim(-1, len(quantitative_vars))
ax.axvline( -numpy.log10(quantitative_bonferonni_cutoff), c="k", zorder = -1 )
ax.axvline( -numpy.log10(quantitative_FDR_cutoff), c="k", linestyle="--", zorder = -1 )
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_quantitative_trait.png")

### Network analysis between quantitative traits and a phenotype
## Connect a quantitative trait to a phenotype if they both associate with the same activity value
quantitative_tests['q_significant'] = (quantitative_tests.q < 0.05).astype(int)
quantitative_associations = quantitative_tests.set_index(["phenotype", "activity_var"]).q_significant.unstack()
phecode_associations = phecode_tests.set_index(["phecode", "activity_var"]).q_significant.unstack()
common_associations = phecode_associations @ quantitative_associations.T
common_associations.index = common_associations.index.map(phecode_info.phenotype)

# Compute the directionality and magnitude of these associations
# I.e. for each activity variable that associates significantly with both a phenotype and a trait
# we compute the direction of the common association between the phenotype and trait
# as the product of the activity variable effects with the two
# Then we aggregate across all the significant associates to get the dominant direction of association
# if any. If all point in the same direction, get +/- 1. If directionality varies randomly, get ~0
quantitative_effects = numpy.sign(quantitative_tests.set_index(["phenotype", "activity_var"]).std_effect.unstack() * quantitative_associations)
phecode_effects = numpy.sign(phecode_tests.set_index(["phecode", "activity_var"]).std_effect.unstack() * phecode_associations)
common_direction = (phecode_effects @ quantitative_effects.T)
common_direction.index = common_direction.index.map(phecode_info.phenotype)
common_direction = (common_direction / common_associations).fillna(0)

ax = plot_heatmap(common_associations)
ax.figure.savefig(OUTDIR+"/common_assocations_heatmap.png")

# Compute the number of expected associations assuming independence
expected_associations = phecode_associations.mean(axis=1).values.reshape((-1,1)) @ quantitative_associations.mean(axis=1).values.reshape((1,-1))
expected_associations = pandas.DataFrame(expected_associations,
                            index = common_associations.index,
                            columns = common_associations.columns)
common_associations_ratio = (common_associations / (expected_associations+0.001)).fillna(0)
ax = plot_heatmap(common_associations_ratio)
ax.figure.savefig(OUTDIR+"/common_assocations_ratio_heatmap.png")

def plot_connections(associations, directionality = None, **kwargs):
    fig, ax = pylab.subplots(**kwargs)
    scale = 1 / associations.max().max()
    index_heights = numpy.linspace(0,1, len(associations.index))
    column_heights = numpy.linspace(0,1, len(associations.columns))
    colormap = pylab.get_cmap("Spectral")
    for i, a in zip(index_heights, associations.index):
        ax.text(0, i, a, horizontalalignment='right')
    for j, b in zip(column_heights, associations.columns):
        ax.text(1, j, b)
    for i,a in zip(index_heights, associations.index):
        for j,b in zip(column_heights, associations.columns):
            conns = associations.loc[a,b]
            if conns < 10:
                continue
            if directionality is not None:
                color = colormap((directionality.loc[a,b] + 1)/2)
            else:
                color = "k"
            ax.plot([0,1], [i,j],
                    alpha= conns * scale,
                    #linewidth = conns * scale * 10,
                    c=color)
    fig.tight_layout()
    return fig, ax
selected_associations = common_associations.sum(axis=1) > 50
fig, ax = plot_connections(common_associations[selected_associations],
                        common_direction[selected_associations],
                        figsize=(8,30))
fig.savefig(OUTDIR+"/common_associations.png")

category_associations = phecode_tests.groupby(["phecode_category", "activity_var"]).q_significant.sum().unstack()
common_category_associations = category_associations @ quantitative_associations.T
common_category_direction = (phecode_effects @ quantitative_effects.T)
common_category_direction = common_category_direction.groupby(
    common_category_direction.index.map(phecode_info.category)
).apply( lambda x: x.sum(axis=0) / x.abs().sum(axis=0)
).fillna(0)

fig, ax = plot_connections(common_category_associations, common_category_direction, figsize=(8,20))
fig.savefig(OUTDIR+"/common_category_associations.png")


## heatmap of hypergeometric enrichment test
## for the phecode - trait associations via activity variables
#common_associations_long = common_associations.unstack().reset_index()
#common_associations_long['category'] = common_associations_long["group"].map(phecode_info.set_index("phenotype").category)
#total_significant = common_associations_long.groupby(["phenotype"]).sum()
#num_tests = common_associations_long[0].sum()
#def hypergeom_enrichment_2(data):
#    print(data.columns)
#    quant_trait = data['phenotype'].iloc[0]
#    phecode_cat = data['category'].iloc[0]
#    k = data.sum()
#    M = num_tests
#    n = total_significant[quant_trait]
#    N = len(data)
#    p =  scipy.stats.hypergeom.sf(k, M, n, N)
#    if n == 0:
#        return 1
#    return p
#common_pvalue_enrichment_stacked = common_associations_long.groupby(["phenotype", "category"]).apply(hypergeom_enrichment_2)
#pvalue_enrichment = pvalue_enrichment_stacked.unstack()
#enrichment_qs = BH_FDR(pvalue_enrichment.values.ravel()).reshape(pvalue_enrichment.shape)
#fig, ax = pylab.subplots(figsize=(9,9))
#h = ax.imshow(-numpy.log10(enrichment_qs))
#ax.set_xticks(range(len(pvalue_enrichment.columns)))
#ax.set_xticklabels(pvalue_enrichment.columns, rotation=90)
#ax.set_xlim(-0.5, len(pvalue_enrichment.columns)-0.5)
#ax.set_yticks(range(len(pvalue_enrichment.index)))
#ax.set_yticklabels(pvalue_enrichment.index)
#ax.set_ylim(-0.5, len(pvalue_enrichment.index)-0.5)
#ax.set_title("Enrichment of significant phenotypes within a category")
#c = fig.colorbar(h)
#c.ax.set_ylabel("-log10(enrichment q-value)")
#fig.tight_layout()


##### Age-associations
if RECOMPUTE:
    print("Computing age associations")
    age_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if (c != 'birth_year'))
    for group in phecode_groups:
        N = data[group].sum()
        if N < 200:
            print(f"Skipping {group} - only {N} cases found")
            continue
        
        for activity_variable in activity.columns:
            #if not phecode_tests[(phecode_tests.phecode == group)
            #                     & (phecode_tests.activity_var == activity_variable)].q_significant.any():
            #    continue # Only check for age-effects in significant main-effects variables

            fit = OLS(f"{activity_variable} ~ Q({group}) * age_at_actigraphy + ({covariate_formula})",
                         data=data)
            p = fit.pvalues[f"Q({group}):age_at_actigraphy"]
            main_coeff = fit.params[f"Q({group})"]
            age_coeff = fit.params[f"Q({group}):age_at_actigraphy"]
            std_effect = age_coeff / data[activity_variable].std()
            age_tests_list.append({"phecode": group,
                                    "activity_var": activity_variable,
                                    "p": p,
                                    "main_coeff": main_coeff,
                                    "age_effect_coeff": age_coeff,
                                    "std_age_effect": std_effect,
                                    "N_cases": N,
                                   })
    age_tests = pandas.DataFrame(age_tests_list)

    age_tests['q'] = BH_FDR(age_tests.p)
    age_tests["phecode_meaning"] = age_tests.phecode.map(phecode_info.phenotype)
    age_tests["phecode_category"] = age_tests.phecode.map(phecode_info.category)

    age_tests.to_csv(OUTDIR+f"phecodes.age_effects.txt", sep="\t", index=False)
else:
    age_tests = pandas.read_csv(OUTDIR+"phecodes.age_effects.txt", sep="\t")

age_tests['activity_var_category'] = age_tests['activity_var'].map(activity_variable_descriptions.Category)


## Plot summary of age tests
mean_age = data.age_at_actigraphy.mean()
young_offset = 55 - mean_age
old_offset = 70 - mean_age
d = pandas.merge(
        age_tests,
        phecode_tests[['phecode', 'activity_var', 'std_effect', 'p', 'q']],
        suffixes=["_age", "_overall"],
        on=["activity_var", "phecode"]).reset_index()
d = d[d.N_cases > 500]
d['age_55_effect'] = d["std_effect"] + d['std_age_effect'] * young_offset
d['age_75_effect'] = d["std_effect"] + d['std_age_effect'] * old_offset

def age_effect_plot(d, legend=True, annotate=True, color_by="phecode_category"):
    fig, ax = pylab.subplots(figsize=(9,9))
    if color_by == "phecode_category":
        colormap = color_by_phecode_cat
    else:
        colormap = {cat:color for cat, color in
                            zip(d[color_by].unique(),
                                [pylab.get_cmap("Dark2")(i) for i in range(20)])}
    color = [colormap[c] for c in d[color_by]]
    # The points
    ax.scatter(
        d.age_55_effect,
        d.age_75_effect,
        s=-numpy.log10(numpy.minimum(d.p_overall, d.p_age))*3,
        c=color)
    ax.set_title("Effect sizes by age\nAmong signifcant associations")
    ax.spines['bottom'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.axvline(c="k", lw=1)
    ax.axhline(c="k", lw=1)
    ax.set_xlabel("Effect size at 55")
    ax.set_ylabel("Effect size at 70")
    ax.set_xlim(-0.45,0.45)
    ax.set_ylim(-0.45,0.45)
    ax.set_xticks(numpy.linspace(-0.4,0.4,11))
    ax.set_yticks(numpy.linspace(-0.4,0.4,11))
    # Diagonal y=x line
    bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                numpy.max([ax.get_xlim(), ax.get_ylim()]))
    diag = numpy.array([-bound, bound])
    ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal", linewidth=1)
    ax.plot(diag, -diag, linestyle="--", c='k', zorder=-1, label="diagonal", linewidth=1)
    ax.set_aspect("equal")
    if annotate:
        bbox = {'facecolor': (1,1,1,0.8), 'edgecolor':(0,0,0,0)}
        ax.annotate("Age 55 Effect Larger", xy=(0.3,0), ha="center", bbox=bbox, zorder=3)
        ax.annotate("Age 55 Effect Larger", xy=(-0.3,0), ha="center", bbox=bbox, zorder=3)
        ax.annotate("Age 70 Effect Larger", xy=(0,0.15), ha="center", bbox=bbox, zorder=3)
        ax.annotate("Age 70 Effect Larger", xy=(0,-0.30), ha="center", bbox=bbox, zorder=3)
        ax.annotate("Equal Effects", xy=(0.3,0.3), ha="center", va="center", bbox=bbox, zorder=3, rotation=45)
        ax.annotate("Opposite Effects", xy=(0.3,-0.3), ha="center", va="center", bbox=bbox, zorder=3, rotation=-45)
    if legend:
        legend_elts = [matplotlib.lines.Line2D(
                                [0],[0],
                                marker="o", markerfacecolor=c, markersize=10,
                                label=cat if not pandas.isna(cat) else "NA",
                                c=c, lw=0)
                            for cat, c in colormap.items()]
        ax.legend(handles=legend_elts, ncol=2, fontsize="small", loc="upper left")
    return fig,ax
fig, ax = age_effect_plot(d)
fig.savefig(f"{OUTDIR}/age_effects.png")

fig, ax = age_effect_plot(d[d.phecode_category == 'mental disorders'], annotate=False, color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/age_effects.mental_disorders.png")
fig, ax = age_effect_plot(d[d.phecode_category == 'circulatory system'], annotate=False, color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/age_effects.circulatory.png")
fig, ax = age_effect_plot(d[d.phecode_category == 'endocrine/metabolic'], annotate=False, color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/age_effects.endorcine.png")
fig, ax = age_effect_plot(d[d.phecode_category == 'genitourinary'], annotate=False, color_by="phecode_meaning")
fig.savefig(f"{OUTDIR}/age_effects.genitourinary.png")


######## PHENOTYPE-SPECIFIC PLOTS

# Plot config by variables
plot_config = {
    "acceleration_RA": {
        "xbottom": 0.6,
        "xtop": 1.0,
        "point_width": 0.01,
        "bandwidth": 0.15,
        "label": "RA",
    },
    "amplitude": {
        "xbottom": 0.1,
        "xtop": 0.9,
        "point_width": 0.01,
        "bandwidth": 0.25,
        "label": "Amplitude",
    },
}

# Fancy style plot
# Only really works for highly abundant phenotypes like hypertension (401)
def fancy_case_control_plot(data, code, var="acceleration_RA", normalize=False, confidence_interval=False, rescale=True, annotate=False):
    CONTROL_COLOR = "teal"
    CASE_COLOR = "orange"
    UNCERTAIN_COLOR = (0.8, 0.8, 0.8)

    case = data[code] == True
    config = plot_config[var]
    xbottom = config['xbottom']
    xtop = config['xtop']
    point_width = config['point_width']
    bandwidth = config['bandwidth']
    if numpy.sum(case) < 200:
        # Small numbers of cases need to be averaged across a wider band
        bandwidth *= 2
    eval_x = numpy.linspace(xbottom, xtop, int(0.5/point_width + 1))

    case_scaling = (case).sum() * point_width if rescale else 1
    control_scaling = (~case).sum() * point_width if rescale else 1
    case_avg = data[var][case].mean()
    control_avg = data[var][~case].mean()
    total_incidence = case.sum()/len(case)

    def densities_and_incidence(data):
        case_density = scipy.stats.gaussian_kde(data[var][case], bw_method=bandwidth)(eval_x) * case_scaling
        control_density = scipy.stats.gaussian_kde(data[var][~case], bw_method=bandwidth)(eval_x) * control_scaling
        incidence = case_density / (control_density  + case_density)
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
    ax3.fill_between(eval_x, 0, case_density, color=CASE_COLOR)
    if confidence_interval:
        ax2.fill_between(eval_x, lower_bound, middle, color='lightgray')
        ax2.fill_between(eval_x, middle, upper_bound, color='lightgray')
    ax2.plot(eval_x, middle, color='k')

    # Plot avgs
    ax1.axvline(control_avg, c='k', linestyle="--")
    ax3.axvline(case_avg, c='k', linestyle="--")
    ax2.axhline(total_incidence, c='k', linestyle="--")

    # Label plot
    ax1.set_ylabel(f"Controls\nN={(~case).sum()}")
    ax2.set_ylabel(f"Prevalence\n(overall={total_incidence:0.1%})")
    ax3.set_ylabel(f"Cases\nN={case.sum()}") 
    ax2.set_xlabel(config['label'])

    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax3.tick_params(bottom=False)
    ax1.yaxis.set_ticks([])
    #ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('right')
    if not normalize:
        ax2.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax2.yaxis.set_ticklabels(["0%", "25%", "50%", "75%","100%"])
    else:
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.yaxis.set_ticks([])

    # Set axis limits
    ax1.set_xlim(xbottom, xtop)
    if not normalize:
        max_density = max(numpy.max(case_density), numpy.max(control_density))
        ax1.set_ylim(0, max_density)
        ax3.set_ylim(0, max_density)
        ax2.set_ylim(0, 1)
    else:
        ax1.set_ylim(0)
        ax3.set_ylim(0)
        ax2.set_ylim(0, numpy.minimum(numpy.max(middle)*1.3, 1.0))
    ax3.invert_yaxis()

    if annotate:
        ax1.annotate("Control mean",
                        xy=(control_avg, numpy.max(control_density)/2),
                        xytext=(-50,0),
                        textcoords="offset pixels",
                        ha="right",
                        va="center",
                        arrowprops={"arrowstyle": "->"})
        ax3.annotate("Case mean",
                        xy=(case_avg, numpy.max(control_density)/2),
                        xytext=(-50,0),
                        textcoords="offset pixels",
                        ha="right",
                        va="center",
                        arrowprops={"arrowstyle": "->"})
        ax2.annotate("Overall prevalence",
                        xy=((xtop*0.9 + xbottom*0.1), total_incidence),
                        xytext=(0,25),
                        textcoords="offset pixels",
                        ha="right",
                        va="center",
                        arrowprops={"arrowstyle": "->"},
                        )
        i = len(eval_x)//5
        ax2.annotate("95% confidence interval",
                        xy=(eval_x[i], upper_bound[i]),
                        xytext=(0,25),
                        textcoords="offset pixels",
                        ha="center",
                        va="bottom",
                        arrowprops={"arrowstyle": "->"},#"-[, lengthB=5.0"},
                        )

    try:
        ax1.set_title(phecode_info.loc[code].phenotype)
    except KeyError:
        ax1.set_title(code)
    return fig

def incidence_rate_by_category(data, code, categories, var="acceleration_RA", normalize=False, confidence_interval=False, rescale=False):
    # Break up individuals by categorical variable (eg: sex, age bins)
    # and plot the incidence rate of the phecode by the variable
    case = data[code] == True

    config = plot_config[var]
    xbottom = config['xbottom']
    xtop = config['xtop']
    point_width = config['point_width']
    bandwidth = config['bandwidth']
    eval_x = numpy.linspace(xbottom, xtop, int(0.5/point_width + 1))

    case_scaling = (case).sum() * point_width if rescale else  point_width
    control_scaling = (~case).sum() * point_width if rescale else point_width
    case_avg = data[var][case].median()
    control_avg = data[var][~case].median()
    total_incidence = case.sum()/len(case)

    def densities_and_incidence(data):
        case_kde = scipy.stats.gaussian_kde(data[var][case], bw_method=bandwidth)
        case_density = case_kde(eval_x) * case_scaling * case_kde.n
        control_kde = scipy.stats.gaussian_kde(data[var][~case], bw_method=bandwidth)
        control_density = control_kde(eval_x) * control_scaling * control_kde.n
        if not normalize:
            incidence = case_density / (control_density  + case_density)
        if normalize:
            incidence = case_density / total_incidence / 2 / (control_density + case_density / total_incidence / 2)
        return case_density, control_density, incidence

    lower_bounds = []
    upper_bounds = []
    middle_values = []
    all_categories = data[categories].cat.categories
    for value in all_categories:
        category_data = data[data[categories] == value]

        if confidence_interval:
            N = 40
        else:
            N = 1
        incidences = numpy.empty((len(eval_x), N))
        for i in range(N):
            sample = category_data.sample(len(category_data), replace=True)
            _, _, incidence = densities_and_incidence(sample)
            incidences[:,i] = incidence
        incidences = numpy.sort(incidences, axis=1)
        lower_bound = incidences[:,0]
        upper_bound = incidences[:,-1]
        middle = incidences[:,incidences.shape[1]//2]


        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        middle_values.append(middle)

    fig, ax = pylab.subplots()

    # Plot the data
    for lower_bound, upper_bound, middle, cat in zip(lower_bounds, upper_bounds, middle_values, all_categories):
        ax.fill_between(eval_x, lower_bound, upper_bound, color='lightgrey', alpha=0.3, label=None)
        ax.plot(eval_x, middle, label=cat)

    # Plot avgs
    ax.axhline(total_incidence, c='k', linestyle="--")

    # Label plot
    ax.set_ylabel(f"Prevalence\n(overall={total_incidence:0.1%})")
    ax.set_xlabel(config['label'])

    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_ticklabels(["0%", "25%", "50%", "75%","100%"])

    # Set axis limits
    ax.set_xlim(xbottom, xtop)
    ax.set_ylim(0, 1)
    try:
        ax.set_title(phecode_info.loc[code].phenotype + ("\n(normalized)" if normalize else ""))
    except KeyError:
        ax.set_title(code)

    fig.legend()
    return fig

# By-age plots
def age_plot(data, var, phecode, difference=False):
    CONTROL_COLOR = "teal"
    CASE_COLOR = "orange"
    fig, ax = pylab.subplots()
    age_at_actigraphy = (data.actigraphy_start_date - birth_year) / pandas.to_timedelta("1Y")
    eval_x = numpy.arange(numpy.floor(numpy.min(age_at_actigraphy)), numpy.ceil(numpy.max(age_at_actigraphy))+1, 3)
    cases = (data[phecode] == 1)
    controls = (data[phecode] == 0)
    def reg_with_conf_interval(subset):
        main = local_regression(age_at_actigraphy.iloc[subset], data.iloc[subset][var], eval_x, bw=3.0)
        samples = []
        for i in range(40):
            s = numpy.random.choice(subset, size=len(subset))
            samples.append(local_regression(age_at_actigraphy.iloc[s], data.iloc[s][var], eval_x, bw=3.0))
        samples = numpy.array(samples)
        samples = numpy.sort(samples, axis=0)
        bottom = samples[0,:]
        top = samples[-1,:]
        return main, bottom, top
    case_mid, case_bottom, case_top = reg_with_conf_interval(numpy.where(cases)[0])
    control_mid, control_bottom, control_top = reg_with_conf_interval(numpy.where(controls)[0])
    if difference == False:
        ax.plot(eval_x, case_mid, label="cases", c=CASE_COLOR)
        ax.plot(eval_x, control_mid, label="controls", c=CONTROL_COLOR)
        ax.fill_between(eval_x, case_bottom, case_top, color=CASE_COLOR, alpha=0.5)
        ax.fill_between(eval_x, control_bottom, control_top, color=CONTROL_COLOR, alpha=0.5)
    else:
        ax.plot(eval_x, case_mid - control_mid, c="k")
        ax.fill_between(eval_x, case_bottom - control_mid, case_top - case_bottom, color="k", alpha=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel(var)
    ax.set_title(phecode_info.loc[phecode].phenotype)
    fig.legend()
    return fig, ax

# Hypertension
fig = fancy_case_control_plot(data, 401, normalize=False, confidence_interval=True, annotate=True)
fig.savefig(OUTDIR+"phenotypes.hypertension.png")

fig = incidence_rate_by_category(data, 401, categories="birth_year_category", confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.hypertension.by_age.png")

sns.lmplot("acceleration_RA", "cholesterol", data=data, hue="birth_year_category", markers='.')
pylab.gcf().savefig(OUTDIR+"phenotypes.LDL.png")

sns.lmplot("acceleration_RA", "hdl_cholesterol", data=data, hue="birth_year_category", markers='.')
pylab.gcf().savefig(OUTDIR+"phenotypes.HDL.png")

sns.lmplot("acceleration_RA", "systolic_blood_pressure_V0", data=data, hue="birth_year_category", markers='.')
pylab.gcf().savefig(OUTDIR+"phenotypes.systolic_blood_pressure.png")

sns.lmplot("acceleration_RA", "diastolic_blood_pressure_V0", data=data, hue="birth_year_category", markers='.')
pylab.gcf().savefig(OUTDIR+"phenotypes.diastolic_blood_pressure.png")

fig, ax = age_plot(data, "amplitude", 401)
fig.savefig(OUTDIR+"phenotypes.hypertension.amplitude_age_effect.png")

fig, ax = age_plot(data, "cosinor_rsquared", 401)
fig.savefig(OUTDIR+"phenotypes.hypertension.cosinor_rsquared_age_effect.png")

# Diabetes
fig = fancy_case_control_plot(data, 250, normalize=False, confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.diabetes.png")

fig = incidence_rate_by_category(data, 250, categories="birth_year_category", confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.diabetes.by_age.png")

sns.lmplot("acceleration_RA", "glycated_heamoglobin", data=data, hue="birth_year_category", markers='.')
pylab.gcf().savefig(OUTDIR+"phenotypes.glycated_heamoglobin.png")

percent_diabetes_with_hypertension = (data[401].astype(bool) & data[250].astype(bool)).mean() / data[250].mean()
print(f"Percentage of participants with diabetes that also have hypertension: {percent_diabetes_with_hypertension:0.2%}")

# Mood disorders
#TODO!!
#fig = fancy_case_control_plot(data, 250, normalize=False, confidence_interval=True)
#fig.savefig(OUTDIR+"phenotypes.diabetes.png")

# Some more phenotypes:
fig = fancy_case_control_plot(data, 585, normalize=True, confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.renal_failure.png")
fig = fancy_case_control_plot(data, 276, normalize=True, confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.disorders_fuild_electrolyte_etc.png")
fig = fancy_case_control_plot(data, 290, normalize=True, confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.delirium_dementia_alzheimers.png")
fig = fancy_case_control_plot(data, 332, normalize=True, confidence_interval=True)
fig.savefig(OUTDIR+"phenotypes.parkinsons.png")

#### Age trajectories plot
#TODO: finalize these plots - are they of any value?
fig, (ax1, ax2) = pylab.subplots(nrows=2)
age_at_actigraphy = (data.actigraphy_start_date - birth_year) / pandas.to_timedelta("1Y")
eval_x = numpy.arange(numpy.floor(numpy.min(age_at_actigraphy)), numpy.ceil(numpy.max(age_at_actigraphy))+1)
for _, row in phecode_tests.sort_values(by="p").head(300).iterrows():
    if row.N_cases < 1500:
        continue
    cases = (data[row.phecode] == 1)
    controls = (data[row.phecode] == 0)
    bandwidth = 2
    case_value = local_regression(age_at_actigraphy[cases], data[cases][row.activity_var], eval_x, bw=bandwidth)
    control_value = local_regression(age_at_actigraphy[controls], data[controls][row.activity_var], eval_x, bw=bandwidth)
    midpoint = len(eval_x)//2
    if case_value[midpoint] - control_value[midpoint] > 0:
        ax = ax1
    else:
        ax = ax2
    ax.plot(eval_x, (case_value - control_value) / data[row.activity_var].std(), c="k", alpha=0.2)
ax.set_xlabel("Age")
ax.set_ylabel("Case-control difference in standard deviations")
fig.savefig(OUTDIR+"age_trajectories.png")

### TIMELINE
# Make a timeline of the study design timing so that readers can easily see when data was collected
ACTIGRAPHY_COLOR = "#1b998b"
REPEAT_COLOR = "#c5d86d"
DIAGNOSIS_COLOR = "#f46036"
ASSESSMENT_COLOR = "#aaaaaa"
DEATH_COLOR = "#333333"
fig, (ax1, ax2, ax3) = pylab.subplots(figsize=(8,6), nrows=3, sharex=True)
#ax2.yaxis.set_inverted(True)
ax1.yaxis.set_label_text("Participants per month")
ax2.yaxis.set_label_text("Diagnoses per month")
ax3.yaxis.set_label_text("Deaths per month")
#ax2.xaxis.tick_top()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
bins = pandas.date_range("2000-1-1", "2019-1-1", freq="1M")
def date_hist(ax, values, bins, **kwargs):
    # Default histogram for dates doesn't cooperate with being given a list of bins
    # since the list of bins doesn't get converted to the same numerical values as the values themselves
    counts, edges = numpy.histogram(values, bins)
    # Fudge factor fills in odd gaps between boxes
    ax.bar(edges[:-1], counts, width=(edges[1:]-edges[:-1])*1.05, **kwargs)
assessment_time = pandas.to_datetime(data.blood_sample_time_collected_V0)
actigraphy_time = pandas.to_datetime(activity_summary.loc[data.index, 'file-startTime'])
actigraphy_seasonal_time = pandas.to_datetime(activity_summary_seasonal.loc[activity_summary_seasonal.ID.isin(data.index), 'file-startTime'], cache=False)
death_time = pandas.to_datetime(data[~data.date_of_death.isna()].date_of_death)
diagnosis_time = pandas.to_datetime(icd10_entries[icd10_entries.ID.isin(data.index)].first_date)
date_hist(ax1, assessment_time, color=ASSESSMENT_COLOR, label="assessment", bins=bins)
date_hist(ax1, actigraphy_time, color=ACTIGRAPHY_COLOR, label="actigraphy", bins=bins)
date_hist(ax1, actigraphy_seasonal_time, color=REPEAT_COLOR, label="repeat actigraphy", bins=bins)
date_hist(ax2, diagnosis_time, color=DIAGNOSIS_COLOR, label="Diagnoses", bins=bins)
date_hist(ax3, death_time, color=DEATH_COLOR, label="Diagnoses", bins=bins)
ax1.annotate("Assessment", (assessment_time.mean(), 1250), ha="center")
ax1.annotate("Actigraphy", (actigraphy_time.mean(), 1250), ha="center")
ax1.annotate("Repeat\nActigraphy", (actigraphy_seasonal_time.mean(), 1250), ha="center")
ax2.annotate("Medical Record\nDiagnoses", (diagnosis_time.mean(), 1200), ha="center")
ax3.annotate("Deaths", (death_time.mean(), 13), ha="center")
fig.savefig(OUTDIR+"summary_timeline.png")

time_difference = (actigraphy_time - assessment_time).mean()
print(f"Mean difference between actigraphy time and initial assessment time: {time_difference/pandas.to_timedelta('1Y')} years")


### Diagnosis summary
num_diagnoses = icd10_entries.groupby(pandas.Categorical(icd10_entries.ID, categories=data.index)).size()
icd10_entries_at_actigraphy = icd10_entries[pandas.to_datetime(icd10_entries.first_date) < pandas.to_datetime(icd10_entries.ID.map(activity_summary['file-startTime']))]
num_diagnoses_at_actigraphy = icd10_entries_at_actigraphy.groupby(pandas.Categorical(icd10_entries_at_actigraphy.ID, categories=data.index)).size()
icd10_entries_at_assessment = icd10_entries[pandas.to_datetime(icd10_entries.first_date) < pandas.to_datetime(icd10_entries.ID.map(data['blood_sample_time_collected_V0']))]
num_diagnoses_at_assessment = icd10_entries_at_assessment.groupby(pandas.Categorical(icd10_entries_at_assessment.ID, categories=data.index)).size()
ID_without_actigraphy = ukbb.index[ukbb.actigraphy_file.isna()]
icd10_entries_without_actigraphy = icd10_entries_all[icd10_entries_all.ID.isin(ID_without_actigraphy)]
num_diagnoses_no_actigraphy = icd10_entries_without_actigraphy.groupby(pandas.Categorical(icd10_entries_without_actigraphy.ID, categories=ID_without_actigraphy)).size()
fig,ax = pylab.subplots()
ax.boxplot([num_diagnoses_at_assessment, num_diagnoses_at_actigraphy, num_diagnoses, num_diagnoses_no_actigraphy], showfliers=False)
ax.set_xticklabels(["At Assessment", "At Actigraphy", "Final", "Without Actigraphy\nFinal"])
ax.set_ylabel("Medical Record Diagnoses per Participant")
ax.set_title("Disease Burden")
fig.savefig(OUTDIR+"summary_disease_burden.png")

print(f"Median number of diagnoses by category:")
print("At assessment:", num_diagnoses_at_assessment.describe())
print("At actigraphy:", num_diagnoses_at_actigraphy.describe())
print("Final:", num_diagnoses.describe())
print("Final without actigraphy:", num_diagnoses_no_actigraphy.describe())


### Death data
# Survival curves
start_date = pandas.to_datetime(data.date_of_death).min()
def survival_curve(data, ax, **kwargs):
    N = len(data)
    data = data[~data.date_of_death.isna()]
    date = pandas.to_datetime(data.date_of_death).sort_values()
    date_ = pandas.concat((pandas.Series([start_date]), date))
    ax.step(date_,
            (1 - numpy.concatenate(([0], numpy.arange(len(data))))/N)*100,
            where='post',
            **kwargs)

RA_quintiles = pandas.cut(data.acceleration_RA,
                          data.acceleration_RA.quantile([0,0.2,0.4,0.6,0.8,1.0]))
quintile_labels = ["First", "Second", "Third", "Fourth", "Fifth"]

def quintile_survival_plot(data, var, var_label=None):
    if var_label is None:
        var_label = var
    quintiles = pandas.qcut(data[var], 5)
    fig, ax = pylab.subplots(figsize=(8,6))
    for quintile, label in list(zip(quintiles.cat.categories, quintile_labels))[::-1]:
        survival_curve(data[quintiles == quintile], ax, label= var_label + " " + label + " Quintile")
    fig.legend(loc=(0.15,0.15))
    ax.set_ylabel("Survival Probability")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax2 = ax.twinx() # The right-hand side axis label
    scale = len(data)/5
    ax2.set_ylim(ax.get_ylim()[0]*scale, ax.get_ylim()[1]*scale)
    ax2.set_ylabel("Participants")
    fig.tight_layout()
    return fig

def categorical_survival_plot(data, var, var_label=None):
    if var_label is None:
        var_label = var
    fig, ax = pylab.subplots(figsize=(8,6))
    for cat in data[var].cat.categories:
        d = data[data[var] == cat]
        survival_curve(d, ax, label= cat)
    fig.legend(loc=(0.15,0.15))
    ax.set_ylabel("Survival Probability")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    fig.tight_layout()
    return fig

# Survival by RA
fig = quintile_survival_plot(data, "acceleration_RA", "RA")
fig.savefig(OUTDIR+"survival.RA.png")

# Survival by main_sleep_offset_avg
fig = quintile_survival_plot(data, "main_sleep_offset_mean", "Sleep Offset")
fig.savefig(OUTDIR+"survival.main_sleep_offset_mean.png")

# Survival by MVPA_overall_avg
fig = quintile_survival_plot(data, "MVPA_overall", "MVPA Mean")
fig.savefig(OUTDIR+"survival.MVPA_overall.png")

# Survival by MVPA_overall_avg
fig = quintile_survival_plot(data, "MVPA_hourly_SD", "MVPA Std Dev")
fig.savefig(OUTDIR+"survival.MVPA_hourly_SD.png")

# Survival by phase
fig, ax = pylab.subplots()
data['phase_adjusted'] = (data.phase - 8) % 24 + 8
fig = quintile_survival_plot(data, "phase_adjusted", "phase")
fig.savefig(OUTDIR+"survival.phase.png")

# Survival by RA and sex
fig, axes = pylab.subplots(ncols=2, sharey=True, sharex=True, figsize=(10,5))
for ax, sex in zip(axes, ["Male", "Female"]):
    for quintile, label in list(zip(RA_quintiles.cat.categories, quintile_labels))[::-1]:
        survival_curve(data[(data.sex == sex) & (RA_quintiles == quintile)], ax,
                        label=("RA " + label + " Quintile" if sex == "Male" else None))
        ax.set_title(sex)
ax1, ax2 = axes
#ax1.xaxis.set_tick_params(labelrotation=45, ha="right")
#ax2.xaxis.set_tick_params(labelrotation=45, ha="right")
ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
ax1.set_ylabel("Survival Probability")
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
fig.legend()
fig.savefig(OUTDIR+"survival.RA.by_sex.png")
# Survival summary
def survival_plot(data, var, ax, **kwargs):
    quintiles = pandas.qcut(data[var].rank(method="first"), 5)
    for quintile, label in list(zip(quintiles.cat.categories, quintile_labels))[::-1]:
        survival_curve(data[quintiles == quintile], label=label, ax=ax, **kwargs)
fig, axes = pylab.subplots(ncols=5, nrows=len(activity_variables)//5+1, figsize=(10,len(activity_variables)//5*3))
for variable, ax in zip(activity_variables, axes.flatten()):
    survival_plot(data, variable, ax)
    ax.set_title(variable)
    #TODO this is broken!
    #TODO: sort by the significance? or by category?

### Tests Survival
# Cox proportional hazards model
data['date_of_death_censored'] = pandas.to_datetime(data.date_of_death)
data.date_of_death_censored.fillna(data.date_of_death_censored.max(), inplace=True)
data['date_of_death_censored_number'] = (data.date_of_death_censored - data.date_of_death_censored.min()).dt.total_seconds()
uncensored = (~data.date_of_death.isna()).astype(int)
birth_year = pandas.to_datetime(data.birth_year.astype(int).astype(str) + "-01-01")
data['age_at_death_censored'] = (pandas.to_datetime(data.date_of_death) - birth_year) / pandas.to_timedelta("1Y")
entry_age = (data.actigraphy_start_date - birth_year) / pandas.to_timedelta("1Y")
data.age_at_death_censored.fillna(data.age_at_death_censored.max(), inplace=True)

if RECOMPUTE:
    covariate_formula = ' + '.join(["BMI", "smoking"])
    survival_tests_data = []
    for var in activity_variables:
        print(var)
        for method in ['newton', 'cg']:# Try two convergence methods - some work for only one method
            try:
                formula = f"age_at_death_censored ~ {var} + sex + {covariate_formula}"
                result = smf.phreg(formula=formula,
                                    data=data,
                                    status=uncensored,
                                    entry=entry_age,
                                    ).fit(method=method)
                print('.')
                interaction_formula = f"age_at_death_censored ~ {var} * sex + {covariate_formula}"
                interaction_result = smf.phreg(formula=interaction_formula,
                                    data=data,
                                    status=uncensored,
                                    entry=entry_age,
                                    ).fit(method=method)
            except numpy.linalg.LinAlgError:
                print(f"MLE fit method {method} failed, trying alternative")
                continue
            break
        pvalues = pandas.Series(result.pvalues, index=result.model.exog_names)
        params = pandas.Series(result.params, index=result.model.exog_names)
        interaction_pvalues = pandas.Series(interaction_result.pvalues, index=interaction_result.model.exog_names)
        survival_tests_data.append({
            "activity_var": var,
            "p": pvalues[var],
            "log Hazard Ratio": params[var],
            "standardized log Hazard Ratio": params[var] * data[var].std(),
            "sex_difference_p": interaction_pvalues[f"{var}:sex[T.Male]"],
        })
    survival_tests = pandas.DataFrame(survival_tests_data)
    survival_tests['q'] = BH_FDR(survival_tests.p)
    survival_tests['sex_difference_q'] = BH_FDR(survival_tests.sex_difference_p)
    survival_tests = pandas.merge(survival_tests, activity_variable_descriptions[["Category", "Subcategory", "Units"]], left_on="activity_var", right_index=True)
    survival_tests.to_csv(OUTDIR+"survival.by_activity_variable.txt", sep="\t", index=False)
else:
    survival_tests = pandas.read_csv(OUTDIR+"survival.by_activity_variable.txt", sep="\t")
survival_tests_raw = survival_tests.copy()

###  Connecting actigraphy and quantitative traits
Q_CUTOFF = 1e-9
significant_phecode_tests = phecode_tests[phecode_tests.q <= Q_CUTOFF]
significant_quantitative_tests = quantitative_tests[quantitative_tests.q <= Q_CUTOFF]
both_significant = significant_phecode_tests[["phecode", "activity_var"]].set_index("activity_var").join(significant_quantitative_tests[["phenotype", "activity_var"]].set_index("activity_var")).reset_index()
# drop bogus 'BMI' correlation, since included in the covariates, as well as 'nan'
both_significant = both_significant.query("phenotype != 'BMI' and phenotype == phenotype")

# Plot all the connected actigraphy <-> quantitative traits
import itertools
colormap = matplotlib.cm.get_cmap("Dark2").colors
def colormap_extended(colormap):
    i = 1
    def modify(c):
        return 1-(1-c)/i**0.7
    while True:
        yield from ((modify(c[0]), modify(c[1]), modify(c[2])) for c in colormap)
        i += 1
colormap = colormap_extended(colormap)
activity_var_colors = {var:color for var,color in zip(both_significant['activity_var'].unique(), colormap)}
max_N = both_significant.groupby(["phenotype", "phecode"]).count().max()
max_in_row = 4
def split_label(label, max_chars):
    #Split on whitespace then combine, inserting newlines whenever too many characters are reached
    words = label.split()
    parts = []
    line = []
    for word in words:
        line.append(word)
        if sum(len(x) for x in line) >= max_chars:
            # End line, too long
            parts.append(' '.join(line))
            line = []
    parts.append(' '.join(line))
    return '\n'.join(parts).strip()
def plot_interconnections(data):
    selected_phecodes = data.phecode.unique()
    selected_traits = data.phenotype.unique()
    fig_width = len(selected_phecodes)*0.6 + 3
    fig, ax = pylab.subplots(figsize=(fig_width,8))
    for i, phecode in enumerate(selected_phecodes):
        for j, trait_var in enumerate(selected_traits):
            d = data[(data.phecode == phecode)
                                    & (data.phenotype == trait_var)]
            for k, activity_var in enumerate(d.activity_var):
                x = i + (k % max_in_row  - 0.5 * min(len(d)-1, max_in_row-1)) / (max_in_row + 1)
                y = j + (k // max_in_row) / (max_N // max_in_row + 1)
                ax.plot([x], [y], "o", c=activity_var_colors[activity_var], label=activity_var)
    ax.set_xticks(numpy.arange(len(selected_phecodes)))
    ax.set_xticklabels(phecode_info.loc[selected_phecodes].phenotype.apply(lambda x: split_label(x,15)),
                            rotation=90, ha="center")
    ax.set_yticks(numpy.arange(len(selected_traits)))
    ax.set_yticklabels(selected_traits)
    fig.tight_layout()
    return fig, ax
mental_health_phecodes = [327, 296, 300]
fig, ax = plot_interconnections(both_significant[~both_significant.phecode.isin(mental_health_phecodes)])
fig.savefig(OUTDIR+"interconnections.general.png")
fig, ax = plot_interconnections(both_significant[both_significant.phecode.isin(mental_health_phecodes)])
fig.savefig(OUTDIR+"interconnections.mental_health.png")


# Draw a legend
fig, ax = pylab.subplots()
ax.set_visible(False)
fig.legend([matplotlib.lines.Line2D([0],[0], linestyle='', marker="o", color=activity_var_colors[var]) for var in activity_var_colors.keys()],
    activity_var_colors.keys(),
    loc="center",
    #bbox_to_anchor=(1.1,1),
    ncol=2)
fig.tight_layout()
fig.savefig(OUTDIR+"interconnections.legend.png")



### NETWORK ANALYSIS

def weight(p):
    if p == 0:
        return 100
    else:
        return -numpy.log10(p)
P_VALUE_CUTOFF = 1e-10
g_all = networkx.Graph()
g_all.add_nodes_from(activity_variables, type="activity_var")
g_all.add_nodes_from(phecode_tests.phecode_meaning.unique(), type="phecode")
g_all.add_nodes_from(quantitative_tests.phenotype.unique(), type="trait")
g_all.add_edges_from((row['activity_var'], row.phecode_meaning, {"p": row.p, "weight": weight(row.p)})
                    for _, row in phecode_tests.iterrows() if row.p < P_VALUE_CUTOFF)
g_all.add_edges_from((row.activity_var, row.phenotype, {"p": row.p, "weight": weight(row.p)})
                    for _, row in quantitative_tests.iterrows() if row.p < P_VALUE_CUTOFF)
if 'BMI' in g_all:
    g_all.remove_node('BMI') # Bogus
# Remove nodes without any edges
g = g_all.edge_subgraph(g_all.edges)

node_color_dict = {'phecode': '#ef476f', 'activity_var': '#118ab2', 'trait': '#06d6a0'}
node_colors = [node_color_dict[g.nodes[n]['type']] for n in g.nodes]
fig, ax = pylab.subplots(figsize=(12,10))
networkx.draw(g, with_labels=True,
                  node_color=node_colors,
                  edge_color="#aaaaaa")
fig.savefig(OUTDIR+"network.png")

def network_to_table(g, output_prefix):
    edge_table = pandas.concat([
            pandas.DataFrame(g.edges).rename(columns={0:"node1", 1:"node2"}),
            pandas.DataFrame(g.edges.values()),
        ],
        axis = 1)
    node_table = pandas.concat([
            pandas.DataFrame(g.nodes).rename(columns={0:"node"}),
            pandas.DataFrame(g.nodes.values()),
        ],
        axis = 1)
    node_table = pandas.merge(
        node_table,
        activity_variable_descriptions[["Category", "Subcategory", "Units"]],
        left_on="node",
        right_index=True,
        how="left")
    node_table = pandas.merge(
        node_table,
        numpy.log10(survival_tests.set_index('activity_var')[["p"]]).rename(columns=lambda x: "survival_log10" + x),
        left_on="node",
        right_on="activity_var",
        how="left")
    edge_table.to_csv(OUTDIR+output_prefix+".edges.txt", index=False, sep="\t")
    node_table.to_csv(OUTDIR+output_prefix+".nodes.txt", index=False, sep="\t")
network_to_table(g, "network")


g_min = networkx.empty_graph([n for n in g.nodes if n in activity_variables])
def weights_between(n1, n2):
    common_neighbors = set(g.adj[n1]).intersection(g.adj[n2])
    return sum(g.adj[n1][nbr]['weight'] * g.adj[n2][nbr]['weight']
                for nbr in common_neighbors)
g_min.add_edges_from((n1, n2, {'weight': weights_between(n1, n2)})
                        for n1 in g_min.nodes
                        for n2 in g_min.nodes)
g_min.remove_edges_from(edge for edge, data in g_min.edges.items() if data['weight'] < 100)
fig, ax = pylab.subplots(figsize=(12,10))
networkx.draw(g_min, with_labels=True, edge_color="#aaaaaa")
#fig.savefig(OUTDIR+"network.just_activity_vars.png")

# graphs by just the activity variable type
fig, axes = pylab.subplots(nrows=2,ncols=3, figsize=(15,15))
for ax, cat in zip(axes.flatten(), activity_variable_descriptions.Category.unique()):
    category = activity_variable_descriptions.index[activity_variable_descriptions.Category == cat]
    g_cat = g.edge_subgraph([e for e in g.edges
                                if (e[0] in category or e[1] in category)])
    networkx.draw(g_cat,
                    ax=ax,
                    with_labels=True,
                    edge_color="#aaaaaa",
                    node_color=[node_color_dict[g_cat.nodes[n]['type']] for n in g_cat.nodes])
    ax.set_title(cat)



### Assess variability versus average values for acc
# P-values show that acc_overall_sd associates more strongly with many phenotypes than acc_overall_avg does
# but the two are very strongly correlated. We need to test if there really is a significant difference between them
#results = smf.logit(f"Q(401) ~ sex + household_income + smoking + BMI + birth_year + acc_overall + acc_overall_sd", data=data).fit()
#print("Comparing physical activity variability to averages:")
#print("(In hypertension)")
#print(results.summary())
#print(f"p-value that acc_overall_sd contributes above and beyond acc_overall_avg: p = {results.pvalues['acc_overall_sd']}")
#
#formula = f"date_of_death_censored_number ~ birth_year + sex + BMI + smoking + acc_overall_avg + acc_overall_sd"
#results = smf.phreg(formula=formula,
#                    data=data,
#                    status=uncensored,
#                    ).fit()
#print(formula)
#print(results.summary())
#print(f"p-value that acc_overall_sd contributes above and beyond acc_overall_avg: p = {results.pvalues[-1]}")


### Assess the repeatability of some variables
#full_repeat_data = pandas.read_hdf("../processed/repeat.ukbb_data_table.h5")
#repeat_data = full_repeat_data[full_repeat_data.index.isin(data.index)]
#
#for key in self_report_circadian_variables.keys():
#    question_data = data[[key]].join(repeat_data[repeat_data.columns[repeat_data.columns.str.contains(key)]])


### Plot survival assocations versus inter/intra personal variance for validation
fig, ax = pylab.subplots(figsize=(8,6))
ax.scatter(-numpy.log10(survival_tests.p),
            survival_tests.activity_var.map(activity_variance.normalized),
            c='k')
ax.set_xlabel("Survival Association -log10(p)")
ax.set_ylabel("Within-person variation / Between-person variation")
ax.set_ylim(0,1)
for indx, row in survival_tests.sort_values(by="p").head(5).iterrows():
    # Label the top points
    ax.annotate(
        row.activity_var,
        (-numpy.log10(row.p), activity_variance.loc[row.activity_var].normalized),
        xytext=(0,15),
        textcoords="offset pixels",
        arrowprops={'arrowstyle':"->"})
fig.tight_layout()
fig.savefig(OUTDIR+"survival_versus_variation.svg")

### Correlation plot of top survival variables
top_survival_vars = survival_tests.sort_values(by="p").activity_var.head(20)
def matrix_plot(data, **kwargs):
    fig, ax = pylab.subplots(figsize=(10,10))
    h = ax.imshow(data, **kwargs)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90)
    ax.set_xlim(-0.5, len(data.index)-0.5)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_ylim(-0.5, len(data.index)-0.5)
    c = fig.colorbar(h)
    fig.tight_layout()
    return fig, ax
fig, ax = matrix_plot(data[top_survival_vars].corr(), vmin=-1, vmax=1, cmap="bwr")
fig.savefig(OUTDIR+"correlation.top_survival_activity_vars.png")




### Assess what variables add to acceleration_RA the most
if RECOMPUTE:
    beyond_RA_tests_list = []
    for var in activity_variables:
        if var == "acceleration_RA":
            continue
        formula = f"date_of_death_censored_number ~ birth_year + sex + BMI + smoking + acceleration_RA + {var}"
        try:
            results = smf.phreg(formula=formula,
                                data=data,
                                status=uncensored,
                                ).fit()
        except numpy.linalg.LinAlgError:
            print(f"Failed regression on {var} - skipping")
            continue
        beyond_RA_tests_list.append({
            "activity_var": var,
            "p": results.pvalues[-1],
            "standardized log Hazard Ratio": result.params[-1] * data[var].std(),
        })
    beyond_RA_tests = pandas.DataFrame(beyond_RA_tests_list)
    beyond_RA_tests = pandas.merge(beyond_RA_tests, activity_variable_descriptions[["Category", "Subcategory"]],
                            left_on="activity_var",
                            right_index=True)
    beyond_RA_tests.to_csv(OUTDIR+"beyond_RA_tests.txt", sep="\t", index=False)
else:
    beyond_RA_tests = pandas.read_csv(OUTDIR+"beyond_RA_tests.txt", sep="\t")


### Assess the correlations of the various measures
base_vars = ["acceleration", "moderate", "walking", "sleep", "sedentary", "tasks_light", "MET", "MVPA"]
additions = ["_overall", "_hourly_SD", "_within_day_SD", "_between_day_SD", "_peak_value_mean", "_RA", "_IV"]
results = {}
for base_var in base_vars:
    def corr(v1,v2):
        try:
            #return numpy.corrcoef(data[v1], data[v2])[0,1]
            return scipy.stats.spearmanr(data[v1], data[v2], nan_policy="omit")[0]
        except Exception as e:
            print(e)
            return float("NaN")
    row = {(a1,a2): corr(base_var + a1, base_var + a2)
            for a1 in additions
            for a2 in additions
            if a1 != a2}
    results[base_var] = row
correlations = pandas.DataFrame(results)

# Investigate using a subset of the activity variables for clarity
# by removing highly correlated variables (> 0.9)
selected_activity_variables = []
correlations = data[activity_variables].corr()
ordered_activity_variables = survival_tests.sort_values(by="p").activity_var
for var in ordered_activity_variables:
    corrs = correlations.loc[var][selected_activity_variables]
    if (corrs.abs() < 0.9).all():
        selected_activity_variables.append(var)
    else:
        print(f"Dropping {var} due to {corrs.abs().idxmax()}")

### Overall disease burden (number of phecodes) versus RA
fig, ax = pylab.subplots()
num_phecodes = data[phecode_groups].sum(axis=1)
phecode_ranges = pandas.cut(num_phecodes, [0,1,2,4,8,16,32,num_phecodes.max()+1])
xticklabels = []
for i, (phecode_range, group) in enumerate(data.groupby(phecode_ranges)):
    ax.boxplot(group.acceleration_RA.values, positions=[i], showfliers=False, widths=0.8)
    if phecode_range.right == phecode_range.left+1:
        xticklabels.append(f"{int(phecode_range.left)}")
    else:
        xticklabels.append(f"{int(phecode_range.left)}-{int(phecode_range.right-1)}")
ax.set_xticks(range(len(xticklabels)))
ax.set_xticklabels(xticklabels)
ax.set_xlabel("Number of unique diagnoses")
ax.set_ylabel("RA")
fig.savefig(OUTDIR+"num_phecodes.RA.png")

### Relative Risks
# Create interpretable risks from the phecode associations
def invlogit(s):
    return numpy.exp(s)/(1 + numpy.exp(s))
if RECOMPUTE:
    d = phecode_tests[(phecode_tests.q < 0.01)].copy()
    covariate_formula = 'sex + age_at_actigraphy + BMI'
    #covariate_formula = ' + '.join(covariates)
    risk_quantification_data = []
    for _, row in d.iterrows():
        #results = smf.logit(f"Q({row.phecode}) ~ {row.activity_var} + {covariate_formula}", data=data).fit()
        #risk_quantification_data.append({
        #    "p": results.pvalues[row.activity_var],
        #    "coef": results.params[row.activity_var],
        #    "converged": results.mle_retvals['converged'],
        #})
        data_ = data[[row.phecode, row.activity_var] + covariates +['age_at_actigraphy']].copy()
        if row.activity_var.startswith("self_report"):
            data_['high_low'] = data_[row.activity_var].astype(float)
        else:
            bottom_quintile = data_[row.activity_var].quantile(0.2)
            top_quintile = data_[row.activity_var].quantile(0.8)
            data_['high_low'] = 0
            data_.loc[(data_[row.activity_var] > bottom_quintile), 'high_low'] = float("NaN")
            data_.loc[(data_[row.activity_var] > top_quintile), 'high_low'] = 1
        results = smf.ols(f"Q({row.phecode}) ~ high_low + {covariate_formula}", data=data_).fit()
        mean_person = pandas.Series(numpy.mean(results.model.exog, axis=0), index=results.model.exog_names)
        mean_person['high_low'] = 0
        incidence = (data_[~data_.high_low.isna()][row.phecode].mean())
        effect_size = results.params['high_low']
        try:
            results_logit = smf.logit(f"Q({row.phecode}) ~ {row.activity_var} + {covariate_formula}", data=data_).fit()
            marginal_effect = results_logit.get_margeff().summary_frame().loc[row.activity_var, 'dy/dx']
        except numpy.linalg.LinAlgError:
            marginal_effect = float("NaN")
        risk_quantification_data.append({
            "activity_var": row.activity_var,
            "phecode": row.phecode,
            "phecode_meaning": row.phecode_meaning,
            "phecode_category": row.phecode_category,
            "effect_size": effect_size,
            "incidence": incidence,
            "relative_risk": (incidence - effect_size/2) / (incidence + effect_size/2),
            "relative_risk_pred": high_prediction/low_prediction,
            'raw_low_incidence': data_.loc[data_.high_low == 0, row.phecode].mean(),
            'raw_high_incidence': data_.loc[data_.high_low == 1, row.phecode].mean(),
            "marginal_effect": marginal_effect,
            "p": results.pvalues['high_low'],
            "N_cases": data_.loc[~data_.high_low.isna(), row.phecode].sum(),
        })
    risk_quantification = pandas.DataFrame(risk_quantification_data)
    risk_quantification.to_csv(OUTDIR+"relative_risks.txt", sep="\t", index=False)
else:
    risk_quantification = pandas.read_csv(OUTDIR+"relative_risks.txt", ,sep="\t")

## Plot relative risks
fig, ax = pylab.subplots(figsize=(9,9))
#color = risk_quantification.phecode_category.map(color_by_phecode_cat)
colorby = risk_quantification.activity_var.map(activity_variable_descriptions.Subcategory)
colormap = {cat:color for cat, color in
                    zip(colorby.unique(),
                        [pylab.get_cmap("Dark2")(i) for i in range(20)])}
color = colorby.map(colormap)
ax.scatter(
    numpy.log10(risk_quantification.relative_risk),
    #numpy.log10(risk_quantification.raw_low_incidence/risk_quantification.raw_high_incidence),
    -numpy.log10(risk_quantification.p),
    s = risk_quantification.N_cases/60,
    c = color)
ax.set_xlabel("log10 relative risk")
ax.set_ylabel("-log10 p-value")

#### Investigate medications
medications = pandas.read_csv("../processed/ukbb_medications.txt", sep="\t")
metformin_code = 1140884600
metformin = (medications.medication_code == metformin_code).groupby(medications.ID).any()
data['metformin'] = (data.index.map(metformin) == True)
print("Metformin analysis:")
print( data.groupby("metformin").phase.describe())
#TODO: metformin analysis in more detail

#### Combine all tables into the summary with the header file
print(f"Writing out the complete results table to {OUTDIR+'results.xlsx'}")
import openpyxl
workbook = openpyxl.load_workbook("../table_header.xlsx")
descriptions = workbook['Variables']
start_column = len(list(descriptions.tables.values())[0].tableColumns) + 1
var_stats = data[activity_variable_descriptions.index].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).T
for i, col in enumerate(var_stats.columns):
    descriptions.cell(1, start_column + i, col)
    for j, value in enumerate(var_stats[col].values):
        descriptions.cell(j+2, start_column + i, value)
workbook.save(OUTDIR+"results.xlsx")
with pandas.ExcelWriter(OUTDIR+"results.xlsx", mode="a") as writer:
    survival_tests_raw.sort_values(by="p").to_excel(writer, sheet_name="Survival Associations", index=False)
    phecode_tests_raw.sort_values(by="p").to_excel(writer, sheet_name="Phecode Associations", index=False)
    quantitative_tests_raw.sort_values(by="p").to_excel(writer, sheet_name="Quantitative Associations", index=False)
    phecode_tests_by_sex_raw.sort_values(by="p_diff").to_excel(writer, sheet_name="Sex-specific Associations", index=False)
    age_tests.sort_values(by="p").to_excel(writer, sheet_name="Age-dependence", index=False)
