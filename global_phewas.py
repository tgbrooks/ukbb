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
import matplotlib.patches as mpatches
from scipy.cluster import hierarchy
import pylab
import pandas

COHORT = 1
OUTDIR = f"../global_phewas/cohort{COHORT}/"

full_activity = pandas.read_csv("../processed/activity_features_aggregate.txt", index_col=0, sep="\t")
activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")

ukbb.columns = ukbb.columns.str.replace("[,:/]","_") # Can't use special characters easily

activity = full_activity.join(activity_summary)


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


# drop activity for people who fail basic QC
okay = (activity_summary['quality-goodCalibration'].astype(bool)
            & (~activity_summary['quality-daylightSavingsCrossover'].astype(bool))
            & (activity_summary['quality-goodWearTime'].astype(bool))
       )
activity = activity[okay]
activity.columns = activity.columns.str.replace("-","_") # Can't use special characters easily
activity_variables = activity_variables.str.replace("-","_")
print(f"Dropping {(~okay).sum()} entries out of {len(okay)} due to bad quality or wear-time")

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
    m = 1;
    for i, r in enumerate(sort_order[::-1]):
        if numpy.isfinite(adjusted[r]):
            m = min(adjusted[r], m)
            adjusted[r] = m

    return adjusted # the q-values

# ## Load the ICD10/9 code data
icd10_entries_all = pandas.read_csv("../processed/ukbb_icd10_entries.txt", sep="\t")
# Select our cohort from all the entries
icd10_entries = icd10_entries_all[icd10_entries_all.ID.isin(selected_ids)].copy()
icd10_entries.rename(columns={"ICD10_code": "ICD10"}, inplace=True)

# Load the PheCode mappings
# Downloaded from https://phewascatalog.org/phecodes_icd10
# Has columns:
# ICD10 | PHECODE | Exl. Phecodes | Excl. Phenotypes
phecode_info = pandas.read_csv("../phecode_definitions1.2.csv", index_col=0)
phecode_map = pandas.read_csv("../Phecode_map_v1_2_icd10_beta.csv")
phecode_map.set_index(phecode_map.ICD10.str.replace(".",""), inplace=True) # Remove '.' to match UKBB-style ICD10 codes

### and the ICD9 data
icd9_entries_all = pandas.read_csv("../processed/ukbb_icd9_entries.txt", sep="\t")
# Select our cohort from all the entries
icd9_entries = icd9_entries_all[icd9_entries_all.ID.isin(selected_ids)].copy()
icd9_entries.rename(columns={"ICD9_code": "ICD9"}, inplace=True)

# and convert to phecodes
# v1.2 Downloaded from https://phewascatalog.org/phecodes
phecode_map_icd9 = pandas.read_csv("../phecode_icd9_map_unrolled.csv")
phecode_map_icd9.rename(columns={"icd9":"ICD9", "phecode":"PHECODE"}, inplace=True)
phecode_map_icd9.set_index( phecode_map_icd9['ICD9'].str.replace(".",""), inplace=True) # Remove dots to match UKBB-style ICD9s


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


# ### Display which sources the cases come from for the top codes


phecode_counts = pandas.DataFrame({"counts": phecode_data.sum(axis=0)})
for name, d in {"icd10": phecode_data_icd10, "icd9": phecode_data_icd9, "self_report": phecode_data_self_report}.items():
    cases = d.reset_index().groupby(by="ID").any()
    phecode_counts[name + "_cases"] = cases.sum(axis=0)
phecode_counts["meaning"] = phecode_counts.index.map(phecode_info.phenotype)
print("Most frequent phecodes by source")
print(phecode_counts.sort_values(by="counts", ascending=False).head(20))

# Gather phecode diagnosis information for each subject
for group in phecode_groups:
    # Note that those without any ICD10 entries at all should be marked as non-case, hence the fillna()
    data[group] = data.index.map(phecode_data[group].astype(int)).fillna(0)

# Correlate each block-level code with our activity variable
# Loop through all the activity variables and phecode groups we are interested in
phecode_tests_list = []
covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
for group in phecode_groups:
    N = data[group].sum()
    if N < 50:
        print(f"Skipping {group} - only {N} cases found")
        continue
    
    for activity_variable in activity.columns:
        fit = smf.ols(f"{activity_variable}~ Q({group}) + sex * ({covariate_formula})",
                     data=data).fit()
        reduced_fit = smf.ols(f"{activity_variable} ~ sex * ({covariate_formula})",
                            data=data).fit()
        f,p,df = fit.compare_f_test(reduced_fit)
        coeff = fit.params[f"Q({group})"]
        std_effect = coeff / data[activity_variable].std()
        phecode_tests_list.append({"group": group,
                                "var": activity_variable,
                                "p": p,
                                "coeff": coeff,
                                "std_effect": std_effect,
                                "N": N,
                               })
phecode_tests = pandas.DataFrame(phecode_tests_list)

phecode_tests['q'] = BH_FDR(phecode_tests.p)
phecode_tests["meaning"] = phecode_tests.group.map(phecode_info.phenotype)
phecode_tests["category"] = phecode_tests.group.map(phecode_info.category)
phecode_tests.index.rename("phecode", inplace=True)

phecode_tests.to_csv(OUTDIR+f"phecodes.txt", sep="\t")

# Summarize the phecode test results
num_nonnull = len(phecode_tests) - phecode_tests.p.sum()*2
bonferonni_cutoff = 0.05 / len(phecode_tests)
FDR_cutoff = phecode_tests[phecode_tests.q < 0.05].p.max()
print(f"Of {len(phecode_tests)} tested, approx {int(num_nonnull)} expected non-null")
print(f"and {(phecode_tests.p <= bonferonni_cutoff).sum()} exceed the Bonferonni significance threshold")
print(f"and {(phecode_tests.p <= FDR_cutoff).sum()} exceed the FDR < 0.05 significance threshold")

fig, ax = pylab.subplots()

ax.scatter(phecode_tests.N, -numpy.log10(phecode_tests.p), marker=".")
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
    ps = phecode_tests[phecode_tests['var'] == activity_variable].p
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
phecode_categories = phecode_tests.category.unique()
for i, category in enumerate(phecode_categories):
    ps = phecode_tests[phecode_tests.category == category].p
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.random(ps.shape)-0.5) * 0.7,
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
ax.scatter(phecode_tests['var'].map(activity_variance.normalized),
            -numpy.log10(phecode_tests.p))
ax.set_xlabel("Ratio of intra- to inter-personal variance")
ax.set_ylabel("-log10(p-value)")
ax.set_title("p-values by variance ratio")
fig.savefig(OUTDIR+"pvalues_by_variance.png")

## Display effect sizes by the inter-intra personal variance ratio
fig, ax = pylab.subplots(figsize=(8,8))
ax.scatter(phecode_tests['var'].map(activity_variance.normalized),
            phecode_tests.std_effect.abs())
ax.set_xlabel("Ratio of intra- to inter-personal variance")
ax.set_ylabel("Standardized Effect Size")
ax.set_title("Effect sizes by variance ratio")
fig.savefig(OUTDIR+"effect_size_by_variance.png")

## heatmap of phenotype-activity relationships
fig, ax = pylab.subplots(figsize=(9,9))
FDR_CUTOFF_VALUE = 0.05
phecode_tests['q_significant'] = (phecode_tests.q < FDR_CUTOFF_VALUE).astype(int)
pvalue_counts = phecode_tests.groupby(["var", "category"]).q_significant.sum().unstack()
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
pvalue_percent = phecode_tests.groupby(["var", "category"]).q_significant.mean().unstack()*100
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
total_significant = phecode_tests.groupby(["var"]).q_significant.sum()
num_tests = phecode_tests.group.nunique()
#category_sizes = phecode_tests.groupby(['category']).group.nunique().

def hypergeom_enrichment(data):
    var = data['var'].iloc[0]
    k = data.q_significant.sum()
    M = num_tests
    n = total_significant[var]
    N = len(data)
    p =  scipy.stats.hypergeom.sf(k, M, n, N)
    if n == 0:
        return 1
    return p
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_enrichment_stacked = phecode_tests.groupby(["var", "category"])[['group', 'q_significant', 'var']].apply(hypergeom_enrichment)
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


## PCA of the different phenotypes
# each point is a phenotype and its given the vector of effect sizes relating to the different associations
# is there a pattern/clustering to the phenotypes?
phecode_effect_vectors = phecode_tests.set_index(["group", "var"]).std_effect.unstack()
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(phecode_effect_vectors)
phecode_pca = pandas.DataFrame({0: pca_coords[:,0], 1:pca_coords[:,1]}, index=phecode_effect_vectors.index)
phecode_pca['category'] = phecode_pca.index.map(phecode_info.category)

fig, ax = pylab.subplots(figsize=(8,8))
for category in phecode_info.category.unique():
    category_points = phecode_pca.loc[phecode_pca.category == category, [0,1]]
    ax.scatter(category_points[0], category_points[1], label=category)
ax.legend()
ax.set_title("PCA of Phenotypes by Activity Effect Sizes")
fig.savefig(OUTDIR+"phecode_pca.png")

## PCA of the different activity variables
activity_effect_vectors = phecode_tests.set_index(["var", "group"]).std_effect.unstack()
pca_coords = pca.fit_transform(activity_effect_vectors)
activity_pca = pandas.DataFrame({0: pca_coords[:,0], 1:pca_coords[:,1]}, index=activity_effect_vectors.index)
fig, ax = pylab.subplots(figsize=(8,8))
ax.scatter(activity_pca[0], activity_pca[1])
for i, var in enumerate(activity_effect_vectors.index):
    ax.annotate(var, (activity_pca.loc[var, 0], activity_pca.loc[var,1]))
ax.legend()
ax.set_title("PCA of Activity Variables by Phenotype Effect Sizes")
fig.savefig(OUTDIR+"activity_variable_pca.png")


## Connection analysis
# We say there is a 'connection' between two phenotypes if there is an activity variable
# that associates with both of them

# count the number of connections for each phenotype-pairing
significance_matrix = phecode_tests.set_index(["var", "group"]).q_significant.unstack()
connections_phecodes = significance_matrix.T @ significance_matrix
connections_activity = significance_matrix @ significance_matrix.T
def plot_heatmap(data, order=True, label=""):
    fig, ax = pylab.subplots()
    if order:
        dist = scipy.spatial.distance.squareform(1/(data+1), checks=False)
        linkage = scipy.cluster.hierarchy.linkage(dist, optimal_ordering=True)
        ordering = scipy.cluster.hierarchy.leaves_list(linkage)
    else:
        ordering = numpy.aranage(len(data.index))

    ax.imshow(data.iloc[ordering, ordering])

    if "x" in label:
        ax.set_xticks(numpy.arange(len(data.columns)))
        ax.set_xticklabels(data.columns[ordering])
    if "y" in label:
        ax.set_yticks(numpy.arange(len(data.index)))
        ax.set_yticklabels(data.index[ordering])
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
        fit = smf.ols(f"{activity_variable} ~ 0 + C(sex, Treatment(reference=-1)) : ({sex_covariate_formula} +  Q({group}))",
                         data=data).fit()


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
            "group": group,
            "var": activity_variable,
            "male_coeff": float(male_coeff) / male_std,
            "female_coeff": float(female_coeff) /  female_std,
            "p_male": float(p_male),
            "p_female": float(p_female),
            "p_diff": float(p_diff),
            "N_male": N_male,
            "N_female": N_female,
            "male_coeff_low": float(male_conf_int[0]) / male_std,
            "male_coeff_high": float(male_conf_int[1]) / male_std,
            "female_coeff_low": float(female_conf_int[0]) / female_std,
            "female_coeff_high": float(female_conf_int[1]) / female_std,
        })

phecode_tests_by_sex = pandas.DataFrame(phecode_tests_by_sex_list)

phecode_tests_by_sex["meaning"] = phecode_tests_by_sex.group.map(phecode_info.phenotype)
phecode_tests_by_sex["category"] = phecode_tests_by_sex.group.map(phecode_info.category)
phecode_tests_by_sex = phecode_tests_by_sex.join(phecode_tests.q, how="left")
phecode_tests_by_sex['q_diff'] = BH_FDR(phecode_tests_by_sex.p_diff)
phecode_tests_by_sex.sort_values(by="p_diff", inplace=True)

phecode_tests_by_sex.to_csv(OUTDIR+"/all_phenotypes.by_sex.txt", sep="\t")


### Generate summaries of the phecode test by-sex results

## Display the p-values of each actiivty variable
fig, ax = pylab.subplots(figsize=(8,8))
for i, activity_variable in enumerate(activity_variables):
    ps = phecode_tests_by_sex[phecode_tests_by_sex['var'] == activity_variable].p_diff
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.random(ps.shape)-0.5) * 0.7,
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
phecode_categories = phecode_tests_by_sex.category.unique()
for i, category in enumerate(phecode_categories):
    ps = phecode_tests_by_sex[phecode_tests_by_sex.category == category].p_diff
    ax.scatter(-numpy.log10(ps),
                numpy.ones(ps.shape)*i + (numpy.random.random(ps.shape)-0.5) * 0.7,
                marker=".", s=1.5)
ax.set_xlabel("-log10(p-value)")
ax.set_title("Sex differences\ngrouped by phecode category")
ax.set_yticks(range(len(phecode_categories)))
ax.set_yticklabels(phecode_categories)
ax.set_ylim(-1, len(phecode_categories))
fig.tight_layout()
fig.savefig(OUTDIR+"pvalues_by_phecode_category.by_sex.png")

# Plot the regression coefficients for each of the phenotypes
fig, ax = pylab.subplots(figsize=(9,9))
num_male = (data.sex == "Male").sum()
num_female = (data.sex == "Female").sum()

d = phecode_tests_by_sex[(phecode_tests_by_sex.q < FDR_CUTOFF_VALUE)
                        & (phecode_tests_by_sex.N_male > 5)
                        & (phecode_tests_by_sex.N_female > 5)]
x = d["male_coeff"] # / (d["N_male"] / num_male)
y = d["female_coeff"] # / (d["N_female"] / num_female)
xerr = (d["male_coeff_high"] - d["male_coeff_low"])/2 #/ (d.N_male / num_male)
yerr = (d["female_coeff_high"] - d["female_coeff_low"])/2 #/ (d.N_female / num_female)
# The points
#ax.errorbar(x = x,
#            y =y,
#            xerr = xerr,
#            yerr = yerr,
#            fmt = "o",
#            label = "phenotypes")
ax.scatter(numpy.abs(x), numpy.abs(y), label="phenotypes")

# Diagonal y=x line
diag = numpy.array([ numpy.min([ax.get_xlim(), ax.get_ylim()]),
                    numpy.max([ax.get_xlim(), ax.get_ylim()]) ])
ax.plot(diag, diag, c='k', zorder=-1, label="diagonal")

# The regression line through the points
# Linear Deming/Orthogonal-distance regression since error in both variables
def deming(x, y, xerr, yerr):
    from scipy.odr import ODR, RealData, Model
    d = RealData(x,y, sy=yerr, sx=xerr)
    def linfit(args,x):
        return args[0] + args[1]*x
    est = [0,1]
    m = Model(linfit)
    odr = ODR(d, m, beta0=est).run()
    odr.pprint()
    return odr
odr = deming(x, y, xerr, yerr)
intercept, coeff = odr.beta
ax.plot(diag, diag * coeff + intercept, label="regression")

ax.set_title("Effect sizes by sex\nAmong signifcant associations")
ax.set_xlabel("Effect size in males (absolute value)")
ax.set_ylabel("Effect size in females (absolute value)")
ax.set_aspect("equal")
ax.legend()

fig.savefig(f"{OUTDIR}/sex_differences.all_phenotypes.png")


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

selected = (phecode_tests_by_sex.q < FDR_CUTOFF_VALUE)
d = phecode_tests_by_sex[selected]
ax.scatter(numpy.log10(d.N_male),
            d.male_coeff.abs(),
            c="r", label="Male", marker='.')
ax.scatter(numpy.log10(d.N_female),
            d.female_coeff.abs(),
            c="b", label="Female", marker='.')

for i in range(1):
    if i > 0:
        d = phecode_tests_by_sex[selected].sample(len(phecode_tests_by_sex),replace=True)
    else:
        d = phecode_tests_by_sex[selected]
    
    male_smooth = sm.nonparametric.lowess(
                        d.male_coeff.abs(),
                        numpy.log10(d.N_male),
                        return_sorted=True,
                        frac=0.6,
                        it=0)
    ax.plot(male_smooth[:,0], male_smooth[:,1], c="r", alpha=1, linewidth=5)
    female_smooth = sm.nonparametric.lowess(
                        d.female_coeff.abs(),
                        numpy.log10(d.N_female),
                        return_sorted=True,
                        frac=0.6,
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
num_significant_male = phecode_tests_by_sex.groupby(["var", "category"]).significant_male.sum()
num_significant_female = phecode_tests_by_sex.groupby(["var", "category"]).significant_female.sum()
num_significant_either = phecode_tests_by_sex.groupby(["var", "category"]).significant_either.sum()

#TODO: is there a meaningful way to test for male/female enrichment by category?
#male_enriched = [scipy.stats.hypergeom(M, n, num_significant_male

# ### Check the overall average of effect size by sex of the RA-phenotype associations

male_weights = 1 / (phecode_tests_by_sex.male_coeff_high - phecode_tests_by_sex.male_coeff_low)**2 * (phecode_tests_by_sex.male_coeff != 0.0)
female_weights = 1 / (phecode_tests_by_sex.female_coeff_high - phecode_tests_by_sex.female_coeff_low)**2 * (phecode_tests_by_sex.female_coeff != 0.0)
rel_male_coeff = numpy.abs(phecode_tests_by_sex.male_coeff  * male_weights)
rel_female_coeff = numpy.abs(phecode_tests_by_sex.female_coeff * female_weights)

print(f"Weighted mean male effect:   {rel_male_coeff.mean() / male_weights.mean():0.4f}")
print(f"Median male effect:          {phecode_tests_by_sex.male_coeff.abs().median():0.4f}")
print(f"Weighted mean female effect: {rel_female_coeff.mean() / male_weights.mean():0.4f}")
print(f"Median female effect:        {phecode_tests_by_sex.female_coeff.abs().median():0.4f}")
#print(f"Note: effects are the difference in mean RA values between cases and controls of the phenotype.")
#print(f"   standard deviation of RA:  {data.acceleration_RA.std():0.4f}")

## Heatmap of sex-difference signifiances
fig, ax = pylab.subplots(figsize=(9,9))
FDR_CUTOFF_VALUE = 0.05
phecode_tests_by_sex['q_significant'] = (phecode_tests_by_sex.q_diff < FDR_CUTOFF_VALUE).astype(int)
pvalue_counts = phecode_tests_by_sex.groupby(["var", "category"]).q_significant.sum().unstack()
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
pvalue_percent = phecode_tests_by_sex.groupby(["var", "category"]).q_significant.mean().unstack()*100
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
total_significant = phecode_tests_by_sex.groupby(["var"]).q_significant.sum()
num_tests = phecode_tests_by_sex.group.nunique()
fig, ax = pylab.subplots(figsize=(9,9))
pvalue_enrichment_stacked = phecode_tests_by_sex.groupby(["var", "category"])[['group', 'q_significant', 'var']].apply(hypergeom_enrichment)
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
phecode_effect_vectors_male = phecode_tests_by_sex.set_index(["group", "var"])['male_coeff'].unstack()
phecode_effect_vectors_female = phecode_tests_by_sex.set_index(["group", "var"])['female_coeff'].unstack()
pca = PCA(n_components=2)
pca.fit(pandas.concat([phecode_effect_vectors_male, phecode_effect_vectors_female]))
pca_coords_male = pca.transform(phecode_effect_vectors_male)
pca_coords_female = pca.transform(phecode_effect_vectors_female)
phecode_pca_male = pandas.DataFrame({0: pca_coords_male[:,0], 1: pca_coords_male[:,1]},
                                        index=phecode_effect_vectors_male.index)
phecode_pca_female = pandas.DataFrame({0: pca_coords_female[:,0], 1: pca_coords_female[:,1]},
                                        index=phecode_effect_vectors_female.index)
fig, ax = pylab.subplots(figsize=(8,8))
for var in phecode_pca_male.index:
    ax.plot([phecode_pca_male.loc[var,0], phecode_pca_female.loc[var,0]],
            [phecode_pca_male.loc[var,1], phecode_pca_female.loc[var,1]],
            c="k",
            zorder=-1)
ax.scatter(phecode_pca_male[0], phecode_pca_male[1], label="male")
ax.scatter(phecode_pca_female[0], phecode_pca_female[1], label="female")
ax.legend()
ax.set_title("PCA of Phenotypes by Activity Effect Sizes by Sex")
fig.savefig(OUTDIR+"phecode_pca.by_sex.png")


### Associate with quantitiative traits
# Quantitative traits:
import fields_of_interest
quantitative_blocks = [
    fields_of_interest.blood_fields,
    fields_of_interest.urine,
    fields_of_interest.arterial_stiffness,
    fields_of_interest.physical_measures,
]
quantitative_vars = [c for block in quantitative_blocks
                        for c in block
                        if (c in data.columns) and (pandas.api.types.is_numeric_dtype(data[c].dtype))]

quantitative_tests_list = []
covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
for phenotype in quantitative_vars:
    N = data[phenotype].count()
    if N < 50:
        print(f"Skipping {phenotype} - only {N} cases found")
        continue
    
    for activity_variable in activity.columns:
        fit = smf.ols(f"{phenotype} ~ {activity_variable} + sex * ({covariate_formula})",
                     data=data).fit()
        reduced_fit = smf.ols(f"{phenotype} ~ sex * ({covariate_formula})",
                            data=data).fit()
        f,p,df = fit.compare_f_test(reduced_fit)
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
quantitative_tests['ukbb_field'] = quantitative_tests.phenotype.map(fields_of_interest.all_fields)
quantitative_tests.to_csv(OUTDIR+"/quantitative_traits.txt", sep="\t")
