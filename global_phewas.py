#!/usr/bin/env python
# # PheWAS analysis
# 
# Check if there are associations of phenotypes with circadian problems,
# particularly for unusual timing or for lack of consistent rhythm.

import re

import scipy
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf
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


## Select the activity variables that have between-person variance greater than their within-person variance
# and for the summary variables, use only those that are overall summary variables
activity_variance = pandas.read_csv("../processed/inter_intra_personal_variance.txt", sep="\t", index_col=0)
activity_variance['summary_var'] = activity_variance.index.isin(activity_summary.columns)
activity_variance['use'] = (~activity_variance.summary_var) | activity_variance.index.str.contains("overall-")
good_variance = (activity_variance.normalized < 1)
activity_variables = activity_variance.index[good_variance & activity_variance.use]

activity = full_activity.join(activity_summary)
activity = activity[activity.columns[activity.columns.isin(activity_variables)]]


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
print(f"Of {len(phecode_tests)} tested, approx {int(num_nonnull)} expected non-null")
print(f"and {(phecode_tests.p < bonferonni_cutoff).sum()} exceed the Bonferonni significance threshold")

fig, ax = pylab.subplots()

ax.scatter(phecode_tests.N, -numpy.log10(phecode_tests.p), marker=".")
ax.set_xlabel("Number cases")
ax.set_ylabel("-log10(p-value)")
ax.axhline( -numpy.log10(0.05/len(phecode_tests)), c="k", zorder = -1 )
ax.set_title("PheCode - Activity associations")
fig.savefig(OUTDIR+"phewas_summary.png")


fig, ax = pylab.subplots()

ax.scatter(phecode_tests.std_effect, -numpy.log10(phecode_tests.p), marker=".")
ax.set_xlabel("Effect size")
ax.set_ylabel("-log10(p-value)")
ax.axhline( -numpy.log10(0.05/len(phecode_tests)), c="k", zorder = -1 )
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
            phecode_tests.std_effect)
ax.set_xlabel("Ratio of intra- to inter-personal variance")
ax.set_ylabel("Standardized Effect Size")
ax.set_title("Effect sizes by variance ratio")
fig.savefig(OUTDIR+"effect_size_by_variance.png")



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

d = phecode_tests_by_sex[(phecode_tests_by_sex.q < 0.01)
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

ax.set_title("Effect sizes by sex")
ax.set_xlabel("Effect size in males")
ax.set_ylabel("Effect size in females")
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

fig, ax = pylab.subplots( figsize=(9,9) )

ax.scatter(numpy.log10(phecode_tests_by_sex.N_male),
            phecode_tests_by_sex.male_coeff.abs(),
            c="r", label="Male", marker='.')
ax.scatter(numpy.log10(phecode_tests_by_sex.N_female),
            phecode_tests_by_sex.female_coeff.abs(),
            c="b", label="Female", marker='.')

for i in range(1):
    if i > 0:
        d = phecode_tests_by_sex.sample(len(phecode_tests_by_sex),replace=True)
    else:
        d = phecode_tests_by_sex
    
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



# TODO: rest of script!
exit()

























# Plot phecode incidence versus a variable
MALE_COLOR = "b"
FEMALE_COLOR = "r"
def plot_phecode_incidence(code, normalize_sexes=False, yscale="all"):
    # if yscale = 'auto' then it will fit to the data, if 'all' then it uses 0-1 as range
    fig, ax = pylab.subplots()
    male_max = None
    male_min = None
    female_max = None
    female_min = None
    eval_x = numpy.linspace(0.5,1.0, 21)
    
    for i in range(20):
        d = data.sample(frac=1, replace=True)
        d_male = d[d.sex == "Male"]
        d_female = d[d.sex == "Female"]

        if normalize_sexes:    
            # Downsample to equal percent of cases in both male and female
            # so that the differences between sexes are more visible
            p_male = d_male[code].sum() / len(d_male)
            p_female = d_female[code].sum() / len(d_female)
            down_to = min(p_male, p_female)
            if p_male > p_female:
                to_keep = numpy.random.choice(numpy.where(d_male[code].astype(bool))[0],
                                              replace=False,
                                              size=int(down_to*len(d_male)))
                control = numpy.where(~d_male[code].astype(bool))[0]
                d_male = d_male.iloc[numpy.concatenate((to_keep, control))]
            else:
                to_keep = numpy.random.choice(numpy.where(d_female[code].astype(bool))[0],
                                              replace=False,
                                              size=int(down_to*len(d_female)))
                control = numpy.where(~d_female[code].astype(bool))[0]
                d_female = d_female.iloc[numpy.concatenate((to_keep, control))]

        s = local_regression(d_male['acceleration_RA'], d_male[code], eval_x)

        if male_max is None:
            male_max = s
        else:
            male_max = numpy.maximum(male_max, s)
        if male_min is None:
            male_min = s
        else:
            male_min = numpy.minimum(male_min, s)
            
        s = local_regression(d_female['acceleration_RA'], d_female[code], eval_x)

        if female_max is None:
            female_max = s
        else:
            female_max = numpy.maximum(female_max, s)
        if female_min is None:
            female_min = s
        else:
            female_min = numpy.minimum(female_min, s)
            
    # Draw the confidence interval
    ax.fill_between(eval_x, male_min, male_max, color=MALE_COLOR, alpha=0.5)
    ax.fill_between(eval_x, female_min, female_max, color=FEMALE_COLOR, alpha=0.5)
            
    d_male = data[data.sex == "Male"]
    s_male = local_regression(d_male['acceleration_RA'], d_male[code], eval_x)

                
    ax.plot(eval_x, s_male, c=MALE_COLOR)
    
    d_female = data[data.sex == "Female"]
    s_female = local_regression(d_female['acceleration_RA'], d_female[code], eval_x)

    ax.plot(eval_x, s_female, c=FEMALE_COLOR)

    if yscale == 'all':
        ax.set_ylim(0,1)
    elif yscale == "auto":
        if max(numpy.max(s_male), numpy.max(s_female)) > 0.1:
            # If prevelant enough, show the whole range
            # but some are too rare too see, so we use the default zoom
            ax.set_ylim(top=1)
        ax.set_ylim(bottom=0)
    ax.set_ylabel(f"Incidence rate")
    ax.set_xlabel("RA")
    ax.legend(handles=[mpatches.Patch(facecolor=MALE_COLOR, label="Male"),
                       mpatches.Patch(facecolor=FEMALE_COLOR, label="Female")])
    ax.set_title(phecode_tests.meaning[code])
    return fig


# In[33]:


fig = plot_phecode_incidence(495, normalize_sexes = False)


# In[34]:


# Generate and save plots for a variety of phecodes
phecodes = [480, 495] + list(phecode_tests.sort_values(by="p").index[:20])
pylab.ioff()
for phecode in phecodes:
    fig = plot_phecode_incidence(phecode, yscale="auto")
    fig.savefig(OUTDIR + f"{phecode}.incidence.png")
    pylab.close()
pylab.ion()


# In[35]:


# Compare RATIO of the prevelance of the two sexes within cases, as a function of RA score
group = 563
fig, ax = pylab.subplots()
for i in range(20):
    d = data[data[group] == 1].sample(frac=1, replace=True)

    s = sm.nonparametric.lowess(d.sex == "Male",
                                  d['acceleration_RA'],
                                  return_sorted=True,
                                   frac=0.8,
                                  delta=0.005,
                                  it=0)
    ax.plot(s[:,0], s[:,1], c="k")
ax.set_ylim(0,1)
ax.set_ylabel(f"Fraction male within those with\n{phecode_tests.meaning[group]}")
ax.set_xlabel("RA")
ax.set_title(phecode_tests.meaning[group])


# ## Test age interactions with the RA-phenotype association

# In[36]:


# Correlate each block-level code with our activity variable within age buckets
age_buckets = range(int(data.birth_year.min()), int(data.birth_year.max()+1), 10)
def to_age_bucket(birth_year):
    i = numpy.searchsorted(age_buckets, birth_year)
    return f"{age_buckets[i-1]}-{age_buckets[i-1]+10}"
data['age_bucket'] = data.birth_year.map(to_age_bucket)

activity_variable = "acceleration_RA"

phecode_tests_by_age = {}

covariate_formula = ' + '.join(c for c in covariates if c != 'sex')


for group in phecode_groups:  
    N = data[group].sum()
    if N <= 0:
        print(f"Skipping {group} - only {N} cases found")
        continue
        
    if phecode_tests.loc[group, "q"] > 0.01:
        # Skip test, not significant
        #print(f"Skipping {group} since q > 0.01")
        continue
        
    fit = smf.ols(f"acceleration_RA ~ 0 + (age_bucket) : (sex +  Q({group})) + BMI",
                     data=data).fit()
    restricted_fit = smf.ols(f"acceleration_RA ~ 0 + (age_bucket + sex +  Q({group}))  + BMI",
                     data=data).fit()

    f,p,df = fit.compare_f_test(restricted_fit)
    conf_ints = fit.conf_int()
    #male_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
    #female_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']
    
    phecode_tests_by_age[group] = {
        "N": N,
        "p": p,
    }

phecode_tests_by_age = pandas.DataFrame(phecode_tests_by_age).T


# In[37]:


phecode_tests_by_age['q'] = BH_FDR(phecode_tests_by_age.p)
phecode_tests_by_age["meaning"] = phecode_tests_by_age.index.map(phecode_info.phenotype)
phecode_tests_by_age["category"] = phecode_tests_by_age.index.map(phecode_info.category)
print("Age-effect of RA-phenotype interactions")
phecode_tests_by_age.sort_values("p")


# In[38]:


# Phecode incidence plots by age buckets
decades = range(int(data.birth_year.min()), int(data.birth_year.max()+1), 10)
def plot_phecode_incidence_by_age(data, code, yscale="all", ax = None,  c="k"):
    # if yscale = 'auto' then it will fit to the data, if 'all' then it uses 0-1 as range
    if not ax:
        fig, ax = pylab.subplots()
    eval_x = numpy.linspace(0.5,1.0, 21)
    
    peak = 0
    values = numpy.linspace(0.3, 1.0, len(decades))
    for i, decade in enumerate(decades):
        in_decade = (data.birth_year >= decade) & (data.birth_year < decade + 10)
        d = data[in_decade]
        
        smoothed = local_regression(d['acceleration_RA'], d[code], eval_x, bw=0.1)
        
        ax.plot(eval_x, smoothed, c = c, alpha=values[i])
        peak = max(numpy.max(smoothed), peak)
        
    if yscale == 'all':
        ax.set_ylim(0,1)
    elif yscale == "auto":
        if peak > 0.1:
            # If prevelant enough, show the whole range
            # but some are too rare too see, so we use the default zoom
            ax.set_ylim(top=1)
        ax.set_ylim(bottom=0)
        
    ax.set_ylabel(f"Incidence rate")
    ax.set_xlabel("RA")
    ax.legend(handles=[mpatches.Patch(facecolor='k', alpha=val, label=f"{decade}-{decade+9}")
                       for val,decade in zip(values, decades)])
    ax.set_title(phecode_tests.meaning[code])
    return ax.get_figure()


# In[39]:


fig = plot_phecode_incidence_by_age(data, 401, yscale='auto')


# In[40]:


fig = plot_phecode_incidence_by_age(data[data['sex'] == "Male"], 401, yscale='auto', c="b")
plot_phecode_incidence_by_age(data[data['sex'] == "Female"], 401, yscale='auto', ax = fig.gca(), c="r")
fig.savefig(OUTDIR+"401.incidence.by_age_and_sex.png")


# In[41]:


# Generate and save by-age plots for a variety of phecodes
phecodes = [480, 495] + list(phecode_tests.sort_values(by="p").index[:20])
pylab.ioff()
for phecode in phecodes:
    fig = plot_phecode_incidence_by_age(data, phecode, yscale="auto")
    fig.savefig(OUTDIR + f"{phecode}.incidence.by_age.png")
    pylab.close()
pylab.ion()


# In[62]:


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

    fig, (ax1,ax2,ax3) = pylab.subplots(nrows=3, sharex=True,
                                        gridspec_kw = {"hspace":0.1,
                                                       "height_ratios":[0.2,0.6,0.2]})

    # Plot the data
    ax1.fill_between(eval_x, 0, control_density, color=CONTROL_COLOR)
    if confidence_interval:
        ax2.fill_between(eval_x, 0, lower_bound, color=CASE_COLOR)
        ax2.fill_between(eval_x, lower_bound, middle, color=CASE_COLOR, alpha=0.5)
        ax2.fill_between(eval_x, middle, upper_bound, color=CONTROL_COLOR, alpha=0.5)
        ax2.fill_between(eval_x, upper_bound, 1, color=CONTROL_COLOR)
        #ax2.fill_between(eval_x, lower_bound, upper_bound, color=UNCERTAIN_COLOR)
    else:
        ax2.fill_between(eval_x, 0, incidence, color=CASE_COLOR)
        ax2.fill_between(eval_x, incidence, 1, color=CONTROL_COLOR)
    ax3.fill_between(eval_x, 0, case_density, color=CASE_COLOR)

    # Plot avgs
    ax1.axvline(control_avg, c='k', linestyle="--")
    ax3.axvline(case_avg, c='k', linestyle="--")
    ax2.axhline(total_incidence, c='k', linestyle="--")

    # Label plot
    ax1.set_ylabel(f"controls\nN={(~case).sum()}")
    ax2.set_ylabel(f"ratio\n{total_incidence:0.1%}")
    ax3.set_ylabel(f"cases\nN={case.sum()}") 
    ax3.set_xlabel("RA")

    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks([])
    ax1.yaxis.set_ticks([])
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks([0,0.25, 0.5, 0.75, 1])
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
        ax1.set_title(phecode_tests.loc[code].meaning + ("\n(normalized)" if normalize else ""))
    except KeyError:
        ax1.set_title(code)
    return fig

fig = fancy_case_control_plot(data, 401, normalize=True, confidence_interval=True)


# In[43]:


fig.savefig(OUTDIR+"test.401.6.png")


# In[44]:


phe306_codes = phecode_map[phecode_map.PHECODE == 306.00].index
for code in phe306_codes:
    print(code, (icd10_entries.ICD10 == code).sum())


# In[45]:


# Generate and save some of the 'fancy' plots
pylab.ioff()
phecodes = [480, 495] + list(phecode_tests.sort_values(by="p").index[:20])
for code in  phecodes:
    if data[code].sum() < 20:
        continue
    fig = fancy_case_control_plot(data, code, confidence_interval=True, normalize=True)
    fig.savefig(OUTDIR+f"{code}.fancy.plot.png")
    pylab.close()
pylab.ion()


# # Hypertension analysis

# In[46]:


ukbb.columns[ukbb.columns.str.contains('rate')]


# In[47]:


# Average all the BP readings
data['systolic_bp'] = ukbb[['systolic_blood_pressure_V1',
                                        'systolic_blood_pressure_V0',
                                        'systolic_blood_pressure_manual_V0',
                                        'systolic_blood_pressure_manual_V1'
                                        ]].mean(axis=1)
data['diastolic_bp'] = ukbb[['diastolic_blood_pressure_V1',
                            'diastolic_blood_pressure_V0',
                            'diastolic_blood_pressure_manual_V0',
                            'diastolic_blood_pressure_manual_V1']].mean(axis=1)
data['pulse_rate'] = ukbb[[
                                    'pulse_rate_V0',
                                    'pulse_rate_V1'
                                    ]].mean(axis=1)
#data = data.join(ukbb[['systolic_bp', 'diastolic_bp']])


# In[82]:


# Plot of the BP-RA relation
def weighted_quantile(series, weights, quantile):
    # Approximate weighted quantile
    ordering = numpy.argsort(series.values)
    cum_weights = numpy.cumsum(weights[ordering]) / weights.sum()
    q_idx = numpy.searchsorted(cum_weights, quantile)
    if q_idx >= len(ordering):
        # A 1.0 quantile is always technically past the end
        # so just give it the last value
        q_idx = len(ordering) -1 
    return series.iloc[ordering[q_idx]]

def plot_quantitative_var(data, var, ax = None,  c="k", nbins=5, scatter=False):
    quantiles = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    
    xmin = 0.5

    if not ax:
        fig, ax = pylab.subplots()
    eval_x = numpy.linspace(xmin, 1.0, 21)
    
    
    if scatter:
        ax.scatter(data['acceleration_RA'].values, data[var].values, s = 1, marker='.')
    
    alpha_values = numpy.linspace(0.3, 1.0, len(quantiles))
    for i, quantile in enumerate(quantiles):
        smoothed = numpy.empty(len(eval_x))
        for j, x in enumerate(eval_x):
            
            weights = numpy.exp(-(data['acceleration_RA'].values - x)**2 / 0.02)
            smoothed[j] = weighted_quantile(data[var], weights, quantile)

        ax.plot(eval_x, smoothed, c = c, alpha=alpha_values[i])
    
    #ax.axhline(data[var].median(), c="r")
        
    '''
    means = numpy.empty(len(eval_x))
    for j, x in enumerate(eval_x):
        weights = numpy.exp(-(data['acceleration_RA'].values - x)**2 / 0.02)
        means[j] = (data[var] * weights).sum() / weights.sum()
    ax.plot(eval_x, means, c="r")
    '''
        
    ax.set_ylabel(var)
    ax.set_xlabel("RA")
    ax.set_xlim(min(eval_x), max(eval_x))
    ax.legend(handles=[mpatches.Patch(facecolor='k', alpha=val, label=f"{quantile:0.0%}")
                       for val,(quantile) in zip(alpha_values, quantiles)])
    ax.set_title(f"{var} quantiles by RA")
    return ax.get_figure()


# In[83]:


fig = plot_quantitative_var(data, 'systolic_bp', scatter=True)
fig.savefig("systolic_bp.png")


# In[84]:


fig = plot_quantitative_var(data, 'diastolic_bp', scatter=True)
fig.savefig(OUTDIR+"diastolic_bp.png")


# In[51]:


fig = plot_quantitative_var(data, 'pulse_rate')
fig.savefig(OUTDIR+"pulse_rate.png")


# ## Medications

# In[52]:


# Load the medications list
medications = pandas.read_csv("../processed/ukbb_medications.txt", sep="\t", dtype={'medication_code': "int"})
medications.medication_code = medications.medication_code.astype(str)
MEDICATIONS_CODING = 4
med_code_to_name = codings[codings.Coding == MEDICATIONS_CODING].set_index("Value").Meaning
medications['medication'] = pandas.Series(medications.medication_code.map(med_code_to_name), dtype="category")


# In[53]:


# Load the list of hypertension-related drugs
hypertension_medications = pandas.read_csv("../hypertension_medications.txt", sep="\t")
hypertension_medications.Drug = hypertension_medications.Drug.str.lower()


# In[54]:


on_hypertensive_medication = medications.groupby("ID").medication.apply(lambda x: any(x.isin(hypertension_medications.Drug)))


# In[55]:


data['on_hypertensive_medication'] = on_hypertensive_medication
data.on_hypertensive_medication.fillna(False, inplace=True)
data['hypertensive_and_medication'] = (data.on_hypertensive_medication & data[401])
data['hypertensive_no_medication'] = (~data.on_hypertensive_medication) & data[401]


# In[56]:


fig = fancy_case_control_plot(data,
                        'hypertensive_no_medication', confidence_interval=True)
fig.savefig(OUTDIR+"hypertensive_no_medication.png")


# In[57]:


fig = fancy_case_control_plot(data,
                        'hypertensive_and_medication', confidence_interval=True)
fig.savefig(OUTDIR+"hypertensive_and_medication.png")


# In[58]:


fig = fancy_case_control_plot(data[data.on_hypertensive_medication == True],
                        401, confidence_interval=True)


# In[59]:


fig = fancy_case_control_plot(data[data.on_hypertensive_medication == False],
                        401, confidence_interval=True)


# In[60]:


fig = plot_quantitative_var(data[data.on_hypertensive_medication == True],
                        "systolic_bp")
fig.savefig(OUTDIR+"systolic_bp.on_hypertensive_medication.png")


# In[61]:


fig = plot_quantitative_var(data[data.on_hypertensive_medication == False],
                        "systolic_bp")
fig.savefig(OUTDIR+"systolic_bp.no_hypertensive_medication.png")

