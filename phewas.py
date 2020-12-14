import pandas
import numpy
import pylab
import seaborn as sns
import matplotlib
import statsmodels.formula.api as smf


import phewas_preprocess
from phewas_preprocess import self_report_circadian_variables
import phewas_tests
from phewas_tests import covariates, survival_covariates, OLS
import phewas_plots as plots

import util
from util import BH_FDR, legend_of_pointscale, legend_from_colormap, truncate, wrap

def summary():
    # Summarize the phecode test results
    num_nonnull = len(phecode_tests) - phecode_tests.p.sum()*2
    bonferonni_cutoff = 0.05 / len(phecode_tests)
    FDR_cutoff = phecode_tests[phecode_tests.q < 0.05].p.max()
    print(f"Of {len(phecode_tests)} tested, approx {int(num_nonnull)} expected non-null")
    print(f"and {(phecode_tests.p <= bonferonni_cutoff).sum()} exceed the Bonferonni significance threshold")
    print(f"and {(phecode_tests.p <= FDR_cutoff).sum()} exceed the FDR < 0.05 significance threshold")

    ### Create summary plots
    fig, ax = pylab.subplots(figsize=(8,6))
    color = phecode_tests.phecode_category.map(color_by_phecode_cat)
    ax.scatter(phecode_tests.N_cases, -numpy.log10(phecode_tests.p), marker="+", c=color)
    ax.set_xlabel("Number cases")
    ax.set_ylabel("-log10 (p-value)")
    ax.axhline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
    ax.axhline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
    ax.set_title("PheCode - Activity associations")
    fig.savefig(OUTDIR+"phewas_summary.png")


    fig, ax = pylab.subplots(figsize=(8,6))
    pt = phecode_tests.sample(frac=1) # Randomly reorder for the plot
    color = pt.phecode_category.map(color_by_phecode_cat)
    ax.scatter(pt.std_effect, -numpy.log10(pt.p), marker="+", c=color)
    ax.set_xlabel("Effect size")
    ax.set_ylabel("-log10(p-value)")
    ax.axhline( -numpy.log10(bonferonni_cutoff), c="k", zorder = -1 )
    ax.axhline( -numpy.log10(FDR_cutoff), c="k", linestyle="--", zorder = -1 )
    ax.set_title("PheCode - Activity associations")
    legend_from_colormap(ax, color_by_phecode_cat, ncol=2, fontsize="small")
    fig.savefig(OUTDIR+"phewas.volcano_plot.png")

    ### TIMELINE
    # Make a timeline of the study design timing so that readers can easily see when data was collected
    ACTIGRAPHY_COLOR = "#1b998b"
    REPEAT_COLOR = "#c5d86d"
    DIAGNOSIS_COLOR = "#f46036"
    ASSESSMENT_COLOR = "#aaaaaa"
    DEATH_COLOR = "#333333"
    fig, (ax1, ax2, ax3) = pylab.subplots(figsize=(8,6), nrows=3)
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
    bins = pandas.date_range("2000-1-1", "2020-6-1", freq="1M")
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
    ax2.annotate("Medical Record\nDiagnoses", (diagnosis_time.mean(), 1500), ha="center")
    ax3.annotate("Deaths", (death_time.mean(), 20), ha="center")
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

def sex_difference_plots():
    d = phecode_tests_by_sex[True #(phecode_tests_by_sex.q < 0.05 )
                            & (phecode_tests_by_sex.N_male > 300)
                            & (phecode_tests_by_sex.N_female > 300)]
    ## Create the phewas_plots.sex_difference_plots
    fig, ax = phewas_plots.sex_difference_plot(d)
    fig.savefig(f"{OUTDIR}/sex_differences.all_phenotypes.png")

    fig, ax = phewas_plots.sex_difference_plot(d[d.phecode_category == 'circulatory system'], color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/sex_differences.circulatory.png")
    fig, ax = phewas_plots.sex_difference_plot(d[d.phecode_category == 'mental disorders'], color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/sex_differences.mental_disorders.png")
    fig, ax = phewas_plots.sex_difference_plot(d[d.phecode_category == 'endocrine/metabolic'], color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/sex_differences.endocrine.png")
    fig, ax = phewas_plots.sex_difference_plot(d[d.phecode_category == 'infectious diseases'], color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/sex_differences.infections.png")
    fig, ax = phewas_plots.sex_difference_plot(d[d.phecode_category == 'respiratory'], color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/sex_differences.respiratory.png")

    #Make 2x2 grid of quantitative sex differences
    fig, axes = pylab.subplots(ncols=2, nrows=2, figsize=(11,11))
    ij = [[0,0], [0,1], [1,0], [1,1]]
    SUBCATEGORIES = ["circulatory system", "mental disorders", "endocrine/metabolic", "respiratory"]
    for cat, ax, (i,j) in zip(SUBCATEGORIES, axes.flatten(), ij):
        tests = d[d.phecode_category == cat]
        phewas_plots.sex_difference_plot(tests.sample(frac=1), color_by="phecode_meaning", ax=ax, legend=True, labels=False, cmap="tab20_r")
        ax.set_title(cat)
        if j == 0:
            ax.set_ylabel("Effect size in females")
        if i == 1:
            ax.set_xlabel("Effect size in males")
    fig.tight_layout()
    fig.savefig(OUTDIR+"sex_differences.2x2.png")


    #Make 2x2 grid of phecode sex differences
    fig, axes = pylab.subplots(ncols=2, nrows=2, figsize=(11,11))
    ij = [[0,0], [0,1], [1,0], [1,1]]
    SUBCATEGORIES = ["circulatory system", "mental disorders", "endocrine/metabolic", "respiratory"]
    for cat, ax, (i,j) in zip(SUBCATEGORIES, axes.flatten(), ij):
        tests = d[d.phecode_category == cat]
        phewas_plots.sex_difference_plot(tests.sample(frac=1), color_by="phecode_meaning", ax=ax, legend=True, labels=False, cmap="tab20_r")
        ax.set_title(cat)
        if j == 0:
            ax.set_ylabel("Effect size in females")
        if i == 1:
            ax.set_xlabel("Effect size in males")
    fig.tight_layout()
    fig.savefig(OUTDIR+"sex_differences.2x2.png")

    # Run sex-difference and age-difference plots on the quantitative tests
    fig, ax = phewas_plots.sex_difference_plot(quantitative_tests.sample(frac=1), color_by="Functional Category", cmap=color_by_quantitative_function, lim=0.25)
    fig.savefig(OUTDIR+"sex_differences.quantitative.png")

    #Make 2x2 grid of quantitative sex differences
    fig, axes = pylab.subplots(ncols=2, nrows=2, figsize=(11,11))
    ij = [[0,0], [0,1], [1,0], [1,1]]
    SUBCATEGORIES = ["Metabolism", "Lipoprotein Profile", "Cardiovascular Function", "Renal Function"]
    for cat, ax, (i,j) in zip(SUBCATEGORIES, axes.flatten(), ij):
        tests = quantitative_tests[quantitative_tests['Functional Category'] == cat]
        phewas_plots.sex_difference_plot(tests.sample(frac=1), color_by="phenotype", lim=0.25, ax=ax, legend=True, labels=False, cmap="tab20_r")
        ax.set_title(cat)
        if j == 0:
            ax.set_ylabel("Effect size in females")
        if i == 1:
            ax.set_xlabel("Effect size in males")
    fig.tight_layout()
    fig.savefig(OUTDIR+"sex_differences.quantitative.2x2.png")

def age_difference_plots():
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


    fig, ax = phewas_plots.age_effect_plot(d)
    fig.savefig(f"{OUTDIR}/age_effects.png")

    fig, ax = phewas_plots.age_effect_plot(d[d.phecode_category == 'mental disorders'], labels=False, color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/age_effects.mental_disorders.png")
    fig, ax = phewas_plots.age_effect_plot(d[d.phecode_category == 'circulatory system'], labels=False, color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/age_effects.circulatory.png")
    fig, ax = phewas_plots.age_effect_plot(d[d.phecode_category == 'endocrine/metabolic'], labels=False, color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/age_effects.endorcine.png")
    fig, ax = phewas_plots.age_effect_plot(d[d.phecode_category == 'genitourinary'], labels=False, color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/age_effects.genitourinary.png")
    fig, ax = phewas_plots.age_effect_plot(d[d.phecode_category == 'respiratory'], labels=False, color_by="phecode_meaning")
    fig.savefig(f"{OUTDIR}/age_effects.respiratory.png")

    #Make 2x2 grid of age effect plots
    fig, axes = pylab.subplots(ncols=2, nrows=2, figsize=(11,11))
    ij = [[0,0], [0,1], [1,0], [1,1]]
    SUBCATEGORIES = ["circulatory system", "mental disorders", "endocrine/metabolic", "respiratory"]
    for cat, ax, (i,j) in zip(SUBCATEGORIES, axes.flatten(), ij):
        tests = d[d.phecode_category == cat]
        phewas_plots.age_effect_plot(tests.sample(frac=1), color_by="phecode_meaning", ax=ax, legend=True, labels=False, cmap="tab20_r")
        ax.set_title(cat)
        if j == 0:
            ax.set_ylabel("Effect size at 70")
        if i == 1:
            ax.set_xlabel("Effect size at 55")
    fig.tight_layout()
    fig.savefig(OUTDIR+"age_effects.2x2.png")


    ## age effect for quantitative traits
    dage = quantitative_tests.copy()
    dage['age_55_effect'] = dage["std_effect"] + dage['std_age_effect'] * young_offset
    dage['age_75_effect'] = dage["std_effect"] + dage['std_age_effect'] * old_offset
    dage['p_overall'] = dage.p
    dage['p_age'] = dage.age_difference_p
    fig, ax = phewas_plots.age_effect_plot(dage.sample(frac=1), color_by="Functional Category", cmap=color_by_quantitative_function, lim=0.3)
    fig.savefig(f"{OUTDIR}/age_effects.quantitative.png")

    #Make 2x2 grid of quantitative age differences
    fig, axes = pylab.subplots(ncols=2, nrows=2, figsize=(11,11))
    ij = [[0,0], [0,1], [1,0], [1,1]]
    SUBCATEGORIES = ["Metabolism", "Lipoprotein Profile", "Cardiovascular Function", "Renal Function"]
    for cat, ax, (i,j) in zip(SUBCATEGORIES, axes.flatten(), ij):
        tests = dage[dage['Functional Category'] == cat]
        phewas_plots.age_effect_plot(tests.sample(frac=1), color_by="phenotype", lim=0.25, ax=ax, legend=True, labels=False, cmap="tab20_r")
        ax.set_title(cat)
        if j == 0:
            ax.set_ylabel("Effect size at 70")
        if i == 1:
            ax.set_xlabel("Effect size at 55")
    fig.tight_layout()
    fig.savefig(OUTDIR+"age_effects.quantitative.2x2.png")

def fancy_plots():
    # Hypertension
    fig = phewas_plots.fancy_case_control_plot(data, 401, normalize=False, confidence_interval=True, annotate=True)
    fig.savefig(OUTDIR+"phenotypes.hypertension.png")

    fig = phewas_plots.incidence_rate_by_category(data, 401, categories="birth_year_category", confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.hypertension.by_age.png")

    sns.lmplot(x="acceleration_RA", y="cholesterol", data=data, hue="birth_year_category", markers='.')
    pylab.gcf().savefig(OUTDIR+"phenotypes.LDL.png")

    sns.lmplot(x="acceleration_RA", y="hdl_cholesterol", data=data, hue="birth_year_category", markers='.')
    pylab.gcf().savefig(OUTDIR+"phenotypes.HDL.png")

    sns.lmplot(x="acceleration_RA", y="systolic_blood_pressure_V0", data=data, hue="birth_year_category", markers='.')
    pylab.gcf().savefig(OUTDIR+"phenotypes.systolic_blood_pressure.png")

    sns.lmplot(x="acceleration_RA", y="diastolic_blood_pressure_V0", data=data, hue="birth_year_category", markers='.')
    pylab.gcf().savefig(OUTDIR+"phenotypes.diastolic_blood_pressure.png")

    # Diabetes
    fig = phewas_plots.fancy_case_control_plot(data, 250, normalize=False, confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.diabetes.png")

    fig = phewas_plots.incidence_rate_by_category(data, 250, categories="birth_year_category", confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.diabetes.by_age.png")

    sns.lmplot(x="acceleration_RA", y="glycated_heamoglobin", data=data, hue="birth_year_category", markers='.')
    pylab.gcf().savefig(OUTDIR+"phenotypes.glycated_heamoglobin.png")

    percent_diabetes_with_hypertension = (data[401].astype(bool) & data[250].astype(bool)).mean() / data[250].mean()
    print(f"Percentage of participants with diabetes that also have hypertension: {percent_diabetes_with_hypertension:0.2%}")

    # Mood disorders
    #TODO!!
    #fig = phewas_plots.fancy_case_control_plot(data, 250, normalize=False, confidence_interval=True)
    #fig.savefig(OUTDIR+"phenotypes.diabetes.png")

    # Some more phenotypes:
    fig = phewas_plots.fancy_case_control_plot(data, 585, normalize=True, confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.renal_failure.png")
    fig = phewas_plots.fancy_case_control_plot(data, 276, normalize=True, confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.disorders_fuild_electrolyte_etc.png")
    fig = phewas_plots.fancy_case_control_plot(data, 290, normalize=True, confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.delirium_dementia_alzheimers.png")
    fig = phewas_plots.fancy_case_control_plot(data, 332, normalize=True, confidence_interval=True)
    fig.savefig(OUTDIR+"phenotypes.parkinsons.png")

def survival_curves():
    # Survival by RA
    fig = phewas_plots.quintile_survival_plot(data, "acceleration_RA", "RA")
    fig.savefig(OUTDIR+"survival.RA.png")

    # Survival by main_sleep_offset_avg
    fig = phewas_plots.quintile_survival_plot(data, "main_sleep_offset_mean", "Sleep Offset")
    fig.savefig(OUTDIR+"survival.main_sleep_offset_mean.png")

    # Survival by MVPA_overall_avg
    fig = phewas_plots.quintile_survival_plot(data, "MVPA_overall", "MVPA Mean")
    fig.savefig(OUTDIR+"survival.MVPA_overall.png")

    # Survival by MVPA_overall_avg
    fig = phewas_plots.quintile_survival_plot(data, "MVPA_hourly_SD", "MVPA hourly SD")
    fig.savefig(OUTDIR+"survival.MVPA_hourly_SD.png")

    # Survival by acceleration_hourly_SD
    fig = phewas_plots.quintile_survival_plot(data, "acceleration_hourly_SD", "Acceleration Hourly SD")
    fig.savefig(OUTDIR+"survival.acceleration_hourly_SD.png")

    # Survival by phase
    fig, ax = pylab.subplots()
    data['phase_adjusted'] = (data.phase - 8) % 24 + 8
    fig = phewas_plots.quintile_survival_plot(data, "phase_adjusted", "phase")
    fig.savefig(OUTDIR+"survival.phase.png")

    # Survival by RA and sex
    fig, axes = pylab.subplots(ncols=2, sharey=True, sharex=True, figsize=(10,5))
    RA_quintiles = pandas.qcut(data.acceleration_RA, 5)
    for ax, sex in zip(axes, ["Male", "Female"]):
        for quintile, label in list(zip(RA_quintiles.cat.categories, phewas_plots.quintile_labels))[::-1]:
            phewas_plots.survival_curve(data[(data.sex == sex) & (RA_quintiles == quintile)], ax,
                            label=("RA " + label + " Quintile" if sex == "Male" else None))
            ax.set_title(sex)
    ax1, ax2 = axes
    ax1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax1.set_ylabel("Survival Probability")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    fig.legend()
    fig.savefig(OUTDIR+"survival.RA.by_sex.png")

def survival_plots():
    ### Plot survival assocations versus inter/intra personal variance for validation
    fig, ax = pylab.subplots(figsize=(8,6))
    colorby = survival_tests.activity_var.map(activity_variable_descriptions.Category)
    color = colorby.map(color_by_actigraphy_cat)
    def get_variance_ratio(var):
        if var.endswith("_abs_dev"):
            var = var[:-8]
        try:
            return activity_variance.corrected_intra_personal_normalized.loc[var]
        except KeyError:
            print(var)
            return float("NaN")
    variance_ratio = survival_tests.activity_var.apply(get_variance_ratio)
    variance_ratio.index = survival_tests.activity_var
    ax.scatter(#-numpy.log10(survival_tests.p),
                survival_tests['standardized log Hazard Ratio'],
                variance_ratio,
                s=1-numpy.log10(survival_tests.p)*3,
                c=color)
    ax.set_xlabel("Standardized log Hazard Ratio")
    ax.set_ylabel("Within-person variation / Between-person variation")
    ax.set_ylim(0,1)
    ax.axvline(0, c='k')
    for index, row in survival_tests.sort_values(by="p").head(20).iterrows():
        # Label the top points
        ax.annotate(
            row.activity_var,
            (#-numpy.log10(row.p),
            row['standardized log Hazard Ratio'],
            variance_ratio.loc[row.activity_var]),
            xytext=(0,15),
            textcoords="offset pixels",
            arrowprops={'arrowstyle':"->"})
    legend_elts = [matplotlib.lines.Line2D(
                            [0],[0],
                            marker="o", markerfacecolor=c, markersize=10,
                            label=cat if not pandas.isna(cat) else "NA",
                            c=c, lw=0)
                        for cat, c in color_by_actigraphy_cat.items()]
    fig.legend(handles=legend_elts, ncol=2, fontsize="small")
    #fig.tight_layout()
    fig.savefig(OUTDIR+"survival_versus_variation.svg")

    ### Sex-difference survival plot
    def sex_difference_survival_plot(d, color_by="activity_var_category"):
        colormap = {cat:color for cat, color in
                            zip(d[color_by].unique(),
                                [pylab.get_cmap("Set3")(i) for i in range(20)])}
        color = [colormap[c] for c in d[color_by]]
        fig, ax = pylab.subplots(figsize=(9,9))
        # The points
        ax.scatter(
            d.std_male_logHR,
            d.std_female_logHR,
            label="phenotypes",
            #s=-numpy.log10(d.p)*10,
            c=color)
        ax.set_title("Survival associations by sex")
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.axvline(c="k", lw=1)
        ax.axhline(c="k", lw=1)
        ax.set_xlabel("log HR in males")
        ax.set_ylabel("log HR in females")
        ax.set_aspect("equal")
        #ax.set_xlim(-0.5,0.5)
        #ax.set_ylim(-0.5,0.5)
        # Diagonal y=x line
        bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                    numpy.max([ax.get_xlim(), ax.get_ylim()]))
        diag = numpy.array([-bound, bound])
        ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal")
        ax.plot(diag, -diag, linestyle="--", c='k', zorder=-1, label="diagonal")
        bbox = {'facecolor': (1,1,1,0.8), 'edgecolor':(0,0,0,0)}
        #ax.annotate("Male Effect Larger", xy=(0.4,0), ha="center", bbox=bbox, zorder=3)
        #ax.annotate("Male Effect Larger", xy=(-0.4,0), ha="center", bbox=bbox, zorder=3)
        #ax.annotate("Female Effect Larger", xy=(0,0.4), ha="center", bbox=bbox, zorder=3)
        #ax.annotate("Female Effect Larger", xy=(0,-0.25), ha="center", bbox=bbox, zorder=3)
        util.legend_from_colormap(ax, colormap)
        ax.legend(handles=legend_elts, ncol=2, fontsize="small")
        return fig, ax
    d = survival_tests.copy()
    activity_var_stds = data[activity_variables].std() #TODO: should we separate male/female stds?
    d['std_male_logHR'] = d.activity_var.map(activity_var_stds) * d['male_logHR']
    d['std_female_logHR'] = d.activity_var.map(activity_var_stds) * d['female_logHR']
    d['activity_var_category'] = d.activity_var.map(activity_variable_descriptions.Subcategory)
    fig, ax = sex_difference_survival_plot(d)
    fig.savefig(OUTDIR+"survival.by_sex.png")

def circadian_component_plots():
    ## Plot the amount RA goes "beyond" other variables
    fig, (ax1, ax2) = pylab.subplots(ncols=2, figsize=(10,6))
    c = beyond_RA_tests.Subcategory.map(color_by_actigraphy_subcat)
    ax1.scatter(
        beyond_RA_tests['standardized log Hazard Ratio'].abs(),
        beyond_RA_tests['standardized log Hazard Ratio RA'].abs(),
        c=c)
    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax1.set_xlabel("log Hazard Ratio / SD of alternative variable")
    ax1.set_ylabel("log Hazard Ratio / SD of RA")
    ax1.axhline(survival_tests.loc[survival_tests.activity_var == "acceleration_RA", "standardized log Hazard Ratio"].abs().values,
                linestyle="--", c="k")
    ax2.scatter(
        -numpy.log10(beyond_RA_tests.p),
        -numpy.log10(beyond_RA_tests.p_RA),
        c=c)
    ax2.set_ylim(0, ax2.get_ylim()[1])
    ax2.set_xlabel("-log10 p-value of alternative variable")
    ax2.set_ylabel("-log10 p-value of RA")
    ax2.axhline(-numpy.log10(survival_tests.loc[survival_tests.activity_var == "acceleration_RA", "p"].values),
                linestyle="--", c="k")
    legend_from_colormap(fig, color_by_actigraphy_subcat, loc="upper left", fontsize="small", ncol=2)
    fig.savefig(OUTDIR+"additive_benefit_RA.png")

    top_phenotypes = phecode_tests[(~phecode_tests.activity_var.str.startswith('self_report')) & (phecode_tests.N_cases > 1000)].sort_values(by='p').phecode.unique()
    fig, axes = phewas_plots.circadian_component_plot(data, top_phenotypes[:20], len(phecode_groups))
    fig.savefig(OUTDIR+"circadian_vs_other_vars.png")

    top_phenotypes = quantitative_tests[(~quantitative_tests.activity_var.str.startswith('self_report'))].sort_values(by='p').phenotype.unique()
    fig, axes = phewas_plots.circadian_component_plot(data, top_phenotypes[:20], len(quantitative_variables), quantitative=True)
    fig.savefig(OUTDIR+"circadian_vs_other_vars.quantitative.png")

def objective_subjective_plots():
    ### Comparisons of self-reported versus objectively derived variables
    # Some self-reported variables are closely matched by objectively derived variables
    # so which one is best associated with phecodes?

    # self_report_sleep_duration versus main_sleep_duration_mean or total_sleep_mean
    # Sleep durations have a fairly U-shaped curve so we use the abs_dev versions here
    # and the total_sleep_mean variable includes napping time as well as sleep time
    # so we don't use that here since napping is investigated in the self_report_nap_during_day variable
    variable_pairs = { "sleep duration": ("sleep_duration", "main_sleep_duration_mean"),
                    "sleep duration abs. dev.": ("sleep_duration_abs_dev", "main_sleep_duration_mean_abs_dev"),
                    "phase": ("morning_evening_person", "phase"),
                    #("morning_evening_person", "phase_abs_dev"),
                    "napping": ("nap_during_day", "other_sleep_mean"),
                    "sleeplessness": ("sleeplessness", "WASO_mean")}
    #fig, (ax1, ax2) = pylab.subplots(ncols=2, sharey=True)
    fig, ax2 = pylab.subplots(figsize=(4,4))
    for i, (name, (self_report, activity_var)) in enumerate(variable_pairs.items()):
        if self_report == "sleep_duration_abs_dev":
            self_report_var = "self_report_sleep_duration_abs_dev"
        else:
            self_report_var = "self_report_" + self_report_circadian_variables[self_report]['name']
        print(f"Comparing: {self_report} to {activity_var}")
        self_report_survival = survival_tests.query(f"activity_var == '{self_report_var}'").iloc[0]
        self_report_objective = survival_tests.query(f"activity_var == '{activity_var}'").iloc[0]
        # Downsample the objective ot have the same population as the self_report
        downsampled_data = data[~data[self_report_var].isna()]
        downsampled_uncensored = data.uncensored[~data[self_report_var].isna()]
        downsampled_entry_age = data.entry_age[~data[self_report_var].isna()]
        covariate_formula = "BMI + smoking_ever"
        survival_test = smf.phreg(formula = f"age_at_death_censored ~ {activity_var} + sex + {covariate_formula}",
                                        data=downsampled_data,
                                        status=downsampled_uncensored,
                                        entry=downsampled_entry_age,
                                        ).fit()
        pvalues = pandas.Series(survival_test.pvalues, index=survival_test.model.exog_names)
        params = pandas.Series(survival_test.params, index=survival_test.model.exog_names)
        if self_report in data and data[self_report].dtype.name == "category":
            cat_counts = data[self_report].value_counts()
            most_common = cat_counts.idxmax()
            dropped_cats = cat_counts.index[cat_counts < 400].to_list() # Want at least this many in the category
            cats = '"' + '", "'.join(cat_counts.index[cat_counts >= 400].to_list()) + '"'
            subset = ~data[self_report].isin(dropped_cats)
            self_report_survival_test = smf.phreg(
                formula = f'age_at_death_censored ~ C({self_report}, levels=[{cats}]) + sex + {covariate_formula}',
                #formula = f'age_at_death_censored ~ {self_report} + sex + {covariate_formula}',
                data=data[subset],
                status=data.uncensored[subset],
                entry=data.entry_age[subset],
            ).fit()
            self_report_pvalue = self_report_survival_test.f_test(
                [[1 if j == i else 0 for j in range(len(self_report_survival_test.model.exog_names))]
                    for i, var in enumerate(self_report_survival_test.model.exog_names)
                    if self_report in var]
            ).pvalue
            self_report_pvalue = self_report_pvalue.flatten()[0]
        else:
            # Non-categorical values don't have an alternative, use the same as before
            self_report_pvalue = self_report_survival.p
        df = pandas.DataFrame.from_dict({
            "self_report": {
                "survival_p": self_report_survival.p,
                "survival_logHR": self_report_survival['standardized log Hazard Ratio'],
            },
            "objective": {
                "survival_p": self_report_objective.p,
                "survival_logHR": self_report_objective['standardized log Hazard Ratio'],
            },
            "downsampled_objective": {
                "survival_p": pvalues[activity_var],
                "survival_logHR": params[activity_var] * downsampled_data[activity_var].std(),
            },
            "self_report_cat": {
                "survival_p": self_report_pvalue,
            },
        })
        print(df.T)
        # Plot the points
        #ax1.scatter([i], [-numpy.log10(df.loc["survival_p", "self_report"])], c="k")
        #ax1.scatter([i], [-numpy.log10(df.loc["survival_p", "downsampled_objective"])], c="r")
        #ax2.scatter([i], [-numpy.log10(df.loc["survival_p", "self_report_cat"])], c="k")
        #ax2.scatter([i], [-numpy.log10(df.loc["survival_p", "objective"])], c="r")
        BAR_WIDTH = 0.6
        ax2.bar(2*i-BAR_WIDTH/2, -numpy.log10(df.loc['survival_p', 'self_report_cat']), color="k", width=BAR_WIDTH)
        ax2.bar(2*i+BAR_WIDTH/2, -numpy.log10(df.loc['survival_p', 'objective']), color="r", width=BAR_WIDTH)
    #ax1.set_ylabel("-log10 p-value")
    #ax1.set_xticks(range(len(variable_pairs)))
    #ax1.set_xticklabels([f"{self_report}\n{activity_var}" for self_report, activity_var in variable_pairs])
    #ax2.set_xticks(range(len(variable_pairs)))
    #ax2.set_xticklabels([f"{self_report}\n{activity_var}" for self_report, activity_var in variable_pairs])
    #ax1.xaxis.set_tick_params(rotation=90)
    ax2.set_xticks(2*numpy.arange(len(variable_pairs)))
    ax2.set_xticklabels([name for name in variable_pairs.keys()])
    ax2.set_ylabel("-log10 p-value")
    ax2.xaxis.set_tick_params(rotation=90)
    legend_elts = [matplotlib.lines.Line2D(
                            [0],[0],
                            marker="o", markerfacecolor=c, markersize=10,
                            label=l,
                            c=c, lw=0)
                        for c, l in zip(["k", "r"], ["Subjective", "Objective"])]
    ax2.legend(handles=legend_elts)
    fig.tight_layout()
    fig.savefig(OUTDIR+"subjective_objective_comparison.survival.png")


    # Compare the objective/subjective variables again on the phecodes
    fig, axes = pylab.subplots(ncols=len(variable_pairs), figsize=(15,7), sharex=True, sharey=True)
    Q_CUTOFF = 0.05
    for ax, (name, (self_report, activity_var)) in zip(axes.flatten(), variable_pairs.items()):
        if self_report is not "sleep_duration_abs_dev":
            self_report_var = "self_report_" + self_report_circadian_variables[self_report]['name']
        else:
            self_report_var = "self_report_sleep_duration_abs_dev"
        either_significant = phecode_tests[
            ((phecode_tests.activity_var == self_report_var) |
            (phecode_tests.activity_var == activity_var)) &
            (phecode_tests.q < Q_CUTOFF)].phecode.unique()
        downsampled_data = data[~data[self_report_var].isna()]
        objective_tests_list = []
        for phecode in either_significant:
            covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
            N = data[phecode].sum()
            fit = OLS(f"{activity_var} ~ Q({phecode}) + sex * ({covariate_formula})",
                        data=downsampled_data)
            p = fit.pvalues[f"Q({phecode})"]
            coeff = fit.params[f"Q({phecode})"]
            std_effect = coeff / data[activity_var].std()
            N_cases = data.loc[~data[activity_var].isna(), phecode].sum()
            objective_tests_list.append({"phecode": phecode,
                                    "activity_var": activity_var,
                                    "p": p,
                                    "coeff": coeff,
                                    "std_effect": std_effect,
                                    "N_cases": N_cases,
                                })
        objective_tests = pandas.DataFrame(objective_tests_list)
        subjective_ps = objective_tests.phecode.map(phecode_tests[(phecode_tests.activity_var == self_report_var)].set_index("phecode").p)
        colors = objective_tests.phecode.map(phecode_info.category).map(color_by_phecode_cat)
        ax.scatter(-numpy.log10(objective_tests.p), -numpy.log10(subjective_ps), c=colors)
        ax.set_aspect("equal")
        ax.set_xlabel(f"-log10 p objective variable")
        if ax == axes.flatten()[0]:
            ax.set_ylabel(f"-log10 p subjective variable")
        ax.set_title(name)
    for ax in axes.flatten():
        bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                    numpy.max([ax.get_xlim(), ax.get_ylim()]))
        diag = numpy.array([0, bound])
        ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal")
        ax.set_xlim(0,20)
        ax.set_ylim(0,20)
    fig.tight_layout()
    fig.savefig(OUTDIR+"objective_subjective_comparison.png")


def by_date_plots():
    fig, ax, hypertension_by_date = phewas_plots.plot_by_diagnosis_date(data, ["I10"], 401, "Hypertension", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.hypertension.png")

    fig, ax, diabetes_by_date = phewas_plots.plot_by_diagnosis_date(data, ["E09", "E10", "E11", "E13"], 250, "Diabetes", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.diabetes.png")

    fig, ax, IHD_by_date = phewas_plots.plot_by_diagnosis_date(data, ["I20", "I21", "I22", "I23", "I25"], 411, "Ischemic Heart Disease", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.IHD.png")

    fig, ax, pneumonia_by_date = phewas_plots.plot_by_diagnosis_date(data, ["J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18"], 480, "Pneumonia", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.pneumonia.png")

    fig, ax, mood_disorders_by_date = phewas_plots.plot_by_diagnosis_date(data, ["F30", "F31", "F32", "F33", "F34", "F39"], 296, "Mood Disorders", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.mood_disorders.png")

    fig, ax, anxiety_by_date = phewas_plots.plot_by_diagnosis_date(data, ["F40", "F41"], 300, "Anxiety Disorders", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.anxiety_disorders.png")

    fig, ax, renal_failure_by_date = phewas_plots.plot_by_diagnosis_date(data, ["N17", "N18", "N19", "Y60", "Y84", "Z49"], 585, "Renal Failure", icd10_entries, phecode_tests)
    fig.savefig(OUTDIR+"by_date.renal_failure.png")

def generate_results_table():
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
        survival_tests.sort_values(by="p").to_excel(writer, sheet_name="Survival Associations", index=False)
        phecode_tests_raw.sort_values(by="p").to_excel(writer, sheet_name="Phecode Associations", index=False)
        quantitative_tests_raw.sort_values(by="p").to_excel(writer, sheet_name="Quantitative Associations", index=False)
        phecode_tests_by_sex.sort_values(by="p_diff").to_excel(writer, sheet_name="Sex-specific Associations", index=False)
        age_tests.sort_values(by="p").to_excel(writer, sheet_name="Age-dependence", index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="run phewas pipeline on actigraphy\nOutputs to ../global_phewas/cohort#/")
    parser.add_argument("--cohort", help="Cohort number to load data for", type = int)
    parser.add_argument("--force_recompute", help="Whether to force a rerun of the statistical tests, even if already computed", default=False, action="store_const", const=True)
    parser.add_argument("--all", help="Whether to run all analyses. Warning: slow.", default=False, action="store_const", const=True)

    args = parser.parse_args()
    COHORT = args.cohort
    RECOMPUTE = args.force_recompute
    OUTDIR = f"../global_phewas/cohort{COHORT}/"
    FDR_CUTOFF_VALUE = 0.05

    #### Load and preprocess the underlying data
    data, ukbb, activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, phecode_data, phecode_groups, phecode_info, phecode_map, icd10_entries, icd10_entries_all = phewas_preprocess.load_data(COHORT)

    # Load descriptions + categorization of activity variables and quantitative variables
    activity_variable_descriptions = pandas.read_excel("../table_header.xlsx", index_col="Activity Variable", sheet_name="Variables")
    quantitative_variable_descriptions = pandas.read_excel("../quantitative_variables.xlsx", index_col=0)

    # Gather list of the
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
        return None # can't find it
    quantitative_variables = [find_var(c) for block in quantitative_blocks
                            for c in block
                            if find_var(c) is not None]



    #### Run (or load from disk if they already exist) 
    #### the statistical tests
    phecode_tests, phecode_tests_by_sex = phewas_tests.phecode_tests(data, phecode_groups, activity_variables, phecode_info, OUTDIR, RECOMPUTE)
    phecode_tests_raw = phecode_tests.copy()
    phecode_tests['activity_var_category'] = phecode_tests['activity_var'].map(activity_variable_descriptions.Category)
    phecode_tests['q_significant'] = (phecode_tests.q < FDR_CUTOFF_VALUE).astype(int)

    quantitative_tests = phewas_tests.quantitative_tests(data, quantitative_variables, activity_variables, OUTDIR, RECOMPUTE)
    quantitative_tests_raw  = quantitative_tests.copy()
    quantitative_tests['Functional Category'] = quantitative_tests.phenotype.map(quantitative_variable_descriptions['Functional Categories'])

    age_tests = phewas_tests.age_tests(data, phecode_groups, activity_variables, phecode_info, OUTDIR, RECOMPUTE)
    age_tests_raw = age_tests.copy()
    age_tests['activity_var_category'] = age_tests['activity_var'].map(activity_variable_descriptions.Category)

    survival_tests = phewas_tests.survival_tests(data, activity_variables, activity_variable_descriptions, OUTDIR, RECOMPUTE)

    beyond_RA_tests = phewas_tests.beyond_RA_tests(data, activity_variables, activity_variable_descriptions, OUTDIR, RECOMPUTE)

    #### Prepare color maps for the plots
    color_by_phecode_cat = {cat:color for cat, color in
                                zip(phecode_tests.phecode_category.unique(),
                                    [pylab.get_cmap("tab20")(i) for i in range(20)])}
    color_by_actigraphy_cat = {cat:color for cat, color in
                                    zip(["Sleep", "Circadianness", "Physical activity"],
                                        [pylab.get_cmap("Dark2")(i) for i in range(20)])}
    color_by_actigraphy_subcat = {cat:color for cat, color in
                                    zip(activity_variable_descriptions.Subcategory.unique(),
                                        [pylab.get_cmap("Set3")(i) for i in range(20)])}
    color_by_quantitative_function = {cat:color for cat, color in
                                        zip(quantitative_variable_descriptions['Functional Categories'].unique(),
                                            [pylab.get_cmap("tab20b")(i) for i in range(20)])}
    colormaps = {
        "phecode_cat": color_by_phecode_cat,
        "actigraphy_cat": color_by_actigraphy_cat,
        "actigraphy_subcat": color_by_actigraphy_subcat,
        "quantitative_function": color_by_quantitative_function,
    }

    ## Create the plotter object
    # for common plot types
    phewas_plots = plots.Plotter(phecode_info, colormaps, activity_variables, activity_variable_descriptions)

    ## Make the plots
    summary()
    sex_difference_plots()
    age_difference_plots()
    fancy_plots()
    survival_curves()
    survival_plots()
    objective_subjective_plots()
    circadian_component_plots()
    if args.all:
        # Note: slow to run: performs many regressions
        by_date_plots()

    ## Summarize everything
    generate_results_table()