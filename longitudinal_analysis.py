'''
Main script to run the longitudinal associations in a phenome-wide study
'''

import pathlib
import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm


import phewas_preprocess
import longitudinal_statistics
import longitudinal_diagnoses
import longitudinal_diagnostics
import day_plots

import util

DPI = 300
FDR_CUTOFF = 0.1
BONFERRONI_CUTOFF = 0.05
RESULTS_DIR = pathlib.Path("../longitudinal/")

def manhattan_plot(tests_df, minor_group_by="phecode", group_by=None, color_by=None, y_break=5):
    # "Manhattan" plot of the PheWAS
    fig, ax = pylab.subplots(nrows=1, figsize=(8,6), sharex=True)
    x = 0
    x_ticks = []
    x_minorticks = [-0.5]
    x_ticklabels = []
    for key, pt in tests_df.sample(frac=1).groupby(group_by):
        x_ticks.append(x+pt[minor_group_by].nunique()/2-0.5)
        x_ticklabels.append(util.capitalize(key))
        ax.scatter(x+numpy.arange(len(pt)), -numpy.log10(pt.p), color=color_by[key])
        x += len(pt)
        x_minorticks.append(x-0.5)
    ax.set_ylabel("-log10(p-value)")
    ax.axhline(-numpy.log10(BONFERRONI_CUTOFF), c="k", zorder = 2)
    ax.axhline(-numpy.log10(FDR_CUTOFF), c="k", linestyle="--", zorder = 2)
    ax.set_xticks(x_ticks)
    ax.set_xticks(x_minorticks, minor=True)
    ax.set_xticklabels(x_ticklabels, rotation=90)
    ax.xaxis.set_tick_params(which="major", bottom=False, top=False)
    ax.xaxis.set_tick_params(which="minor", length=10)
    ax.set_ylim(0,y_break)
    ax.set_xlim(-1, x+1)
    fig.tight_layout(h_pad=0.01)
    return fig

def summary():

    # Manhattan plot by phecode
    tests = predictive_tests_cox.copy()
    tests['category'] = tests.phecode.map(phecode_info.category).fillna("N/A").astype('category')
    cat_ordering = list(color_by_phecode_cat.keys()) + ['N/A']
    tests['category'].cat.reorder_categories(
        sorted(tests['category'].cat.categories, key=lambda x: cat_ordering.index(x)),
        inplace=True
    )
    color_by = {key:color for key, color in color_by_phecode_cat.items() if key in tests.category.unique()}
    fig = manhattan_plot(
        tests,
        minor_group_by="phecode",
        group_by=tests.category,
        color_by=color_by)
    legend = util.legend_from_colormap(fig, color_by, ncol=2, fontsize="small", loc="upper left", bbox_to_anchor=(1.0,0.9))
    fig.savefig(OUTDIR/"FIG2.manhattan_plot.png", bbox_extra_artists=[legend], bbox_inches='tight')

    ### TIMELINE
    # Make a timeline of the study design timing so that readers can easily see when data was collected
    ACTIGRAPHY_COLOR = "#1b998b"
    REPEAT_COLOR = "#c5d86d"
    DIAGNOSIS_COLOR = "#f46036"
    ASSESSMENT_COLOR = "#aaaaaa"
    DEATH_COLOR = "#333333"
    fig, (ax1, ax2, ax3) = pylab.subplots(figsize=(8,6), nrows=3)
    #ax2.yaxis.set_inverted(True)
    ax1.yaxis.set_label_text("Participants / month")
    ax2.yaxis.set_label_text("Diagnoses / month")
    ax3.yaxis.set_label_text("Deaths / month")
    #ax2.xaxis.tick_top()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    bins = pandas.date_range("2007-1-1", "2022-1-1", freq="1M")
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
    diagnosis_time = pandas.to_datetime(case_status[case_status.ID.isin(data.index)].first_date)
    date_hist(ax1, assessment_time, color=ASSESSMENT_COLOR, label="assessment", bins=bins)
    date_hist(ax1, actigraphy_time, color=ACTIGRAPHY_COLOR, label="actigraphy", bins=bins)
    date_hist(ax1, actigraphy_seasonal_time, color=REPEAT_COLOR, label="repeat actigraphy", bins=bins)
    date_hist(ax2, diagnosis_time, color=DIAGNOSIS_COLOR, label="Diagnoses", bins=bins)
    date_hist(ax3, death_time, color=DEATH_COLOR, label="Diagnoses", bins=bins)
    ax1.annotate("Assessment", (assessment_time.mean(), 0), xytext=(0,75), textcoords="offset points", ha="center")
    ax1.annotate("Actigraphy", (actigraphy_time.mean(), 0), xytext=(0,75), textcoords="offset points", ha="center")
    ax1.annotate("Repeat\nActigraphy", (actigraphy_seasonal_time.mean(), 0), xytext=(0,70), textcoords="offset points", ha="center")
    ax2.annotate("Novel In-Patient\nDiagnoses", (diagnosis_time.mean(), 0), xytext=(0,20), textcoords="offset points", ha="center")
    ax3.annotate("Deaths", (death_time.mean(), 0), xytext=(0,70), textcoords="offset points", ha="center")
    fig.savefig(OUTDIR/"FIG1a.summary_timeline.png")

    time_difference = (actigraphy_time - assessment_time).mean()
    print(f"Mean difference between actigraphy time and initial assessment time: {time_difference/pandas.to_timedelta('1Y')} years")

def generate_results_table():
    '''
    Results table gives full details of statistical tests performed
    '''
    #### Combine all tables into the summary with the header file
    results_file = OUTDIR / "results.xlsx"
    print(f"Writing out the complete results table to {results_file}")
    import openpyxl
    workbook = openpyxl.load_workbook("../longitudinal_study_table_header.xlsx")
    workbook.save(results_file)
    with pandas.ExcelWriter(results_file, mode="a") as writer:
        predictive_tests_cox.sort_values(by="p").to_excel(writer, sheet_name="Overall", index=False)
        predictive_tests_by_sex_cox.sort_values(by="sex_diff_p").to_excel(writer, sheet_name="By Sex", index=False)
        predictive_tests_by_age_cox.sort_values(by="age_diff_p").to_excel(writer, sheet_name="By Age", index=False)
        phecode_details.T.to_excel(writer, sheet_name="PheCODEs")

def temperature_trace_plots(N_IDS=500):
    '''
    Temperature trace plots show average temperature curves over the course of 24 hours
    and broken down by diagnosis or category
    '''
    ids = day_plots.get_ids_of_traces_available()
    temp_trace_dir = OUTDIR / "temperate_traces"
    temp_trace_dir.mkdir(exist_ok=True)

    def temp_to_C(temp):
        # Temperature is already in C at this point
        return temp

    ## Overall temperature cycle
    _, _, ax = day_plots.plot_average_trace(numpy.random.choice(ids, size=N_IDS, replace=False),
                    var="temp",
                    transform = temp_to_C,
                    normalize_mean = True)
    ax.set_ylabel("Temperature (C)")

    ## By categories
    def temp_trace_by_cat(cats, colors=None, show_variance=True, show_confidence_intervals=False, data=data):
        temp_mean = temp_to_C(full_activity.temp_mean_mean.mean())
        fig, ax = pylab.subplots()
        for cat in cats.cat.categories:
            selected_ids = data[(cats == cat) & (data.index.isin(ids))].index
            selected_ids = numpy.random.choice(selected_ids, size=min(len(selected_ids), N_IDS), replace=False)
            day_plots.plot_average_trace(selected_ids,
                        var="temp",
                        transform = temp_to_C,
                        normalize_mean = True,
                        set_mean = 0,
                        ax=ax,
                        color=colors[cat] if colors is not None else None,
                        label=cat,
                        show_variance=show_variance,
                        show_confidence_intervals=show_confidence_intervals)
        ax.set_ylabel("Temperature (C)")
        fig.legend()
        fig.tight_layout()
        return fig

    def case_control(phecode, data=data):
        cats = data[phecode].astype("category").cat.rename_categories({0:"Control", 1:"Case"})
        fig = temp_trace_by_cat(cats,
                                 colors = {"Case": "orange", "Control": "teal"}, data=data)
        fig.gca().set_title(phecode_info.loc[phecode].phenotype)
        fig.tight_layout()
        return fig

    fig = case_control(250)
    fig.savefig(temp_trace_dir+"temperature.diabetes.png")
    fig = case_control(401)
    fig.savefig(temp_trace_dir+"temperature.hypertension.png")
    fig = case_control(496)
    fig.savefig(temp_trace_dir+"temperature.chronic_airway_obstruction.png")
    fig = case_control(443)
    fig.savefig(temp_trace_dir+"temperature.peripheral_vascular_disease.png")
    fig = case_control(495)
    fig.savefig(temp_trace_dir+"temperature.asthma.png")
    fig = case_control(480)
    fig.savefig(temp_trace_dir+"temperature.pneumonia.png")
    fig = case_control(296)
    fig.savefig(temp_trace_dir+"temperature.mood_disorders.png")
    fig = case_control(300)
    fig.savefig(temp_trace_dir+"temperature.anxiety_disorders.png")
    fig = case_control(272)
    fig.savefig(temp_trace_dir+"temperature.lipoid_metabolism.png")

    morning_evening = data.morning_evening_person.cat.remove_categories(["Prefer not to answer", "Do not know"])
    fig = temp_trace_by_cat(morning_evening, show_variance=False)
    fig.gca().set_title("Chronotype")
    fig.tight_layout()
    fig.savefig(temp_trace_dir+"temperature.chronotype.png")

    age_cutoffs = numpy.arange(45,75,5) # every 5 years from 40 to 75
    age_categories = pandas.cut(data.age_at_actigraphy, age_cutoffs)
    fig = temp_trace_by_cat(age_categories, show_variance=False)
    fig.gca().set_title("Age")
    fig.tight_layout()
    fig.savefig(temp_trace_dir+"temperature.age.png")

    fig = temp_trace_by_cat(data.sex, colors=color_by_sex)
    fig.gca().set_title("Sex")
    fig.tight_layout()
    fig.savefig(temp_trace_dir+"temperature.sex.png")

    napping = data.nap_during_day.cat.remove_categories(["Prefer not to answer"])
    fig = temp_trace_by_cat(napping, show_variance=False)
    fig.gca().set_title("Nap During Day")
    fig.tight_layout()
    fig.savefig(temp_trace_dir+"temperature.nap.png")

    ## Asthma
    obese = data.BMI > 30
    normal = (data.BMI < 25) & (data.BMI > 18.5)
    cats = data[495].map({1: "Asthma", 0: "Control"}) + obese.map({True: " Obese", False: " Normal"})
    cats[(~normal) & (~obese)] = float("NaN") # Remove 'overweight-but-not-obese' middle category
    cats = cats.astype("category")
    fig = temp_trace_by_cat(cats, show_variance=False, show_confidence_intervals=True)
    fig.gca().set_title("Asthma by Weight")
    fig.tight_layout()
    fig.savefig(temp_trace_dir+"temperature.asthma.by_bmi.png")

    ## Hypertension interaction with Chronotype
    for label, chronotype in {"morning_person": "Definitely a 'morning' person", "evening_person": "Definitely an 'evening' person"}.items():
        d = data[data.morning_evening_person == chronotype]
        fig= case_control(401, data=d)
        fig.savefig(temp_trace_dir+f"temperature.hypertension.{label}.png")


    ## BMI versus Chronotype
    for label, chronotype in {"morning_person": "Definitely a 'morning' person", "evening_person": "Definitely an 'evening' person"}.items():
        d = data[data.morning_evening_person == chronotype]
        cats = pandas.qcut(d.BMI, numpy.linspace(0,1,6))
        fig= temp_trace_by_cat(cats, show_variance=False, show_confidence_intervals=False, data=d)
        fig.savefig(temp_trace_dir+f"temperature.bmi.{label}.png")

def temperature_calibration_plots():
    # Plot the (mis)-calibration of the tempreature variables
    device = activity_summary.loc[data.index, 'file-deviceID']
    random_device = pandas.Series(device.sample(frac=1.0).values, index=device.index)
    temp_mean = full_activity[full_activity.run == 0].set_index("id").loc[data.index, 'temp_mean_mean']
    temp_RA = data['temp_RA']
    temp_amplitude = data['temp_amplitude']

    fig, axes = pylab.subplots(figsize=(8,5), ncols=2)
    for (name, measure), ax in zip({"mean": temp_mean, "RA": temp_RA}.items(), axes):
        device_mean = measure.groupby(device).mean()
        random_mean = measure.groupby(random_device).mean()
        m = device_mean.quantile(0.01)
        M = device_mean.quantile(0.99)
        bins = numpy.linspace(m,M,21)
        ax.hist(random_mean, bins=bins, color='k', alpha=0.5, label="Randomized" if ax == axes[0] else None)
        ax.hist(device_mean, bins=bins, color='r', alpha=0.5, label="True" if ax == axes[0] else None)
        ax.set_xlabel(f"Temperature {name}")
    fig.legend()
    fig.savefig(OUTDIR/f"temperature_calibration.png")

    # Plot the cluster-level histograms
    activity = full_activity.copy().set_index('id')
    activity = activity[activity.run == 0]
    activity['device_id'] = activity_summary['file-deviceID']
    activity['device_cluster'] = pandas.cut( activity.device_id, [0, 7_500, 12_500, 20_000]).cat.rename_categories(["A", "B", "C"])
    fig, ax = pylab.subplots(figsize=(5,5))
    bins = numpy.linspace(
        activity.temp_amplitude.quantile(0.01),
        activity.temp_amplitude.quantile(0.99),
        31)
    for cluster, grouping in activity.groupby('device_cluster'):
        ax.hist(grouping.temp_amplitude, bins=bins, label="Cluster " + cluster, alpha=0.4, density=True)
    ax.set_xlabel("temp_amplitude (C)")
    ax.set_ylabel("Density")
    fig.legend()
    fig.savefig(OUTDIR/"temperature_calibration.histogram.png")

def demographics_table():
    # Create a table of demographics of the population studied, compared to the overall UK Biobank
    demographics = {}
    ukbb_without_actigraphy = ukbb[ukbb.actigraphy_file.isna()]
    for name, d in zip(["Actigraphy", "Without Actigraphy"], [data, ukbb_without_actigraphy]):
        demographics[name] = {
            "N": len(d),
            "Male": f'{(d.sex == "Male").mean():0.1%}',
            "Female": f'{(d.sex == "Female").mean():0.1%}',

            "White": f'{(d.ethnicity.isin(["British", "Any other white background", "Irish", "White"])).mean():0.1%}',
            "Nonwhite": f'{(~d.ethnicity.isin(["British", "Any other white background", "Irish", "White"])).mean():0.1%}',

            "Birth Year": f'{d.birth_year.mean():0.1f}±{d.birth_year.std():0.1f}',
            "BMI": f'{d.BMI.mean():0.1f}±{d.BMI.std():0.1f}',
        }
    demographics = pandas.DataFrame(demographics)
    demographics.to_csv(OUTDIR/"demographics.txt", sep="\t")
    print(demographics)
    return demographics

def predict_diagnoses_plots():
    phenotypes = top_phenotypes[::-1]
    tests = predictive_tests_cox.set_index('phecode').reindex(phenotypes)
    tests_by_sex = predictive_tests_by_sex_cox.set_index("meaning").reindex(tests.meaning).reset_index()
    tests_by_age = predictive_tests_by_age_cox.set_index("meaning").reindex(tests.meaning).reset_index()

    fig, axes = pylab.subplots(figsize=(9,7), ncols=4, sharey=True)
    ys = numpy.arange(len(tests))
    axes[0].barh(
        ys,
        -numpy.log10(tests.p),
        color='k')
    axes[0].set_yticks(numpy.arange(len(tests)))
    axes[0].set_yticklabels(tests.meaning.apply(lambda x: util.wrap(x, 30)))
    axes[0].set_ylim(-0.5, len(tests)-0.5)
    p_cutoff_for_q05 = predictive_tests_cox[predictive_tests_cox.q > 0.05].p.min()
    axes[0].axvline(-numpy.log10(p_cutoff_for_q05)) # BH FDR cutoff 0.05
    axes[0].set_xlabel("-log10 p-value")

    axes[1].scatter(
        tests.std_logHR,
        ys,
        color='k',
    )
    for i, (idx, row) in enumerate(tests.iterrows()):
        axes[1].plot(
            [row.std_logHR- row.std_logHR_se*1.96, row.std_logHR+ row.std_logHR_se*1.96],
            [ys[i], ys[i]],
            color='k',
        )
    axes[1].axvline(0, color="k")
    axes[1].set_xlabel("logHR per SD")

    # By sex
    axes[2].scatter(
        tests_by_sex.std_male_logHR,
        ys + 0.1,
        color=color_by_sex['Male'],
    )
    axes[2].scatter(
        tests_by_sex.std_female_logHR,
        ys-0.1,
        color=color_by_sex['Female'],
    )
    for i, (idx, row) in enumerate(tests_by_sex.iterrows()):
        axes[2].plot(
            [row.std_male_logHR - row.std_male_logHR_se*1.96, row.std_male_logHR+ row.std_male_logHR_se*1.96],
            [ys[i] + 0.1, ys[i] + 0.1],
            color=color_by_sex["Male"],
        )
        axes[2].plot(
            [row.std_female_logHR- row.std_female_logHR_se*1.96, row.std_female_logHR + row.std_female_logHR_se*1.96],
            [ys[i]-+ 0.1, ys[i] - 0.1],
            color=color_by_sex["Female"],
        )
    axes[2].set_xlabel("logHR per SD\nBy Sex")
    axes[2].axvline(0, color="k")

    #By age
    axes[3].scatter(
        tests_by_age.under_65_std_logHR,
        ys + 0.1,
        color=color_by_age['under 65'],
    )
    axes[3].scatter(
        tests_by_age.over_65_std_logHR,
        ys-0.1,
        color=color_by_age['over 65'],
    )
    for i, (idx, row) in enumerate(tests_by_age.iterrows()):
        axes[3].plot(
            [row.under_65_std_logHR - row.under_65_std_logHR_se*1.96, row.under_65_std_logHR + row.under_65_std_logHR_se*1.96],
            [ys[i] + 0.1, ys[i] + 0.1],
            color=color_by_age['under 65'],
        )
        axes[3].plot(
            [row.over_65_std_logHR - row.over_65_std_logHR_se*1.96, row.over_65_std_logHR + row.over_65_std_logHR_se*1.96],
            [ys[i]-+ 0.1, ys[i] - 0.1],
            color=color_by_age['over 65'],
        )
    axes[3].set_xlabel("logHR per SD\nBy Age")
    axes[3].axvline(0, color="k")
    fig.tight_layout(rect=(0,0,0.85,1))
    util.legend_from_colormap(fig, color_by_sex, loc=(0.85, 0.6))
    util.legend_from_colormap(fig, {str(k):v for k,v in color_by_age.items()}, loc=(0.85, 0.4))

    fig.savefig(OUTDIR / "FIG1.summary_results.png", dpi=300)

def predict_diagnoses_effect_size_tables():
    # Generate a table of the effect sizes for predcitive tests (prop hazard models)
    # for select phecodes
    phecodes = ['250', '401', '496', '272', '585', '480', '300']
    results = predictive_tests_cox.set_index("phecode").loc[phecodes]
    HR_1SD = numpy.exp(results.std_logHR).round(2).astype(str)
    HR_1SD_lower = numpy.exp(results.std_logHR - 1.96* results.std_logHR_se).round(2).astype(str)
    HR_1SD_upper = numpy.exp(results.std_logHR + 1.96* results.std_logHR_se).round(2).astype(str)
    HR_2SD = numpy.exp(results.std_logHR*2).round(2).astype(str)
    HR_2SD_lower = numpy.exp((results.std_logHR - 1.96* results.std_logHR_se)*2).round(2).astype(str)
    HR_2SD_upper = numpy.exp((results.std_logHR + 1.96* results.std_logHR_se)*2).round(2).astype(str)

    print("Hazard Ratios of common diagnoses per SD of temp_RA")
    results = pandas.DataFrame({
        "diagnosis": results.meaning,
        "HR at 1SD": HR_1SD + " (" + HR_1SD_lower + "-" + HR_1SD_upper + ")",
        "HR at 2SD": HR_2SD + " (" + HR_2SD_lower + "-" + HR_2SD_upper + ")",
    })
    print(results)
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="run phewas longidutinal pipeline on actigraphy\nOutputs to {RESULTS_DIR}/cohort#/")
    parser.add_argument("--cohort", help="Cohort number to load data for", type = int)
    parser.add_argument("--force_recompute", help="Whether to force a rerun of the statistical tests, even if already computed", default=False, action="store_const", const=True)
    parser.add_argument("--all", help="Whether to run all analyses. Warning: slow.", default=False, action="store_const", const=True)
    parser.add_argument("--no_plots", help="Disable running plots, useful for just loading the data", default=False, action="store_const", const=True)
    parser.add_argument("--no_display", help="Disable visual output, uses non-graphical backend such as when running on a server", default=False, action="store_const", const=True)

    args = parser.parse_args()

    import matplotlib
    if args.no_display:
        # Use the non-graphical backend Agg
        matplotlib.use("Agg")
    import pylab

    COHORT = args.cohort
    RECOMPUTE = args.force_recompute
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTDIR = RESULTS_DIR / f"cohort{COHORT}"
    OUTDIR.mkdir(exist_ok=True)

    #### Load and preprocess the underlying data
    data, ukbb, activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity = phewas_preprocess.load_data(COHORT)
    selected_ids = data.index
    actigraphy_start_date = pandas.Series(data.index.map(pandas.to_datetime(activity_summary['file-startTime'])), index=data.index)

    case_status, phecode_info, phecode_details = longitudinal_diagnoses.load_longitudinal_diagnoses(selected_ids, actigraphy_start_date)

    #### Run (or load from disk if they already exist) 
    #### the statistical tests
    predictive_tests_cox = longitudinal_statistics.predictive_tests_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE)
    predictive_tests_by_sex_cox = longitudinal_statistics.predictive_tests_by_sex_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE)
    predictive_tests_by_age_cox = longitudinal_statistics.predictive_tests_by_age_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE)


    #### Prepare color maps for the plots
    color_by_phecode_cat = {(cat if cat==cat else 'N/A'):color for cat, color in
                                zip(phecode_info.category.unique(),
                                    [pylab.get_cmap("tab20")(i) for i in range(20)])}
    color_by_sex = {'Male': '#1f77b4', 'Female': '#ff7f0e'}
    color_by_age = {"under 65": '#32a852', "over 65": '#37166b'}

    # The top phenotypes that we will highlight
    top_phenotypes = ['250.2', '401.1', '571', '585', '562.1', '327.3', '480', '272', '327', '740', '550.2', '716']

    ## Make the plots
    if not args.no_plots:
        temperature_calibration_plots()
        predict_diagnoses_plots()
        if args.all:
            # Note: slow to run
            temperature_trace_plots()

        ## Summarize everything
        generate_results_table()
        summary()

    # For running the more compute-intensive diagnositcs processes
    # e.g. checking cox.zph, time-varying models
    if args.all:
        longitudinal_diagnostics.diagnostics(data, case_status, top_phenotypes, OUTDIR)

    predict_diagnoses_effect_size_tables()
    demographics_table()
