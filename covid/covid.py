import pandas
import numpy
import pylab
import phewas_preprocess

COVID_ICD10 = "U071"
FATIGUE_ICD10 = "R53"

hesin = pandas.read_csv("../data/record/hesin.txt", sep="\t", parse_dates=["epistart"])
hesin_diag = pandas.read_csv("../data/record/hesin_diag.txt", sep="\t")
hesin_diag = pandas.merge(
    hesin_diag,
    hesin[['eid', 'ins_index', 'epistart']],
    on=['eid', 'ins_index'],
)

# COVID 19 from medical records
covid19_diag = hesin_diag[hesin_diag.diag_icd10 == COVID_ICD10]
covid19_diag = pandas.merge(
    covid19_diag,
    hesin[['eid', 'ins_index', 'epistart']],
    left_on=["eid", "ins_index"],
    right_on=["eid", "ins_index"]
)


# Death records
death = pandas.read_csv("../data/record/death.txt", sep="\t", parse_dates=["date_of_death"])
death_cause = pandas.read_csv("../data/record/death_cause.txt", sep="\t")
death_cause = pandas.merge(
    death,
    death_cause,
    left_on=["eid", "ins_index"],
    right_on=["eid", "ins_index"],
)

# COVID 19 from test results
covid19_result_england = pandas.read_csv("../data/record/covid19_result_england.txt", sep="\t", parse_dates=["specdate"])
covid19_result_england['source'] = 'England'
covid19_result_wales = pandas.read_csv("../data/record/covid19_result_wales.txt", sep="\t", parse_dates=["specdate"])
covid19_result_wales['source'] = 'Wales'
covid19_result_scotland = pandas.read_csv("../data/record/covid19_result_scotland.txt", sep="\t", parse_dates=["specdate"])
covid19_result_scotland['source'] = 'Scotland'
covid19_result = pandas.concat([covid19_result_england, covid19_result_wales, covid19_result_scotland])

# Load the main tables and actigraphy
ukbb = phewas_preprocess.load_ukbb()
# NOTE: all actigraphy, not divided into cohorts
activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity = phewas_preprocess.load_activity(ukbb)

cohort = activity.index.unique()


### TIMELINE
# Make a timeline of the study design timing so that readers can easily see when data was collected
ACTIGRAPHY_COLOR = "#1b998b"
REPEAT_COLOR = "#c5d86d"
ASSESSMENT_COLOR = "#333333"
TEST_COLOR = "#aaaaaa"
TEST_POSITIVE_COLOR = "#f46036"
DEATH_COLOR = "#333333"
COVID_DEATH_COLOR = "#f46036"
fig, (ax1, ax2, ax3) = pylab.subplots(figsize=(8,6), nrows=3)
#ax2.yaxis.set_inverted(True)
ax1.yaxis.set_label_text("Participants / month")
ax2.yaxis.set_label_text("Tests / month")
ax3.yaxis.set_label_text("Deaths / month")
#ax2.xaxis.tick_top()
for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
bins = pandas.date_range("2013-1-1", "2021-12-31", freq="1M")
def date_hist(ax, values, bins, **kwargs):
    # Default histogram for dates doesn't cooperate with being given a list of bins
    # since the list of bins doesn't get converted to the same numerical values as the values themselves
    counts, edges = numpy.histogram(values, bins)
    ax.bar(edges[:-1]+(edges[:-1] - edges[1:])/2, counts, width=(edges[1:]-edges[:-1])*1.05, **kwargs)
date_hist(ax2, covid19_result[covid19_result.eid.isin(cohort)].specdate, bins, color=TEST_COLOR)
date_hist(ax2, covid19_result[(covid19_result.result == 1) & covid19_result.eid.isin(cohort)].specdate, bins, color=TEST_POSITIVE_COLOR)
date_hist(ax3, death_cause[death_cause.eid.isin(cohort) & (death_cause.level == 1)].date_of_death, bins, color=DEATH_COLOR)
date_hist(ax3, death_cause[death_cause.eid.isin(cohort) & (death_cause.cause_icd10 == COVID_ICD10)].date_of_death, bins, color=COVID_DEATH_COLOR)
assessment_time = pandas.to_datetime(ukbb[ukbb.index.isin(cohort)].blood_sample_time_collected_V0)
actigraphy_time = pandas.to_datetime(activity_summary['file-startTime'])
actigraphy_seasonal_time = pandas.to_datetime(activity_summary_seasonal['file-startTime'], cache=False)
death_time = pandas.to_datetime(ukbb[~ukbb.date_of_death.isna() & ukbb.index.isin(cohort)].date_of_death)
#diagnosis_time = pandas.to_datetime(icd10_entries[icd10_entries.ID.isin(ukbb.index)].first_date)
date_hist(ax1, assessment_time, color=ASSESSMENT_COLOR, label="assessment", bins=bins)
date_hist(ax1, actigraphy_time, color=ACTIGRAPHY_COLOR, label="actigraphy", bins=bins)
date_hist(ax1, actigraphy_seasonal_time, color=REPEAT_COLOR, label="repeat actigraphy", bins=bins)
#date_hist(ax2, diagnosis_time, color=DIAGNOSIS_COLOR, label="Diagnoses", bins=bins)
date_hist(ax3, death_time, color=DEATH_COLOR, label="Diagnoses", bins=bins)
ax1.annotate("Assessment", (assessment_time.mean(), 0), xytext=(0,75), textcoords="offset points", ha="center")
ax1.annotate("Actigraphy", (actigraphy_time.mean(), 0), xytext=(0,75), textcoords="offset points", ha="center")
ax1.annotate("Repeat\nActigraphy", (actigraphy_seasonal_time.mean(), 0), xytext=(0,70), textcoords="offset points", ha="center")
#ax2.annotate("Medical Record\nDiagnoses", (diagnosis_time.mean(), 0), xytext=(0,60), textcoords="offset points", ha="center")
ax3.annotate("Deaths", (death_time.mean(), 0), xytext=(0,70), textcoords="offset points", ha="center")
#fig.savefig(OUTDIR+"summary_timeline.png")


# Extract cases and dates
cohort_covid19_result = covid19_result[covid19_result.eid.isin(cohort)]
covid_case = cohort_covid19_result[cohort_covid19_result.result == 1].eid.unique()
first_covid_date = cohort_covid19_result[cohort_covid19_result.result == 1].groupby("eid").specdate.min()
fatigue_case = hesin_diag[hesin_diag.diag_icd10 == FATIGUE_ICD10].eid.unique()
first_fatigue_date = hesin_diag[hesin_diag.diag_icd10 == FATIGUE_ICD10].groupby("eid").epistart.min()
fatigue_or_covid = pandas.DataFrame({
    "first_covid_date": first_covid_date,
    "first_fatigue_date": first_fatigue_date,
})
cohort_fatigue_case = cohort.intersection(fatigue_case)
fatigue_after_covid = fatigue_or_covid[fatigue_or_covid.first_fatigue_date > fatigue_or_covid.first_covid_date]
cohort_fatigue_after_covid = fatigue_after_covid.index.intersection(cohort)


### Summarize case counts
print("Counts")
print(pandas.Series({
    "total": len(activity),
    "covid_cases": len(covid_case),
    "fatigue_cases": len(cohort_fatigue_case),
    "fatigue_after_covid_cases": len(cohort_fatigue_after_covid),
    "covid_deaths": (death_cause.eid.isin(cohort) & (death_cause.cause_icd10 == COVID_ICD10)).sum(),
}))