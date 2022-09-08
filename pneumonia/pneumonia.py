import pathlib
import builtins

import pandas
import numpy
import seaborn as sns
import matplotlib
import pylab
import phewas_preprocess
import statsmodels.formula.api as smf

from medications import has_med_of_interest

import rpy2.robjects as robjects
import rpy2.robjects.packages #note: Don't remove! Necessary import for robjects
from rpy2.robjects import pandas2ri
rutils = robjects.packages.importr("utils")
rstats = robjects.packages.importr("stats")
rgrDev = robjects.packages.importr("grDevices")
packnames = ['survival']
names_to_install = [x for x in packnames if not robjects.packages.isinstalled(x)]
rutils.chooseCRANmirror(ind=1) # select the first mirror in the list
if len(names_to_install) > 0:
    rutils.install_packages(robjects.vectors.StrVector(names_to_install))
rsurvival = robjects.packages.importr('survival')


debug = True
OUTDIR = pathlib.Path("pneumonia/results/")
OUTDIR.mkdir(exist_ok=True)

log_file = open(OUTDIR / "log.txt", "w")
def print(*msg):
    builtins.print(*msg)
    if not log_file.closed:
        builtins.print(*msg, file = log_file)

# Model covariates included in the various models
MODEL_COVARIATES = {'full': [
        'sex',
        'smoking',
        'acceleration_overall',
        'age_at_ICU_visit',
        'age_at_actigraphy',
        'has_med_of_interest'
    ],
    'small': [
        'sex',
        'age_at_ICU_visit',
        'age_at_actigraphy',
    ]
}

# ICD codes for Pneumonia/FLu/Lower Respiratory Infection
PNEUMONIA_FLU_ICD10 = ["J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J20", "J21", "J22"]

# Join episodes that are separated by a gap of at most
EPI_JOIN_DIFFERENCE = pandas.to_timedelta("1D")
YEAR = 365.25*pandas.to_timedelta("1D")

''' 
Coding 7009 for Discharge status (ccdisstat):
1	Fully ready for discharge
2	Discharge for palliative care
3	Early discharge due to shortage of critical care beds
4	Delayed discharge due to shortage of other ward beds
5	Current level of care continuing in another location
6	More specialised care in another location
7	Self-discharge against medical advice
8	Patient died (no organs donated)
9	Patient died (heart beating solid organ donor)
10	Patient died (cadaveric tissue donor)
11	Patient died (non-heart beating solid organ donor)
'''
DISCHARGE_DEAD = [2,8,9,10,11]
DISCHARGE_RECOVERED = [1, 3, 4]
# TODO: 5,6 loss to followup - check separately?
DISCHARGE_NOT_RECOVERED = [5, 6, 7]
DISCHARGE_TRANSFERED = [4, 5, 6]
def discharge_type(disstatus):
    if disstatus in DISCHARGE_DEAD:
        return "dead"
    if disstatus in DISCHARGE_RECOVERED:
        return "recovered"
    if disstatus in DISCHARGE_NOT_RECOVERED:
        return "not recovered"
    return "unknown"

'''
CODING 268 for dismeth_uni of hesin
1000	Discharged on clinical advice/consent	1000	Top
1001	Discharged on clinical advice/consent: From inpatient/daycase care	1001	1000
1002	Discharged on clinical advice/consent: Transfer within provider	1002	1000
1003	Discharged on clinical advice/consent: Transfer to another provider	1003	1000
1004	Discharged on clinical advice/consent: Leave of absence granted	1004	1000
1005	Discharged on clinical advice/consent: By mental health review	1005	1000
1006	Discharged on clinical advice/consent: Under community care order	1006	1000
2000	Discharged without clinical advice/consent	2000	Top
2001	Discharged without clinical advice/consent: Self, relative or advocate	2001	2000
2002	Discharged without clinical advice/consent: Self discharge	2002	2000
2003	Discharged without clinical advice/consent: Discharged by relative	2003	2000
2004	Discharged without clinical advice/consent: Absconded from detention	2004	2000
3000	Patient death	3000	Top
3001	Patient death: Post-mortem performed	3001	3000
3002	Patient death: Post-mortem not performed	3002	3000
3003	Patient death: Whilst on pass	3003	3000
3004	Patient death: Stillbirth	3004	3000
4000	Not applicable	4000	Top
5000	Not known	5000	Top
'''
HESIN_DISCHARGE_RECOVERED = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 2000, 2001, 2002, 2003, 2004]
HESIN_DISCHARGE_NOT_RECOVERED = [5000]
HESIN_DISCHARGE_DEATH = [3000, 3001, 3002, 3003, 3004]
HESIN_DISCHARGE_DROP = [4000] # These records appear problematic and are dropped - not many of them in our cohort anyway


# medical records
hesin = pandas.read_csv("../data/patient_records/hesin.txt", sep="\t", parse_dates=["epistart", "epiend"], dayfirst=True)
hesin_diag = pandas.read_csv("../data/patient_records/hesin_diag.txt", sep="\t")
hesin_diag = pandas.merge(
    hesin_diag,
    hesin[['eid', 'ins_index', 'epistart', 'epiend']],
    on=['eid', 'ins_index'],
    how="left",
)

# Pneumonia/Flu from inpatient hospital records
unique_diags = hesin_diag.diag_icd10.unique()
pneumonia_flu_diag_codes = [code for code in unique_diags
                                if any(str(code).startswith(match) for match in PNEUMONIA_FLU_ICD10)]
pneumonia_flu_diag = hesin_diag[hesin_diag.diag_icd10.isin(pneumonia_flu_diag_codes)]
# entries that involve pneumonia/flu diagnosis
hesin_entry_pneumonia_flu = pneumonia_flu_diag.groupby(["eid", "ins_index"]).size().reset_index()[['eid', 'ins_index']]

# ICU stays
hesin_critical = pandas.read_csv("../data/patient_records/hesin_critical.txt", sep="\t", parse_dates=["ccstartdate", "ccdisdate"], dayfirst=True)
# Unknowns are coded as 999
hesin_critical.loc[hesin_critical.bressupdays == 999, 'bressupdays'] = 0#float("NaN")
hesin_critical.loc[hesin_critical.aressupdays == 999, 'aressupdays'] = 0#float("NaN")

### Generate a simplified hesin_critical table containing pneumonia/flu diagnoses
### and with visits merged across transfers
icu_visits = hesin_critical[['eid', 'ins_index', 'ccstartdate', 'ccdisdate', 'ccdisstat', 'bressupdays', 'aressupdays']].copy()
#Mark whether any are pneumonia-flu cases
has_pneumonia_flu_diag = pandas.merge(
    icu_visits,
    pneumonia_flu_diag,
    on = ['eid', 'ins_index'],
    how="left"
).groupby(['eid', 'ins_index']).apply(lambda x: x.diag_icd10.count() > 0)
has_pneumonia_flu_diag.name = "pneumonia_flu_diag"
icu_visits['pneumonia_flu_diag'] = pandas.merge(
    icu_visits,
    has_pneumonia_flu_diag.reset_index(),
    on = ['eid', 'ins_index'],
    how = 'left'
)['pneumonia_flu_diag']
# Checked by hand - relevant cases appear to have 0 aressup since bressupdays = total ICU duration
icu_visits.aressupdays.fillna(0, inplace=True)

def merge_visits(data):
    ''' Given data from ONE subject, merge visits that are connected '''
    original = data.copy()
    if len(data) == 1:
        return data # Nothing to join
    data['next_ccstartdate'] = data.ccstartdate.shift(-1)
    #TODO: should we discharge status here?
    data['connected'] = (data.next_ccstartdate - data.ccdisdate) <= EPI_JOIN_DIFFERENCE
    if any(data.connected):
        # Assign instance numbers to the connected episodes:
        # increment by 1 for each non-connected episode
        connected_episode_instance = numpy.cumsum(~data.connected)
        data['connected_episode_instance'] = connected_episode_instance.shift(1, fill_value=0)
        def aggregate(episode):
            return pandas.Series({
                "bressupdays": episode.bressupdays.sum(),
                "aressupdays": episode.aressupdays.sum(),
                "ccstartdate": episode.ccstartdate.min(),
                "ccdisdate": episode.ccdisdate.max(),
                "ccdisstat": episode.ccdisstat.iloc[-1],
                "pneumonia_flu_diag": episode.pneumonia_flu_diag.any(),
            })
        return data.groupby('connected_episode_instance').apply(aggregate).reset_index()
    else:
        return original
print("Merging icu visits")
icu_visits_merged = icu_visits.sort_values(["ccstartdate", 'ccdisdate']).groupby("eid").apply(merge_visits).drop(columns="eid").reset_index()

# Death records
death = pandas.read_csv("../data/patient_records/death.txt", sep="\t", parse_dates=["date_of_death"], dayfirst=True)
death_cause = pandas.read_csv("../data/patient_records/death_cause.txt", sep="\t")
death_cause = pandas.merge(
    death,
    death_cause,
    left_on=["eid", "ins_index"],
    right_on=["eid", "ins_index"],
)

# Load the main tables and actigraphy
ukbb = phewas_preprocess.load_ukbb()
# NOTE: all actigraphy, not divided into cohorts
activity, activity_summary, activity_summary_seasonal, activity_variables, activity_variance, full_activity = phewas_preprocess.load_activity(ukbb)
actigraphy_start = pandas.to_datetime(activity_summary['file-startTime'])

ukbb['actigraphy_start_date'] = ukbb.index.map(pandas.to_datetime(activity_summary['file-startTime']))
def year_to_jan_first(year):
    if year != year:
        return float("NaN")
    else:
        return str(int(year)) + "-01-01"
ukbb['birth_year_dt'] = pandas.to_datetime(ukbb.birth_year.apply(year_to_jan_first)) # As datetime
ukbb['age_at_actigraphy'] = (ukbb.actigraphy_start_date - ukbb.birth_year_dt) / YEAR

# Select the actigraphy cohort - we won't ever use anyone else
# Note that some additional subjects may have activity recordings (and an actigraphy start date)
# but have failed the QC checks
cohort = activity.index.unique()

# Do a 30 day-post discharge lookup for deaths
# Also count based off discharge status since a very small number of deaths appear to be missing in the death records
icu_visits_merged['death_in_30days'] = (
    icu_visits_merged.eid.map(death.groupby('eid').date_of_death.min()) < icu_visits_merged.ccdisdate + pandas.to_timedelta("30D")
) | icu_visits_merged.ccdisstat.isin(DISCHARGE_DEAD)

# Do a 30 day before/after of ICU visit check for pneumonia/flu diagnosis
icu_cross_diag = pandas.merge(
    icu_visits_merged,
    pneumonia_flu_diag[['eid', 'epistart', 'epiend']].rename(columns={"epistart": "diag_start", 'epiend': 'diag_end'}),
    on = "eid"
)
icu_cross_diag['lookupstart'] = icu_cross_diag.ccstartdate - pandas.to_timedelta('30D')# TODO: what time period should we use for the lookup?
icu_cross_diag['lookupend'] = icu_cross_diag.ccdisdate + pandas.to_timedelta('30D')
icu_cross_diag[
    (icu_cross_diag.lookupstart < icu_cross_diag.diag_end) &
    (icu_cross_diag.lookupend > icu_cross_diag.diag_start)
]
pneumonia_flu_30day_lookup = (icu_cross_diag.groupby(["eid", "ccstartdate"]).size() > 0).astype(float)
pneumonia_flu_30day_lookup.name = "pneumonia_flu_30day_lookup"
icu_visits_merged['pneumonia_flu_30day_lookup'] = pandas.merge(
    icu_visits_merged[['eid', 'ccstartdate']],
    pneumonia_flu_30day_lookup.reset_index(),
    on = ["eid", "ccstartdate"],
    how = "left",
).pneumonia_flu_30day_lookup.fillna(0).astype(bool)


### Process hospitalizations w/ actigraphy + pneumonia/flu
hesin_cohort = hesin[hesin.eid.isin(cohort)].copy()
hesin_cohort.index = pandas.RangeIndex(len(hesin_cohort))
#Mark whether any are pneumonia-flu cases
has_pneumonia_flu_diag = pandas.merge(
    hesin_cohort,
    pneumonia_flu_diag,
    on = ['eid', 'ins_index'],
    how="left"
).groupby(['eid', 'ins_index']).apply(lambda x: x.diag_icd10.count() > 0)
has_pneumonia_flu_diag.name = "pneumonia_flu_diag"
hesin_cohort['pneumonia_flu_diag'] = pandas.merge(
    hesin_cohort[['eid', 'ins_index']],
    has_pneumonia_flu_diag.reset_index(),
    on = ['eid', 'ins_index'],
    how = 'left'
)['pneumonia_flu_diag']

first_pneumonia_flu_diag = hesin_cohort[hesin_cohort.pneumonia_flu_diag].groupby('eid').epistart.min()

def merge_hospital_visits(data):
    ''' Given data from ONE subject, merge hospital visits that are connected '''
    original = data.copy()
    if len(data) == 1:
        return data # Nothing to join
    data['next_epistart'] = data.epistart.shift(-1)
    data['connected'] = (data.next_epistart - data.epiend) <= EPI_JOIN_DIFFERENCE
    if any(data.connected):
        # Assign instance numbers to the connected episodes:
        # increment by 1 for each non-connected episode
        connected_episode_instance = numpy.cumsum(~data.connected)
        data['connected_episode_instance'] = connected_episode_instance.shift(1, fill_value=0)
        def aggregate(episode):
            return pandas.Series({
                "epistart": episode.epistart.min(),
                "epiend": episode.epiend.max(),
                "dismeth_uni": episode.dismeth_uni.iloc[-1],
                "pneumonia_flu_diag": episode.pneumonia_flu_diag.any(), #TODO:
            })
        return data.groupby('connected_episode_instance').apply(aggregate).reset_index()
    else:
        return original
print("Merging hospital visits in pneumonia/flu cohort")
hesin_cohort_merged = hesin_cohort.sort_values(['epistart', 'epiend']).groupby('eid').apply(merge_hospital_visits).drop(columns="eid").reset_index()

### Count number of cases
# Not necessarily with no prior pneumonia diagnosis
icu_after_actigraphy = icu_visits_merged[icu_visits_merged.eid.map(actigraphy_start) < icu_visits_merged.ccstartdate].copy()
icu_after_actigraphy['discharge_status'] = icu_after_actigraphy.ccdisstat.map(discharge_type)
outcomes = pandas.DataFrame({
    "eid": icu_after_actigraphy.eid,
    "RA": icu_after_actigraphy.eid.map(activity.acceleration_RA),
    "RA_quartile": pandas.qcut(icu_after_actigraphy.eid.map(activity.acceleration_RA), numpy.linspace(0,1,5)),
    "acceleration_overall": icu_after_actigraphy.eid.map(activity.acceleration_overall),
    "ccstartdate": icu_after_actigraphy.ccstartdate,
    "ccdisdate": icu_after_actigraphy.ccdisdate,
    "total_ressup": icu_after_actigraphy.aressupdays + icu_after_actigraphy.bressupdays,
    "ICU_duration": (icu_after_actigraphy.ccdisdate - icu_after_actigraphy.ccstartdate) / pandas.to_timedelta("1D"),
    "aressup": icu_after_actigraphy.aressupdays,
    "pneumonia_flu_diag": icu_after_actigraphy.pneumonia_flu_diag,
    "pneumonia_flu_30day_lookup": icu_after_actigraphy.pneumonia_flu_30day_lookup,
    "death_in_30days": icu_after_actigraphy.death_in_30days.astype(int),
    "censored": icu_after_actigraphy['discharge_status'] != 'recovered',
    "smoking": icu_after_actigraphy.eid.map(ukbb.smoking).isin(['Previous', 'Current']),
    "sex": icu_after_actigraphy.eid.map(ukbb.sex),
    "age_at_actigraphy": icu_after_actigraphy.eid.map(ukbb.age_at_actigraphy),
    "age_at_ICU_visit": (icu_after_actigraphy.ccstartdate - icu_after_actigraphy.eid.map(ukbb.birth_year_dt)) / YEAR
    "has_med_of_interest": icu_after_actigraphy.eid.map(has_med_of_interest).fillna(False),
})
outcomes['visit_id'] = outcomes.index
outcomes = outcomes[~outcomes.RA.isna()]
outcomes['RA_quintile'] = pandas.qcut(outcomes['RA'], numpy.linspace(0,1,6)).cat.rename_categories(
    ["1st", "2nd", "3rd", "4th", "5th"]
).astype(str).astype('category')

print("ICU Case counts (with actigraphy):")
print(pandas.Series({
    "pneumonia/flu": outcomes.query("pneumonia_flu_diag").eid.nunique(),
    "pneumonia/flu 30days": outcomes.query("pneumonia_flu_30day_lookup").eid.nunique(),
    "pneumonia/flu 30days - death": outcomes.query("pneumonia_flu_30day_lookup & (death_in_30days==1)").eid.nunique(),
    "pneumonia/flu ressup": outcomes.query("pneumonia_flu_diag & (total_ressup > 0)").eid.nunique(),
    "pneumonia/flu 30days - ressup": outcomes.query("pneumonia_flu_30day_lookup & (total_ressup > 0)").eid.nunique(),
    "pneumonia/flu aressup": outcomes.query("pneumonia_flu_diag & (aressup > 0)").eid.nunique(),
    "pneumonia/flu 30days - aressup": outcomes.query("pneumonia_flu_30day_lookup & (aressup > 0)").eid.nunique(),
}))

# Breakdowns of the outcomes among pneumonia/flu cases
print("censored:")
print(outcomes.query("pneumonia_flu_30day_lookup").groupby("RA_quartile").censored.value_counts().unstack(1))
print("ICU_duration:")
print(outcomes.query("pneumonia_flu_30day_lookup").groupby("RA_quartile").ICU_duration.describe())
print("total_ressup:")
print(outcomes.query("pneumonia_flu_30day_lookup").groupby("RA_quartile").total_ressup.describe())
print("aressup:")
print(outcomes.query("pneumonia_flu_30day_lookup").groupby("RA_quartile").aressup.describe())

# Perform the COX models for ICU visits with pneumonia/flu
#TODO: include prior case status as a covariate
outcome_vars = ['ICU_duration', 'total_ressup', 'aressup']
covariates = MODEL_COVARIATES['full']
for outcome_var in outcome_vars:
    selected_ICU = outcomes.query("pneumonia_flu_30day_lookup")[[outcome_var, 'RA', 'RA_quintile', 'censored', 'eid', 'visit_id', 'death_in_30days'] + covariates].dropna(how="any", axis=0).copy()
    selected_ICU['event'] = selected_ICU.censored.map({True: "censored", False: "discharged"})
    selected_ICU.loc[selected_ICU.death_in_30days == 1, 'event'] = 'death'
    selected_ICU['event'] = pandas.Categorical(selected_ICU.event, categories=["censored", "discharged", "death"])

    # Run it in R
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        ricu_selected_ICU = robjects.conversion.py2rpy(selected_ICU)
    robjects.globalenv['icu_selected_ICU'] = ricu_selected_ICU

    # Fit the basic model in R
    rfit = robjects.r(f'''
    res <- coxph(
        Surv({outcome_var}, event) ~ RA_quintile + age_at_ICU_visit + age_at_actigraphy + smoking + acceleration_overall + has_med_of_interest + sex,
        data=icu_selected_ICU,
        id = visit_id,
        cluster=eid,
    )
    print(summary(res))
    res
    ''')
    print(f"N unique individuals: {selected_ICU.eid.nunique()}")
    # effect size summary
    RA_coeff = rstats.coefficients(rfit)[0]
    RA_hr = RA_coeff * selected_ICU.RA.std()
    print(f"logHR of RA/SD {RA_hr}")
    print(f"HR of 1 SD better rhythms {numpy.exp(RA_hr)}")

    if False:
        # Examine the validity of prop hazards assumptions
        zph_output = str(OUTDIR / f"zph.RA.ICU.{outcome_var}.png")
        zph = robjects.r('''
        zph <- cox.zph(res)
        print(zph)
        zph
        ''')

        robjects.r(f'''
        png("pneumonia/results/test.png");
        plot(zph[1])
        dev.off()
        #plot(c(1,2,3), c(4,10,3))
        ''')


# and by survival to 30 days in the same ICU cohort
results = smf.logit(
    "death_in_30days ~ RA + " + " + ".join(covariates),
    data = outcomes.query("pneumonia_flu_30day_lookup"),
).fit()
print(results.summary())
print(f"N cases (deaths): {results.model.endog.sum()}")






### Pneumonia case hospital duration
pneumonia_hosp_after_actigraphy = hesin_cohort_merged[
    (hesin_cohort_merged.eid.map(actigraphy_start) < hesin_cohort_merged.epistart)
    & (hesin_cohort_merged.pneumonia_flu_diag)
    & (~hesin_cohort_merged.dismeth_uni.isin(HESIN_DISCHARGE_DROP))
].copy()
pneumonia_hosp_after_actigraphy['stay_duration'] = (pneumonia_hosp_after_actigraphy.epiend - pneumonia_hosp_after_actigraphy.epistart) / pandas.to_timedelta('1D') + 1
pneumonia_hosp_after_actigraphy['RA'] = pneumonia_hosp_after_actigraphy.eid.map(activity.acceleration_RA)
pneumonia_hosp_after_actigraphy['acceleration_overall'] = pneumonia_hosp_after_actigraphy.eid.map(activity.acceleration_overall)
pneumonia_hosp_after_actigraphy['smoking'] = pneumonia_hosp_after_actigraphy.eid.map(ukbb.smoking.isin(['Previous', 'Current']))
pneumonia_hosp_after_actigraphy['sex'] = pneumonia_hosp_after_actigraphy.eid.map(ukbb.sex)
pneumonia_hosp_after_actigraphy['has_med_of_interest'] = pneumonia_hosp_after_actigraphy.eid.map(has_med_of_interest).fillna(False)
pneumonia_hosp_after_actigraphy['age_at_visit'] = (pneumonia_hosp_after_actigraphy.epistart - pneumonia_hosp_after_actigraphy.eid.map(ukbb.birth_year_dt)) / YEAR
pneumonia_hosp_after_actigraphy['age_at_actigraphy'] = (pneumonia_hosp_after_actigraphy.eid.map(actigraphy_start) - pneumonia_hosp_after_actigraphy.eid.map(ukbb.birth_year_dt))/ YEAR
pneumonia_hosp_after_actigraphy['censored'] = ~pneumonia_hosp_after_actigraphy.dismeth_uni.isin(HESIN_DISCHARGE_RECOVERED)
pneumonia_hosp_after_actigraphy['prior_pneumonia_flu_case'] = (pneumonia_hosp_after_actigraphy.eid.map(first_pneumonia_flu_diag) < pneumonia_hosp_after_actigraphy.eid.map(actigraphy_start))
pneumonia_hosp_after_actigraphy['death_in_30days'] = ((
    pneumonia_hosp_after_actigraphy.eid.map(death.groupby('eid').date_of_death.min()) < pneumonia_hosp_after_actigraphy.epiend + pandas.to_timedelta("30D")
) |pneumonia_hosp_after_actigraphy.dismeth_uni.isin(HESIN_DISCHARGE_DEATH)).astype(int)
selected_hosp = pneumonia_hosp_after_actigraphy[[
    'eid',
    'epistart',
    'stay_duration',
    'RA',
    'acceleration_overall',
    'smoking',
    'sex',
    'has_med_of_interest',
    'age_at_visit',
    'age_at_actigraphy',
    'censored',
    'death_in_30days',
    'prior_pneumonia_flu_case',
]].dropna(how="any")
selected_hosp['event'] = pandas.Categorical(selected_hosp.censored.map({True: "censored", False: "discharged"}), categories=["censored", "discharged", "dead"])
selected_hosp.loc[selected_hosp.death_in_30days ==1, 'event'] = 'dead' # TODO use death or death at 30days?
selected_hosp['start'] = 0
selected_hosp['RA_quintile'] = pandas.qcut(selected_hosp['RA'], numpy.linspace(0,1,6)).cat.rename_categories(
    ["1st", "2nd", "3rd", "4th", "5th"]
).astype(str).astype('category')
selected_hosp['acceleration_overall_quintile'] = pandas.qcut(selected_hosp['acceleration_overall'], numpy.linspace(0,1,6)).cat.rename_categories(
    ["1st", "2nd", "3rd", "4th", "5th"]
).astype(str).astype('category')
selected_hosp['visit_id'] = selected_hosp.index

selected_hosp1 = selected_hosp.sort_values(by="epistart").groupby('eid').apply(lambda x: x.iloc[0])# First epsiode
selected_hosp1['RA_quintile'] = pandas.Categorical(selected_hosp1.RA_quintile, categories=sorted(selected_hosp1.RA_quintile.unique()))
selected_hosp1['acceleration_overall_quintile'] = pandas.Categorical(selected_hosp1.acceleration_overall_quintile, categories=sorted(selected_hosp1.acceleration_overall_quintile.unique()))

#hosp_duration_res = smf.phreg(
#    "stay_duration ~ RA + age_at_visit + age_at_actigraphy + smoking + acceleration_overall + has_med_of_interest + prior_pneumonia_flu_case",
#    data = selected_hosp1,
#    status = ~selected_hosp1.censored,
#    strata="sex",
#).fit()
#print("Hospital stay duration results")
#print(hosp_duration_res.summary())

# Run it in R
with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
    selected_hosp1_r = robjects.conversion.py2rpy(selected_hosp1)
    selected_hosp_r = robjects.conversion.py2rpy(selected_hosp)
robjects.globalenv['selected_hosp1'] = selected_hosp1_r
robjects.globalenv['selected_hosp'] = selected_hosp_r

# Fit the basic model in R
rfit = robjects.r('''
res <- coxph(
    Surv(start, stay_duration, event) ~ relevel(RA_quintile, ref='1st') + age_at_visit + age_at_actigraphy + smoking + acceleration_overall_quintile + has_med_of_interest + prior_pneumonia_flu_case + sex,
    data=selected_hosp,
    id=visit_id,
    cluster=eid,
)
print(summary(res))
res
''')
print(f"N unique individuals: {selected_hosp.eid.nunique()}")
print(f"N visits individuals: {selected_hosp.visit_id.nunique()}")
# effect size summary
#RA_coeff = rstats.coefficients(rfit)[0]
#RA_hr = RA_coeff * selected_hosp.RA.std()
#print(f"logHR of RA/SD {RA_hr}")
#print(f"HR of 1 SD better rhythms {numpy.exp(RA_hr)}")
summary = pandas.DataFrame(
    numpy.array(robjects.r('summary(res)$coefficients')),
    index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
    columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
)

# Plot the hazard ratios by RA quintile in hosp stay duration
with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    fig, (ax1,ax2,ax3) = pylab.subplots(figsize=(7,3.5), ncols=3, sharey=True)
    ys = numpy.array(
        [0] + [summary.loc[f'relevel(RA_quintile, ref = "1st"){quintile}_1:2', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'relevel(RA_quintile, ref = "1st"){quintile}_1:2', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax1.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax1.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax1.set_xlabel("RA quintile")
    ax1.set_ylabel("Hazard ratio")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax1.axhline(y=1, linestyle="--", color='k')
    #Plot the acceleration_overall values
    ys = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}_1:2', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}_1:2', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax2.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax2.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax2.set_xlabel("Average acceleration\nquintile")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax2.axhline(y=1, linestyle="--", color='k')
    # Covariates
    covariates = ['sexMale', 'smokingTRUE', 'has_med_of_interestTRUE', 'prior_pneumonia_flu_caseTRUE']
    covariate_labels = ['Male', 'Smoking ', 'Medication  ', 'Prior history  ']
    ys = numpy.array([
        summary.loc[f'{covariate}_1:2', 'coef'] for covariate in covariates
    ])
    ses = numpy.array(
        [summary.loc[f'{covariate}_1:2', 'robust se'] for covariate in covariates]
    )
    xs = numpy.arange(len(ys))
    ax3.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax3.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax3.set_xticks(xs)
    ax3.set_xticklabels(covariate_labels, rotation=90)
    ax3.axhline(y=1, linestyle="--", color='k')
    fig.suptitle("Hospital Stay Duration")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hosp_stay_duration.hazard_ratios.by_RA_quintile.png", dpi=300)


if debug:
    # Examine the effects of the model
    zph = robjects.r('''
    zph <- cox.zph(res)
    print(zph)
    zph
    ''')

    # Fit a time-interaction model
    # Can no longer assess with cox.zph since it assumes time-invariant model
    rfit2 = robjects.r('''
    res2 <- coxph(
        Surv(start, stay_duration, event == 'discharged') ~ RA_quintile + age_at_visit + age_at_actigraphy + smoking + acceleration_overall + has_med_of_interest + strata(prior_pneumonia_flu_case, sex),
        data=selected_hosp,
        cluster=eid,
        tt=function(x,t,...) x*t
    )
    print(summary(res2))
    res2
    ''')

    # Fit the basic model in R
    rfit3 = robjects.r('''
    res3 <- coxph(
        Surv(start, stay_duration, event == 'discharged') ~ pspline(RA, df=3) + age_at_visit + age_at_actigraphy + smoking + acceleration_overall + has_med_of_interest + strata(prior_pneumonia_flu_case, sex),
        data=selected_hosp,
        cluster=eid,
        tt=function(x,t,...) x*t
    )
    print(summary(res3))
    res3
    ''')




# Survival 30 days post discharge analysis in hospital visit cohort
selected_hosp2 = selected_hosp.sort_values(by="epistart").groupby('eid').apply(lambda x: x.iloc[-1]) # Last episode - first is biased towards non-death
hosp_survival_30days = smf.logit(
    "death_in_30days ~ RA + age_at_visit + age_at_actigraphy + sex + smoking + acceleration_overall + has_med_of_interest + prior_pneumonia_flu_case",
    data = selected_hosp2,
).fit()
print("Hospital stay - survival at 30 days post discharge results")
print(hosp_survival_30days.summary())
print(f"N cases (deaths): {hosp_survival_30days.model.endog.sum()}")
margeffs = pandas.Series(
    hosp_survival_30days.get_margeff(at='overall').margeff,
    index = hosp_survival_30days.model.exog_names[1:] # no intercept term in margeffs
)
print(f"Effect size: {margeffs['RA'] * selected_hosp2.RA.std() / selected_hosp2.death_in_30days.mean()}")

# Describe the data by quantile
print("Hospital data - stay duration by RA quintile")
print(selected_hosp1.groupby("RA_quintile").stay_duration.describe())
color_by_quintile = {
    cat: color
    for cat, color in zip(
        sorted(selected_hosp1.RA_quintile.unique()),
        [(1.0, 0.0, 0.0), (0.7, 0.7, 0.7), (0.5, 0.5, 0.5), (0.3, 0.3, 0.3), (0.0, 0.0, 0.0)]
    )
}
fig = sns.displot(data = selected_hosp1, x = "stay_duration", hue="RA_quintile", kind="ecdf", palette=color_by_quintile)
fig.set(xlim=(0,100))
fig.set(xlabel="Hopsital stay duration\npneumonia/flu/acute lower respiratory infection cases")
fig.savefig(OUTDIR / "hosp_stay_duration.by_RA.png", dpi=300)

with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    # Box plot of the stay duration by quintile
    fig, ax = pylab.subplots(figsize=(3,3))
    sns.boxplot(
        data=selected_hosp1,
        x="RA_quintile",
        y="stay_duration",
        showfliers=False,
        color="white",
        linewidth=2,
        ax = ax,
    )
    ax.set_xlabel("RA quintile")
    ax.set_ylabel("Stay Duration (days)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hosp_stay_duration.by_RA.boxplot.png", dpi=300)


with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    # Box plot of the stay duration by quintile
    fig, ax = pylab.subplots(figsize=(3,3))
    sns.boxplot(
        data=selected_hosp1,
        x="RA_quintile",
        y="stay_duration",
        showfliers=False,
        color="white",
        linewidth=2,
        ax = ax,
    )
    ax.set_xlabel("RA quintile")
    ax.set_ylabel("Stay Duration (days)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hosp_stay_duration.by_RA.boxplot.png", dpi=300)


selected_hosp2['RA_quintile'] = pandas.qcut(selected_hosp2['RA'], numpy.linspace(0,1,6))
print("Death 30 days post discharge by RA quintile")
print(selected_hosp2.groupby("RA_quintile").death_in_30days.value_counts().unstack(1))

### Analysis of admission
admission_data = ukbb.loc[cohort, [
    'sex',
    'age_at_actigraphy',
    'age_at_death',
    'BMI',
    'smoking',
]].copy()
def first_admission_after_actigraphy(d):
    visit_after_actigraphy = d.age_at_visit > d.age_at_actigraphy
    if any(visit_after_actigraphy):
        return d[visit_after_actigraphy].age_at_visit.min()
    return float("NaN")
admission_data['age_at_admission'] = admission_data.index.map(pneumonia_hosp_after_actigraphy.groupby('eid').apply(first_admission_after_actigraphy))
admission_data['admission'] = (admission_data['age_at_admission'].isna() == False).astype(int)
admission_data['RA'] = admission_data.index.map(activity.acceleration_RA)
admission_data['acceleration_overall'] = admission_data.index.map(activity.acceleration_overall)
admission_data['smoking'] = admission_data.index.map(ukbb.smoking.isin(['Previous', 'Current']))
admission_data['has_med_of_interest'] = admission_data.index.map(has_med_of_interest).fillna(False)
#admission_data['censored'] = admission_data.age_at_death < admission_data.age_at_admission
admission_data['prior_pneumonia_flu_case'] = (admission_data.index.map(first_pneumonia_flu_diag) < admission_data.index.map(actigraphy_start))
admission_data['death_in_30days'] = admission_data.index.map(pneumonia_hosp_after_actigraphy.groupby('eid').death_in_30days.any()).fillna(False) # Using death after any pneumonia episode
admission_data['death_at_discharge'] = admission_data.index.map(pneumonia_hosp_after_actigraphy.groupby("eid").last().dismeth_uni.isin(HESIN_DISCHARGE_DEATH)).fillna(False)
admission_data['censored_date'] = hesin_cohort.epiend.max()
admission_data['censored_age'] = (admission_data.censored_date - admission_data.index.map(ukbb.birth_year_dt)) / YEAR
admission_data['event_age'] = admission_data[[
    'age_at_admission',
    'age_at_death',
    'censored_age',
]].min(axis=1)

# Hospital admission prop hazards model
selected_admission = admission_data[[
    "event_age",
    "age_at_actigraphy",
    "admission",
    "RA",
    "acceleration_overall",
    "sex",
    "smoking",
    "has_med_of_interest",
    "prior_pneumonia_flu_case",
    "death_in_30days",
    "death_at_discharge",
]].dropna(how="any", axis=0)
selected_admission['RA_quintile'] = pandas.qcut(selected_admission['RA'], numpy.linspace(0,1,6)).cat.rename_categories(
    ["1st", "2nd", "3rd", "4th", "5th"]
).astype(str).astype('category')
selected_admission['acceleration_overall_quintile'] = pandas.qcut(selected_admission['acceleration_overall'], numpy.linspace(0,1,6)).cat.rename_categories(
    ["1st", "2nd", "3rd", "4th", "5th"]
).astype(str).astype('category')

with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
    selected_admission_r = robjects.conversion.py2rpy(selected_admission.reset_index())
robjects.globalenv['selected_admission'] = selected_admission_r

# Fit the basic hospital admission model in R
# Test of HR of admission with pneumonia/flu diagnosis
rfit = robjects.r('''
res <- coxph(
    Surv(event_age, admission) ~ RA_quintile + age_at_actigraphy + smoking + acceleration_overall_quintile + has_med_of_interest + prior_pneumonia_flu_case + sex,
    data=selected_admission,
    id=id,
    cluster=id,
)
print(summary(res))
res
''')
summary = pandas.DataFrame(
    numpy.array(robjects.r('summary(res)$coefficients')),
    index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
    columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
)
print(f"Hosp. admission test: N = {len(selected_admission)}")
print(f"Hosp. admission test: N cases = {selected_admission.admission.sum()}")


# Plot the hazard ratios by RA quintile in hosp admissions
with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 14}):
    fig, ax = pylab.subplots(figsize=(3,3))
    xs = numpy.arange(5)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax.set_xlabel("RA quintile")
    ax.set_ylabel("Hazard ratio")
    ax.set_title("Hosp. Adm. for\nPneumonia/Flu")
    ax.set_xticks(xs)
    ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax.axhline(y=1, linestyle="--", color='k')
    ax.set_ylim(0,1.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "hosp_admission.hazard_ratios.by_RA_quintile.png", dpi=300)

with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    fig, (ax1,ax2,ax3) = pylab.subplots(figsize=(7,3.5), ncols=3, sharey=True)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax1.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax1.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax1.set_xlabel("RA quintile")
    ax1.set_ylabel("Hazard ratio")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax1.axhline(y=1, linestyle="--", color='k')
    #Plot the acceleration_overall values
    ys = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax2.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax2.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax2.set_xlabel("Average acceleration\nquintile")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax2.axhline(y=1, linestyle="--", color='k')
    # Covariates
    covariates = ['sexMale', 'smokingTRUE', 'has_med_of_interestTRUE', 'prior_pneumonia_flu_caseTRUE']
    covariate_labels = ['Male', 'Smoking ', 'Medication  ', 'Prior history  ']
    ys = numpy.array([
        summary.loc[f'{covariate}', 'coef'] for covariate in covariates
    ])
    ses = numpy.array(
        [summary.loc[f'{covariate}', 'robust se'] for covariate in covariates]
    )
    xs = numpy.arange(len(ys))
    ax3.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax3.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax3.set_xticks(xs)
    ax3.set_xticklabels(covariate_labels, rotation=90)
    ax3.axhline(y=1, linestyle="--", color='k')
    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    fig.suptitle("Hospital Admission for Pneumonia/Flu")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hosp_admission.hazard_ratios.by_RA_quintile.full.png", dpi=300)

## Hospital admission by RA quintile
admit_counts = selected_admission.groupby("RA_quintile").admission.value_counts().unstack(1)
admit_counts.columns = ['No', 'Yes']
print("Admission counts by RA quintile")
print(admit_counts)




# Fit the model for death after 30 days
rfit = robjects.r('''
res <- coxph(
    Surv(event_age, death_in_30days) ~ RA_quintile + age_at_actigraphy + smoking + acceleration_overall_quintile + has_med_of_interest + prior_pneumonia_flu_case + sex,
    data=selected_admission,
    id=id,
    cluster=id,
)
print(summary(res))
res
''')
summary = pandas.DataFrame(
    numpy.array(robjects.r('summary(res)$coefficients')),
    index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
    columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
)
print(f"Death after 30 days data: N = {len(selected_admission)}")
print(f"Death after 30 days data: N = {(selected_admission.death_in_30days.sum())}")


# Plot the hazard ratios by RA quintile in hosp admissions death at 30 days
with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 14}):
    fig, ax = pylab.subplots(figsize=(3,3))
    xs = numpy.arange(5)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax.set_xlabel("RA quintile")
    ax.set_ylabel("Hazard ratio")
    ax.set_title("Death in 30 days\npost discharge")
    ax.set_xticks(xs)
    ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax.set_ylim(0,1.3)
    ax.axhline(1, linestyle='--', color='k')
    fig.tight_layout()
    fig.savefig(OUTDIR / "death_in_30days_discharge.hazard_ratios.by_RA_quintile.png", dpi=300)

with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    fig, (ax1,ax2,ax3) = pylab.subplots(figsize=(7,3.5), ncols=3, sharey=True)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax1.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax1.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax1.set_xlabel("RA quintile")
    ax1.set_ylabel("Hazard ratio")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax1.axhline(y=1, linestyle="--", color='k')
    #Plot the acceleration_overall values
    ys = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax2.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax2.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax2.set_xlabel("Average acceleration\nquintile")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax2.axhline(y=1, linestyle="--", color='k')
    # Covariates
    covariates = ['sexMale', 'smokingTRUE', 'has_med_of_interestTRUE', 'prior_pneumonia_flu_caseTRUE']
    covariate_labels = ['Male', 'Smoking ', 'Medication  ', 'Prior history  ']
    ys = numpy.array([
        summary.loc[f'{covariate}', 'coef'] for covariate in covariates
    ])
    ses = numpy.array(
        [summary.loc[f'{covariate}', 'robust se'] for covariate in covariates]
    )
    xs = numpy.arange(len(ys))
    ax3.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax3.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax3.set_xticks(xs)
    ax3.set_xticklabels(covariate_labels, rotation=90)
    ax3.axhline(y=1, linestyle="--", color='k')
    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    fig.suptitle("Death within 30 days after Pneumonia/Flu")
    fig.tight_layout()
    fig.savefig(OUTDIR / "death_in_30days_discharge.hazard_ratios.by_RA_quintile.full.png", dpi=300)


# Fit the model for death at discharge
rfit = robjects.r('''
res <- coxph(
    Surv(event_age, death_at_discharge) ~ RA_quintile + age_at_actigraphy + smoking + acceleration_overall_quintile + has_med_of_interest + prior_pneumonia_flu_case + sex,
    data=selected_admission,
    id=id,
    cluster=id,
)
print(summary(res))
res
''')
summary = pandas.DataFrame(
    numpy.array(robjects.r('summary(res)$coefficients')),
    index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
    columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
)
print(f"Death at discharge total: N = {len(selected_admission)}")
print(f"Death at discharge deaths: N = {selected_admission.death_at_discharge.sum()}")


# Plot the hazard ratios by RA quintile in hosp admissions
with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 14}):
    fig, ax = pylab.subplots(figsize=(4,4))
    xs = numpy.arange(5)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax.set_xlabel("RA quintile")
    ax.set_ylabel("Hazard ratio")
    ax.set_title("Death at discharge")
    ax.set_xticks(xs)
    ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax.axhline(1.0, linestyle='--', color='k')
    ax.set_ylim(0,1.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "death_at_discharge.hazard_ratios.by_RA_quintile.png", dpi=300)


with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    fig, (ax1,ax2,ax3) = pylab.subplots(figsize=(7,3.5), ncols=3, sharey=True)
    ys = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'RA_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax1.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax1.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax1.set_xlabel("RA quintile")
    ax1.set_ylabel("Hazard ratio")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax1.axhline(y=1, linestyle="--", color='k')
    #Plot the acceleration_overall values
    ys = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'coef'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    xs = numpy.arange(len(ys))
    ses = numpy.array(
        [0] + [summary.loc[f'acceleration_overall_quintile{quintile}', 'robust se'] for quintile in ['2nd' ,'3rd', '4th', '5th']]
    )
    ax2.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax2.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax2.set_xlabel("Average acceleration\nquintile")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
    ax2.axhline(y=1, linestyle="--", color='k')
    # Covariates
    covariates = ['sexMale', 'smokingTRUE', 'has_med_of_interestTRUE', 'prior_pneumonia_flu_caseTRUE']
    covariate_labels = ['Male', 'Smoking ', 'Medication  ', 'Prior history  ']
    ys = numpy.array([
        summary.loc[f'{covariate}', 'coef'] for covariate in covariates
    ])
    ses = numpy.array(
        [summary.loc[f'{covariate}', 'robust se'] for covariate in covariates]
    )
    xs = numpy.arange(len(ys))
    ax3.scatter(xs, numpy.exp(ys), color='k')
    for x, y, se in zip(xs, ys, ses):
        ax3.plot([x, x], [numpy.exp(y - 1.96*se), numpy.exp(y + 1.96*se)], color='k')
    ax3.set_xticks(xs)
    ax3.set_xticklabels(covariate_labels, rotation=90)
    ax3.axhline(y=1, linestyle="--", color='k')
    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    fig.suptitle("Death in Hospital for Pneumonia/Flu")
    fig.tight_layout()
    fig.savefig(OUTDIR / "death_at_discharge.hazard_ratios.by_RA_quintile.full.png", dpi=300)

    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax1.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)


# RA score histogram
with matplotlib.rc_context({"font.sans-serif": ["Arial"], "font.size": 12}):
    fig, ax = pylab.subplots(figsize=(3,3))
    sns.kdeplot(
        data = selected_admission,
        x = "RA",
        ax =ax,
        shade=True,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "RA_distribution.png", dpi=300)

# Demographics description
print("Demographics Table")
hosp_visit_time = selected_hosp1.age_at_visit - selected_hosp1.age_at_actigraphy
print(pandas.DataFrame({
    "Cohort": {
        "N": len(selected_admission),
        "Male": f"{(selected_admission.sex=='Male').mean():0.1%}",
        "Smoking": f"{selected_admission.smoking.mean():0.1%}",
        "Medication": f"{selected_admission.has_med_of_interest.mean():0.1%}",
        "Prior Case": f"{selected_admission.prior_pneumonia_flu_case.mean():0.1%}",
        "Age at Actigraphy (mean±SD)": f"{selected_admission.age_at_actigraphy.mean():0.1f}±{selected_admission.age_at_actigraphy.std():0.1f}",
        "Death in 30 Days": "N/A",
        "Age at Hosp. (mean±SD)": "N/A",
        "Time to Hosp. (mean±SD years)": "N/A",
    },
    "Hospitalized": {
        "N": len(selected_hosp1),
        "Male": f"{(selected_hosp1.sex=='Male').mean():0.1%}",
        "Smoking": f"{selected_hosp1.smoking.mean():0.1%}",
        "Medication": f"{selected_hosp1.has_med_of_interest.mean():0.1%}",
        "Prior Case": f"{selected_hosp1.prior_pneumonia_flu_case.mean():0.1%}",
        "Age at Actigraphy (mean±SD)": f"{selected_hosp1.age_at_actigraphy.mean():0.1f}±{selected_hosp1.age_at_actigraphy.std():0.1f}",
        "Death in 30 Days": f"{selected_hosp1.death_in_30days.mean():0.1%}",
        "Age at Hosp. (mean±SD)": f"{selected_hosp1.age_at_visit.mean():0.1f}±{selected_hosp1.age_at_visit.std():0.1f}",
        "Time to Hosp. (mean±SD years)": f"{hosp_visit_time.mean():0.1f}±{hosp_visit_time.std():0.1f}",
    },
}))


# Demographics by RA quintile
demographics_by_RA_quintile = selected_admission.groupby("RA_quintile").apply(lambda d: pandas.Series({
        "N": len(d),
        "Male": f"{(d.sex=='Male').mean():0.1%}",
        "Smoking": f"{d.smoking.mean():0.1%}",
        "Medication": f"{d.has_med_of_interest.mean():0.1%}",
        "Prior Case": f"{d.prior_pneumonia_flu_case.mean():0.1%}",
        "Age at Actigraphy (mean±SD)": f"{d.age_at_actigraphy.mean():0.1f}±{d.age_at_actigraphy.std():0.1f}",
        "Mean Acceleration (mean±SD)": f"{d.acceleration_overall.mean():0.1f}±{d.acceleration_overall.std():0.1f}",
})).T
print(demographics_by_RA_quintile)
demographics_by_RA_quintile.to_csv(OUTDIR / "demographics_by_RA_quintile.txt", sep="\t", encoding="utf-8-sig")


## Determine the flow diagram values
full_cohort = ukbb.index[~ukbb.birth_year.isna()]
print(f"Full UKB cohort: {len(full_cohort)}")
with_actigraphy = ukbb.index[~ukbb.birth_year.isna() & (~ukbb.actigraphy_start_date.isna())]
print(f"With actigraphy data {len(with_actigraphy)}")
calibrated_actigraphy = with_actigraphy.intersection(activity.index)
print(f"With calibrated data {len(calibrated_actigraphy)}")
good_RA = activity.loc[calibrated_actigraphy].query("acceleration_RA == acceleration_RA and acceleration_overall == acceleration_overall").index
print(f"With good RA score {len(good_RA)} (removed {len(calibrated_actigraphy.difference(good_RA))})")
print(f"Admissions cohort {len(selected_admission)} (same as good RA cohort)")
print(f"Num individuals admitted {selected_admission.admission.sum()}")
print(f"Hospitalized cohort {(selected_hosp).eid.nunique()} patients (same as above), with {len(selected_hosp)} hospitalizations")
print(f"Number of deaths at discharge {selected_admission.death_at_discharge.sum()}")
print(f"Number of deaths within 30 days post discharge {selected_admission.death_in_30days.sum()}")
print(f"ICU cohort {selected_ICU.eid.nunique()} with {len(selected_ICU)} ICU visits")

log_file.close()