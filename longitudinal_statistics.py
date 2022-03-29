import warnings

import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm

import util

COVARIATES = ["sex", "ethnicity_white", "overall_health_good", "smoking_ever", "age_at_actigraphy_cat", "BMI", "college_education", "alcohol", "townsend_deprivation_index"]
MIN_N = 200
MIN_N_BY_AGE = 400
MIN_N_PER_SEX = 200

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages
from rpy2.robjects import pandas2ri
robjects.r("memory.limit(10000)")

utils = robjects.packages.importr("utils")
packnames = ['survival', 'timereg']
names_to_install = [x for x in packnames if not robjects.packages.isinstalled(x)]
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
if len(names_to_install) > 0:
    utils.install_packages(robjects.vectors.StrVector(names_to_install))
survival = robjects.packages.importr('survival')
timereg = robjects.packages.importr('timereg')


def bh_fdr_with_nans(ps):
    okay = ~ps.isna()
    qs = numpy.full(fill_value=float("NaN"), shape=ps.shape)
    qs[okay] = util.BH_FDR(ps[okay])
    return qs

def predictive_tests_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False, variable="temp_amplitude"):
    # Predict diagnoses after actigraphy
    if not RECOMPUTE:
        try:
            predictive_tests_cox = pandas.read_csv(OUTDIR/f"predictive_tests.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()

    predictive_tests_cox_list = []
    print("Starting longitudinal tests")
    for diagnosis, diagnoses in case_status.groupby("PHECODE"):
        print(diagnosis)
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(d.date_of_death.fillna(last_date), inplace=True)
        d['event_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded
        d['event'] = 'censored' # censored/diagnosed/death for event type
        d.loc[~d.censored, 'event'] = 'diagnosed'
        d.loc[d.date_of_death == d.diagnosis_date, 'event'] = 'death'
        d['event'] = pandas.Categorical(d.event, categories = ['censored', 'diagnosed', 'death'])
        d['time_to_event'] = d.event_age - d.age_at_actigraphy

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['event_age', "censored", 'case_status', 'index', 'event', 'time_to_event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases": (d2.case_status == 'case').sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }

        if header['N_cases'] < MIN_N:
            continue

        # Run it in R
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            d2_r = robjects.conversion.py2rpy(d2)
        robjects.globalenv['d2'] = d2_r

        # First, we have the standard base model
        try:
            rfit = robjects.r(f'''
            res <- coxph(
                Surv(time_to_event, event == 'diagnosed') ~ {variable} + BMI + age_at_actigraphy_cat + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
                data=d2,
                id=ID,
                cluster=ID,
            )
            res
            ''')
        except rpy2.rinterface.RRuntimeError as e:
            print(f"Problem in {variable} {diagnosis}:\n\t{e}")
            predictive_tests_cox_list.append(header)
            continue

        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
        )
        header.update({
            'p': summary.loc[variable, 'Pr(>|z|)'],
            'logHR': summary.loc[variable, 'coef'],
            'logHR_se': summary.loc[variable, 'robust se'],
            'std_logHR': summary.loc[variable, 'coef'] * variable_SD,
            'std_logHR_se': summary.loc[variable, 'robust se'] * variable_SD,
        })
        predictive_tests_cox_list.append(header)
    predictive_tests_cox = pandas.DataFrame(predictive_tests_cox_list)
    predictive_tests_cox['q'] = bh_fdr_with_nans(predictive_tests_cox.p.fillna(1))
    predictive_tests_cox.sort_values(by="p").to_csv(OUTDIR / "predictive_tests.cox.txt", sep="\t", index=False)

    return predictive_tests_cox

def predictive_tests_by_sex_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False, variable='temp_amplitude'):
    # Predict diagnoses after actigraphy, separte by male and female
    if not RECOMPUTE:
        try:
            predictive_tests_by_sex_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_sex.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_sex_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()

    predictive_tests_by_sex_cox_list = []
    print("Starting sex-specific longitudinal tests")
    for diagnosis, diagnoses in case_status.groupby("PHECODE"):
        print(diagnosis)
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(d.date_of_death.fillna(last_date), inplace=True)
        d['event_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded
        d['event'] = 'censored' # censored/diagnosed/death for event type
        d.loc[~d.censored, 'event'] = 'diagnosed'
        d.loc[d.date_of_death == d.diagnosis_date, 'event'] = 'death'
        d['event'] = pandas.Categorical(d.event, categories = ['censored', 'diagnosed', 'death'])
        d['time_to_event'] = d.event_age - d.age_at_actigraphy

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['case_status', 'index', 'event', 'time_to_event', 'age_at_actigraphy_cat', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases_male": ((d2.sex == 'Male') & (d2.case_status == 'case')).sum(),
            "N_cases_female": ((d2.sex == 'Female') & (d2.case_status == 'case')).sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }
        if header['N_cases_male'] < MIN_N_PER_SEX or header['N_cases_female'] < MIN_N_PER_SEX:
            continue

        # Run it in R
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            d2_r = robjects.conversion.py2rpy(d2)
        robjects.globalenv['d2'] = d2_r

        # Sex-interaction effect model
        # Stratified by sex and allows a sex-interaction term with temp_amplitude (no need for sex term due to strata)
        try:
            rfit = robjects.r(f'''
            res <- coxph(
                Surv(time_to_event, event == 'diagnosed') ~ ({variable} + BMI + age_at_actigraphy_cat + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index) * sex,
                data=d2,
                id=ID,
                cluster=ID,
            )
            res
            ''')
        except rpy2.rinterface.RRuntimeError as e:
            print(f"Problem in {variable} {diagnosis}:\n\t{e}")
            predictive_tests_by_sex_cox_list.append(header)
            continue
        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
        )
        male_var = f"{variable}:sexMale"
        female_var = f"{variable}"
        # Contrast for the male-vs-0 test:
        # linear combination of temp_amplitude and temp_amplitude:sexMale since female is the baseline
        contrast = pandas.Series(numpy.zeros(len(summary)), summary.index)
        contrast[male_var] = 1
        contrast[female_var] = 1
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            contrast = robjects.conversion.py2rpy(contrast) # move into R
        robjects.globalenv['contrast'] = contrast
        wt = robjects.r("wt <- wald.test(coef=coef(res), vcov=vcov(res), contrast=contrast)")
        header.update({
            'sex_diff_p': summary.loc['temp_amplitude:sexMale', 'Pr(>|z|)'],
            'female_p': summary.loc['temp_amplitude', 'Pr(>|z|)'],
            'male_p': wt[2][0], # Results from the contrast
            'female_logHR': summary.loc['temp_amplitude', 'coef'],
            'male_logHR': wt[4][0],
            'female_logHR_se': summary.loc['temp_amplitude', 'robust se'],
            'male_logHR_se': wt[4][1],
            'std_female_logHR': summary.loc['temp_amplitude', 'coef'] * variable_SD,
            'std_male_logHR': wt[4][0] * variable_SD,
            'std_female_logHR_se': summary.loc['temp_amplitude', 'robust se'] * variable_SD,
            'std_male_logHR_se': wt[4][1] * variable_SD,
        })
        predictive_tests_by_sex_cox_list.append(header)

    predictive_tests_by_sex_cox = pandas.DataFrame(predictive_tests_by_sex_cox_list)
    predictive_tests_by_sex_cox['sex_diff_q'] = bh_fdr_with_nans(predictive_tests_by_sex_cox.sex_diff_p.fillna(1))
    predictive_tests_by_sex_cox.sort_values(by="sex_diff_p").to_csv(OUTDIR / "predictive_tests_by_sex.cox.txt", sep="\t", index=False)

    return predictive_tests_by_sex_cox

def predictive_tests_by_age_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False, variable="temp_amplitude"):
    # Predict diagnoses after actigraphy, separating by age at which actigraphy was recorded
    if not RECOMPUTE:
        try:
            predictive_tests_by_age_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_age.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_age_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()
    d['age_at_actigraphy_cat'] = d.age_at_actigraphy_cat.astype(str) # The Interval objects don't play with R

    predictive_tests_by_age_cox_list = []
    print("Starting sex-specific longitudinal tests")
    for diagnosis, diagnoses in case_status.groupby("PHECODE"):
        print(diagnosis)
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(d.date_of_death.fillna(last_date), inplace=True)
        d['event_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded
        d['event'] = 'censored' # censored/diagnosed/death for event type
        d.loc[~d.censored, 'event'] = 'diagnosed'
        d.loc[d.date_of_death == d.diagnosis_date, 'event'] = 'death'
        d['event'] = pandas.Categorical(d.event, categories = ['censored', 'diagnosed', 'death'])
        d['time_to_event'] = d.event_age - d.age_at_actigraphy

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['time_to_event', 'case_status', 'index', 'event', 'time_to_event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases": (d2.case_status == 'case').sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }
        if header['N_cases'] < MIN_N_BY_AGE:
            continue

        # Run it in R
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            d2_r = robjects.conversion.py2rpy(d2)
        robjects.globalenv['d2'] = d2_r

        # Age-interaction effect model
        # Include an age_at_actigraphy interaction term with temp_amplitude 
        try:
            rfit = robjects.r(f'''
            res <- coxph(
                Surv(time_to_event, event == 'diagnosed') ~ ({variable} + BMI + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index) * age_at_actigraphy_cat -age_at_actigraphy_cat + strata(age_at_actigraphy_cat),
                data=d2,
                id=ID,
                cluster=ID,
            )
            res
            ''')
        except rpy2.rinterface.embedded.RRuntimeError as e:
            print(f"Problem in {variable} {diagnosis}:\n\t{e}")
            predictive_tests_by_age_cox_list.append(header)
            continue
        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
        )
        # Contrast for the age <65 category, since this is relative to the >65 category in our model
        contrast_under65 = pandas.Series(numpy.zeros(len(summary)), summary.index)
        contrast_under65["temp_amplitude"] = 1
        contrast_under65["temp_amplitude:age_at_actigraphy_catunder_65"] = 1
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            contrast_under65 = robjects.conversion.py2rpy(contrast_under65) # move into R
        robjects.globalenv['contrast_under65'] = contrast_under65
        wt_under65 = robjects.r("wt_age55 <- wald.test(coef=coef(res), vcov=vcov(res), contrast=contrast_under65)")
        header.update({
            "age_diff_p": summary.loc['temp_amplitude:age_at_actigraphy_catunder_65', 'Pr(>|z|)'],
            "over_65_p": summary.loc['temp_amplitude', 'Pr(>|z|)'],
            "under_65_p": wt_under65[2][0],
            "over_65_std_logHR": summary.loc['temp_amplitude', 'coef'] * variable_SD,
            "under_65_std_logHR": wt_under65[4][0] * variable_SD,
            "over_65_std_logHR_se": summary.loc['temp_amplitude', 'robust se'] * variable_SD,
            "under_65_std_logHR_se": wt_under65[4][1] * variable_SD,
        })
        predictive_tests_by_age_cox_list.append(header)

    predictive_tests_by_age_cox = pandas.DataFrame(predictive_tests_by_age_cox_list)
    predictive_tests_by_age_cox['age_diff_q'] = bh_fdr_with_nans(predictive_tests_by_age_cox.age_diff_p.fillna(1))
    predictive_tests_by_age_cox.sort_values(by="age_diff_p").to_csv(OUTDIR / "predictive_tests_by_age.cox.txt", sep="\t", index=False)

    return predictive_tests_by_age_cox

def survival_association(data, OUTDIR, RECOMPUTE=False, variable="temp_amplitude"):
    last_date = data.date_of_death.max()
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()

    predictive_tests_cox_list = []
    print(f"Running survival test for {variable}")
    d['event_age'] = (d.date_of_death - d.birth_year_dt) / pandas.to_timedelta("1Y")
    d['status'] = ~d.event_age.isna()
    d.event_age.fillna((last_date - d.birth_year_dt) / pandas.to_timedelta("1Y"), inplace=True)
    d['time_to_event'] = d.event_age - d.age_at_actigraphy

    # Final collection of data we use for the models
    d2 = d.reset_index()[['index', 'status', 'time_to_event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

    results = {
        "activity_var": variable,
        "phecode": "death",
        "N_cases": (d2.status).sum(),
        "N_controls": (d2.status == False).sum(),
    }

    # Run it in R
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        d2_r = robjects.conversion.py2rpy(d2)
    robjects.globalenv['d2'] = d2_r

    # We have the standard base model
    rfit = robjects.r(f'''
    res <- coxph(
        Surv(time_to_event, status) ~ {variable} + BMI + age_at_actigraphy_cat + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
        data=d2,
        id=ID,
        cluster=ID,
    )
    res
    ''')

    summary = pandas.DataFrame(
        numpy.array(robjects.r('summary(res)$coefficients')),
        index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
        columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
    )
    results.update({
        'p': summary.loc[variable, 'Pr(>|z|)'],
        'logHR': summary.loc[variable, 'coef'],
        'logHR_se': summary.loc[variable, 'robust se'],
        'std_logHR': summary.loc[variable, 'coef'] * variable_SD,
        'std_logHR_se': summary.loc[variable, 'robust se'] * variable_SD,
    })

    # Not super informative, won't use for now
    ## Generate a survival plot
    #first, second, third, fourth, fifth = d2[variable].quantile([0.1,0.3,0.5,0.7,0.9])
    #robjects.r(f'''
    #dummy <- expand.grid(
    #    {variable} = c({first}, {second}, {third}, {fourth}, {fifth}),
    #    age_at_actigraphy_cat = 'under_65',
    #    ethnicity_white = TRUE,
    #    overall_health_good = 1,
    #    sex = 'Female',
    #    smoking_ever = 0,
    #    BMI = {d2.BMI.mean()},
    #    college_education = 0,
    #    alcohol = 'often',
    #    townsend_deprivation_index = {d2.townsend_deprivation_index.mean()}
    #)
    #fit <- survfit(res, newdata=dummy)
    #plot(fit, col=1:9, ymin=0.9)
    #print(dummy)
    #''')

    return results
