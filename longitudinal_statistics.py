import warnings

import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm

from util import BH_FDR

COVARIATES = ["sex", "ethnicity_white", "overall_health_good", "smoking_ever", "age_at_actigraphy", "BMI", "college_education", "alcohol", "townsend_deprivation_index"]
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
    qs[okay] = BH_FDR(ps[okay])
    return qs

def predictive_tests_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy
    if not RECOMPUTE:
        try:
            predictive_tests_cox = pandas.read_csv(OUTDIR/f"predictive_tests.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable = 'temp_RA'
    variable_SD = data[variable].std()

    covariate_formula = " + ".join(COVARIATES)

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()

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

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['event_age', "censored", 'case_status', 'index', 'event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

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
            rfit = robjects.r('''
            res <- coxph(
                Surv(age_at_actigraphy, event_age, event) ~ temp_RA + BMI + age_at_actigraphy + strata(sex, overall_health_good) + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
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
            'p': summary.loc['temp_RA_1:2', 'Pr(>|z|)'],
            'logHR': summary.loc['temp_RA_1:2', 'coef'],
            'logHR_se': summary.loc['temp_RA_1:2', 'robust se'],
            'std_logHR': summary.loc['temp_RA_1:2', 'coef'] * variable_SD,
            'std_logHR_se': summary.loc['temp_RA_1:2', 'robust se'] * variable_SD,

        })
        predictive_tests_cox_list.append(header)
    predictive_tests_cox = pandas.DataFrame(predictive_tests_cox_list)
    predictive_tests_cox['q'] = bh_fdr_with_nans(predictive_tests_cox.p.fillna(1))
    predictive_tests_cox.sort_values(by="p").to_csv(OUTDIR / "predictive_tests.cox.txt", sep="\t", index=False)

    return predictive_tests_cox

def predictive_tests_by_sex_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy, separte by male and female
    if not RECOMPUTE:
        try:
            predictive_tests_by_sex_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_sex.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_sex_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable = 'temp_RA'
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()

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

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['event_age', "censored", 'case_status', 'index', 'event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

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
        # Stratified by sex and allows a sex-interaction term with temp_RA (no need for sex term due to strata)
        try:
            rfit = robjects.r('''
            res <- coxph(
                Surv(age_at_actigraphy, event_age, event) ~ (temp_RA + BMI + age_at_actigraphy + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index) * sex - sex + strata(sex, overall_health_good) ,
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
        male_var = f"{variable}:sexMale_1:2"
        female_var = f"{variable}_1:2"
        # Contrast for the male-vs-0 test:
        # linear combination of temp_RA and temp_RA:sexMale since female is the baseline
        contrast = pandas.Series(numpy.zeros(len(summary)), summary.index)
        contrast[male_var] = 1
        contrast[female_var] = 1
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            contrast = robjects.conversion.py2rpy(contrast) # move into R
        robjects.globalenv['contrast'] = contrast
        wt = robjects.r("wt <- wald.test(coef=coef(res), vcov=vcov(res), contrast=contrast)")
        header.update({
            'sex_diff_p': summary.loc['temp_RA:sexMale_1:2', 'Pr(>|z|)'],
            'female_p': summary.loc['temp_RA_1:2', 'Pr(>|z|)'],
            'male_p': wt[2][0], # Results from the contrast
            'female_logHR': summary.loc['temp_RA_1:2', 'coef'],
            'male_logHR': wt[4][0],
            'female_logHR_se': summary.loc['temp_RA_1:2', 'robust se'],
            'male_logHR_se': wt[4][1],
            'std_female_logHR': summary.loc['temp_RA_1:2', 'coef'] * variable_SD,
            'std_male_logHR': wt[4][0] * variable_SD,
            'std_female_logHR_se': summary.loc['temp_RA_1:2', 'robust se'] * variable_SD,
            'std_male_logHR_se': wt[4][1] * variable_SD,
        })
        predictive_tests_by_sex_cox_list.append(header)

    predictive_tests_by_sex_cox = pandas.DataFrame(predictive_tests_by_sex_cox_list)
    predictive_tests_by_sex_cox['sex_diff_q'] = bh_fdr_with_nans(predictive_tests_by_sex_cox.sex_diff_p.fillna(1))
    predictive_tests_by_sex_cox.sort_values(by="sex_diff_p").to_csv(OUTDIR / "predictive_tests_by_sex.cox.txt", sep="\t", index=False)

    return predictive_tests_by_sex_cox

def predictive_tests_by_age_cox(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy, separating by age at which actigraphy was recorded
    if not RECOMPUTE:
        try:
            predictive_tests_by_age_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_age.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_age_cox
        except FileNotFoundError:
            pass

    last_date = case_status.first_date.max()
    variable = 'temp_RA'
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()

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

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['event_age', "censored", 'case_status', 'index', 'event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

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
        # Include an age_at_actigraphy interaction term with temp_RA
        # Don't include it for the other factors since they were not assessed at that time so it makes little sense
        try:
            rfit = robjects.r('''
            res <- coxph(
                Surv(age_at_actigraphy, event_age, event) ~ temp_RA*age_at_actigraphy + BMI + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index + strata(sex, overall_health_good) ,
                data=d2,
                id=ID,
                cluster=ID,
            )
            res
            ''')
        except rpy2.rinterface.RRuntimeError as e:
            print(f"Problem in {variable} {diagnosis}:\n\t{e}")
            predictive_tests_by_age.append(header)
            continue
        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
        )
        # Contrasts for the age55-vs-0 and age70-vs-0 tests:
        # linear combination of temp_RA and temp_RA:age_at_actigraphy 
        contrast_age55 = pandas.Series(numpy.zeros(len(summary)), summary.index)
        contrast_age55["temp_RA_1:2"] = 1
        contrast_age55["temp_RA:age_at_actigraphy_1:2"] = 55
        contrast_age70 = pandas.Series(numpy.zeros(len(summary)), summary.index)
        contrast_age70["temp_RA_1:2"] = 1
        contrast_age70["temp_RA:age_at_actigraphy_1:2"] = 70
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            contrast_age55 = robjects.conversion.py2rpy(contrast_age55) # move into R
            contrast_age70 = robjects.conversion.py2rpy(contrast_age70) # move into R
        robjects.globalenv['contrast_age55'] = contrast_age55
        robjects.globalenv['contrast_age70'] = contrast_age70
        wt_age55 = robjects.r("wt_age55 <- wald.test(coef=coef(res), vcov=vcov(res), contrast=contrast_age55)")
        wt_age70 = robjects.r("wt_age70 <- wald.test(coef=coef(res), vcov=vcov(res), contrast=contrast_age70)")
        header.update({
            "age_diff_p": summary.loc['temp_RA:age_at_actigraphy_1:2', 'Pr(>|z|)'],
            "age55_p": wt_age55[2][0],
            "age70_p": wt_age70[2][0],
            "age55_std_logHR": wt_age55[4][0] * variable_SD,
            "age70_std_logHR": wt_age70[4][0] * variable_SD,
            "age55_std_logHR_se": wt_age55[4][1] * variable_SD,
            "age70_std_logHR_se": wt_age70[4][1] * variable_SD,
        })
        predictive_tests_by_age_cox_list.append(header)

    predictive_tests_by_age_cox = pandas.DataFrame(predictive_tests_by_age_cox_list)
    predictive_tests_by_age_cox['age_diff_q'] = bh_fdr_with_nans(predictive_tests_by_age_cox.age_diff_p.fillna(1))
    predictive_tests_by_age_cox.sort_values(by="age_diff_p").to_csv(OUTDIR / "predictive_tests_by_age.cox.txt", sep="\t", index=False)

    return predictive_tests_by_age_cox