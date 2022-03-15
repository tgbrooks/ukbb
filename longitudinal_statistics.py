import warnings

import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm

from util import BH_FDR

COVARIATES = ["sex", "ethnicity_white", "overall_health_good", "high_income", "smoking_ever", "age_at_actigraphy", "BMI", "college_education"]
MIN_N = 200
MIN_N_BY_AGE = 400
MIN_N_PER_SEX = 200

def bh_fdr_with_nans(ps):
    okay = ~ps.isna()
    qs = numpy.full(fill_value=float("NaN"), shape=ps.shape)
    qs[okay] = BH_FDR(ps[okay])
    return qs

def predictive_tests_cox(data, phecode_info, longitudinal_diagnoses, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy
    if not RECOMPUTE:
        try:
            predictive_tests_cox = pandas.read_csv(OUTDIR/f"predictive_tests.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_cox
        except FileNotFoundError:
            pass

    variable = "temp_RA"
    last_date = longitudinal_diagnoses.first_date.max()

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()
    date_of_death = pandas.to_datetime(d.date_of_death)

    covariate_formula = " + ".join(COVARIATES)

    predictive_tests_cox_list = []
    for diagnosis, diagnoses in longitudinal_diagnoses.groupby("PHECODE"):
        print(diagnosis)
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(date_of_death.fillna(last_date), inplace=True)
        d['diagnosis_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded

        d2 = d[d.use][['diagnosis_age', "censored", 'case_status', variable] + COVARIATES].dropna(how="any")
        model = smf.phreg(
            f"diagnosis_age ~ {variable} + {covariate_formula}",
            data = d2,
            status = ~d2.censored.values,
            entry = d2.age_at_actigraphy.values,
            )
        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases": (d2.case_status == 'case').sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }
        if header['N_cases'] < MIN_N:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("error") # warnings as exceptions
            try:
                fit = model.fit() # Run the model fit
            except (numpy.linalg.LinAlgError, sm.tools.sm_exceptions.ConvergenceWarning, RuntimeWarning) as e:
                print(f"Problem in {variable} {diagnosis}: {e}")
                predictive_tests_cox_list.append(header)
                continue
        pvalues = pandas.Series(fit.pvalues, model.exog_names)
        params = pandas.Series(fit.params, model.exog_names)
        se = pandas.Series(fit.bse, model.exog_names)
        std = d2[variable].std()
        header.update({
            "p": pvalues[variable],
            "logHR": params[variable],
            "logHR_se": se[variable],
            "std_logHR": params[variable] * std,
            "std_logHR_se": se[variable] *std,
        })
        predictive_tests_cox_list.append(header)
    predictive_tests_cox = pandas.DataFrame(predictive_tests_cox_list)
    predictive_tests_cox['q'] = bh_fdr_with_nans(predictive_tests_cox.p.fillna(1))
    predictive_tests_cox.sort_values(by="p").to_csv(OUTDIR / "predictive_tests.cox.txt", sep="\t", index=False)

    return predictive_tests_cox

def predictive_tests_by_sex_cox(data, phecode_info, longitudinal_diagnoses, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy, separte by male and female
    if not RECOMPUTE:
        try:
            predictive_tests_by_sex_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_sex.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_sex_cox
        except FileNotFoundError:
            pass

    variable = "temp_RA"
    last_date = longitudinal_diagnoses.first_date.max()

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()
    date_of_death = pandas.to_datetime(d.date_of_death)

    covariate_formula = " + ".join(COVARIATES)

    predictive_tests_by_sex_cox_list = []
    for diagnosis, diagnoses in longitudinal_diagnoses.groupby("PHECODE"):
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(date_of_death.fillna(last_date), inplace=True)
        d['diagnosis_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded

        d2 = d[d.use][['diagnosis_age', "censored", 'case_status', variable] + COVARIATES].dropna(how="any")
        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases_male": ((d2.sex == 'Male') & (d2.case_status == 'case')).sum(),
            "N_cases_female": ((d2.sex == 'Female') & (d2.case_status == 'case')).sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }
        print(diagnosis)
        if header['N_cases_male'] < MIN_N_PER_SEX or header['N_cases_female'] < MIN_N_PER_SEX:
            continue
        model = smf.phreg(
            f"diagnosis_age ~ sex:({variable} + {covariate_formula})",
            data = d2,
            status = ~d2.censored.values,
            entry = d2.age_at_actigraphy.values,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("error") # warnings as exceptions
            try:
                fit = model.fit() # Run the model fit
            except (numpy.linalg.LinAlgError, sm.tools.sm_exceptions.ConvergenceWarning, RuntimeWarning) as e:
                print(f"Problem in {variable} {diagnosis}: {e}")
                predictive_tests_by_sex_cox_list.append(header)
                continue
        pvalues = pandas.Series(fit.pvalues, model.exog_names)
        params = pandas.Series(fit.params, model.exog_names)
        se = pandas.Series(fit.bse, model.exog_names)
        std = d2[variable].std()
        male_var = f"sex[Male]:{variable}"
        female_var = f"sex[Female]:{variable}"
        contrast = pandas.Series(numpy.zeros(params.shape), model.exog_names)
        contrast[male_var] = 1
        contrast[female_var] = -1
        sex_diff_p = float(fit.f_test(contrast).pvalue)
        header.update({
            "sex_diff_p": sex_diff_p,
            "male_p": pvalues[male_var],
            "female_p": pvalues[female_var],
            "male_logHR": params[male_var],
            "female_logHR": params[female_var],
            "male_logHR_se": se[male_var],
            "female_logHR_se": se[female_var],
            "male_std_logHR": params[male_var] * std,
            "female_std_logHR": params[female_var] * std,
            "male_std_logHR_se": se[male_var] *std,
            "female_std_logHR_se": se[female_var] *std,
        })
        predictive_tests_by_sex_cox_list.append(header)

    predictive_tests_by_sex_cox = pandas.DataFrame(predictive_tests_by_sex_cox_list)
    predictive_tests_by_sex_cox['sex_diff_q'] = bh_fdr_with_nans(predictive_tests_by_sex_cox.sex_diff_p.fillna(1))
    predictive_tests_by_sex_cox.sort_values(by="sex_diff_p").to_csv(OUTDIR / "predictive_tests_by_sex.cox.txt", sep="\t", index=False)

    return predictive_tests_by_sex_cox

def predictive_tests_by_age_cox(data, phecode_info, longitudinal_diagnoses, OUTDIR, RECOMPUTE=False):
    # Predict diagnoses after actigraphy, separating by age at which actigraphy was recorded
    if not RECOMPUTE:
        try:
            predictive_tests_by_age_cox = pandas.read_csv(OUTDIR/f"predictive_tests_by_age.cox.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_by_age_cox
        except FileNotFoundError:
            pass

    variable = "temp_RA"
    last_date = longitudinal_diagnoses.first_date.max()

    d = data[[variable, 'date_of_death', 'birth_year_dt'] + COVARIATES].copy()
    date_of_death = pandas.to_datetime(d.date_of_death)

    covariate_formula = " + ".join(COVARIATES)

    predictive_tests_by_age_cox_list = []
    for diagnosis, diagnoses in longitudinal_diagnoses.groupby("PHECODE"):
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['censored'] = d.case_status != 'case'
        d.diagnosis_date.fillna(date_of_death.fillna(last_date), inplace=True)
        d['diagnosis_age'] = (d.diagnosis_date - d.birth_year_dt) / pandas.to_timedelta("1Y")
        d['use'] = d.case_status.isin(['case', 'control']) # Drop anyone excluded

        d2 = d[d.use][['diagnosis_age', "censored", 'case_status', variable] + COVARIATES].dropna(how="any")
        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases": (d2.case_status == 'case').sum(),
            "N_controls": (d2.censored).sum(),
        }
        if header['N_cases'] < MIN_N_BY_AGE:
            continue
        print(diagnosis)
        model = smf.phreg(
            f"diagnosis_age ~ age_at_actigraphy*({variable} + {covariate_formula})",
            data = d2,
            status = ~d2.censored.values,
            entry = d2.age_at_actigraphy.values,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("error") # warnings as exceptions
            try:
                fit = model.fit() # Run the model fit
            except (numpy.linalg.LinAlgError, sm.tools.sm_exceptions.ConvergenceWarning, RuntimeWarning) as e:
                print(f"Problem in {variable} {diagnosis}: {e}")
                predictive_tests_by_age_cox_list.append(header)
                continue
        pvalues = pandas.Series(fit.pvalues, model.exog_names)
        params = pandas.Series(fit.params, model.exog_names)
        cov_params = pandas.DataFrame(fit.cov_params(), index=params.index, columns=params.index)
        std = d2[variable].std()
        interaction = f"age_at_actigraphy:{variable}"
        age55_vec = pandas.Series(numpy.zeros(len(params)), index=params.index)
        age55_vec[variable] = 1
        age55_vec[interaction] = 55
        age70_vec = pandas.Series(numpy.zeros(len(params)), index=params.index)
        age70_vec[variable] = 1
        age70_vec[interaction] = 70
        age_diff_p = pvalues[interaction]
        header.update({
            "age_diff_p": age_diff_p,
            "age55_std_logHR": (params @ age55_vec) * std,
            "age70_std_logHR": (params @ age70_vec) * std,
            "age55_std_logHR_se": numpy.sqrt(age55_vec.T @ cov_params @ age55_vec) * std,
            "age70_std_logHR_se": numpy.sqrt(age70_vec.T @ cov_params @ age70_vec) * std,
            "age55_p": fit.f_test(age55_vec).pvalue,
            "age70_p": fit.f_test(age70_vec).pvalue,
        })
        predictive_tests_by_age_cox_list.append(header)

    predictive_tests_by_age_cox = pandas.DataFrame(predictive_tests_by_age_cox_list)
    predictive_tests_by_age_cox['age_diff_q'] = bh_fdr_with_nans(predictive_tests_by_age_cox.age_diff_p.fillna(1))
    predictive_tests_by_age_cox.sort_values(by="age_diff_p").to_csv(OUTDIR / "predictive_tests_by_age.cox.txt", sep="\t", index=False)

    return predictive_tests_by_age_cox