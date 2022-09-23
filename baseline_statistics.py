import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm

import util

COVARIATES = ["sex", "ethnicity_white", "overall_health", "smoking", "age_at_actigraphy_cat", "BMI", "college_education", "alcohol_frequency", "townsend_deprivation_index"]
MIN_N = 200
MIN_N_BY_AGE = 400
MIN_N_PER_SEX = 200

def bh_fdr_with_nans(ps):
    okay = ~ps.isna()
    qs = numpy.full(fill_value=float("NaN"), shape=ps.shape)
    qs[okay] = util.BH_FDR(ps[okay])
    return qs

def baseline_stats(data, phecode_info, case_status, OUTDIR, RECOMPUTE=False, variable="temp_amplitude"):
    # Predict diagnoses after actigraphy
    if not RECOMPUTE:
        try:
            predictive_tests_cox = pandas.read_csv(OUTDIR/f"baseline_stats.txt", sep="\t", dtype={"phecode": str})
            return predictive_tests_cox
        except FileNotFoundError:
            pass

    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()

    baseline_tests_list = []
    print("Starting baseline tests")
    for diagnosis, diagnoses in case_status.groupby("PHECODE"):
        print(diagnosis)
        diagnoses = diagnoses.set_index('ID')
        d['case_status'] = d.index.map(diagnoses.case_status.astype(str)).fillna('control')
        d['diagnosis_date'] = d.index.map(diagnoses.first_date)
        d['use'] = d.case_status.isin(['prior_case_exact', 'control']) # Drop anyone excluded or who later gets the diagnosis

        # Final collection of data we use for the models
        d2 = d[d.use].reset_index()[['case_status', 'index', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})
        d2['case'] = d2.case_status.map({"prior_case_exact": 1, "control": 0})

        header = {
            "activity_var": variable,
            "phecode": diagnosis,
            "meaning": phecode_info.phenotype.get(diagnosis, "NA"),
            "N_cases": (d2.case_status == 'prior_case_exact').sum(),
            "N_controls": (d2.case_status == 'control').sum(),
        }

        if header['N_cases'] < MIN_N:
            continue

        # Run the statistics
        try:
            model = smf.ols(
                formula = f"{variable} ~ case + BMI + age_at_actigraphy_cat + sex + overall_health + smoking + college_education + ethnicity_white + alcohol_frequency + townsend_deprivation_index",
                data = d2,
            )
            fit = model.fit()
        except numpy.linalg.LinAlgError:
            print(f"LinAlg error in {diagnosis}")
            continue
        header.update({
            "p": fit.pvalues['case'],
            "effect": fit.params['case'], # effect size per unit change
            "std_effect": fit.params['case'] / variable_SD, # effect size per SD change
        })
        baseline_tests_list.append(header)
    baseline_stats = pandas.DataFrame(baseline_tests_list)
    baseline_stats['q'] = bh_fdr_with_nans(baseline_stats.p.fillna(1))
    baseline_stats.sort_values(by="p").to_csv(OUTDIR / "baseline_stats.txt", sep="\t", index=False)

    return baseline_stats