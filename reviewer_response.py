import statsmodels.formula.api as smf
import scipy.stats

import longitudinal_statistics

def response(data, phecode_info, case_status):
    # morning vs evening person
    res = smf.ols("temp_amplitude ~ self_report_chronotype + " + " + ".join(longitudinal_statistics.COVARIATES), data=data).fit()
    delta = -res.params['self_report_chronotype']
    hi,low = -res.conf_int().loc['self_report_chronotype']
    p = res.pvalues['self_report_chronotype']
    print(f'''Individuals who identified as a morning person had {delta:0.2f} °C ({low:0.3f}-{hi:0.3f} °C, 95% CI)
    higher temperature amplitudes than those who identified as an evening person
    (p = {p:0.1e} OLS regression with the same covariates as the main analysis).''')

    # Napping
    res = smf.ols("temp_amplitude ~ self_report_nap_during_day + " + " + ".join(longitudinal_statistics.COVARIATES), data=data).fit()
    delta = -res.params['self_report_nap_during_day']
    hi,low = -res.conf_int().loc['self_report_nap_during_day']
    p = res.pvalues['self_report_nap_during_day']
    print(f'''Individuals who reported never/rarely napping during the day had {delta:0.2f} °C ({low:0.3f}-{hi:0.3f} °C, 95% CI)
    higher temperature amplitudes than those who reported usually napping during the day
    (p = {p:0.1e} OLS regression with the same covariates as the main analysis).''')


    # Hypertension after excluding shiftworkers
    # Note that if someone was unemployed, they did not answer the job_invovles_shiftwork question. Therefore we treat NA as never shiftwork
    d = data.query("job_involves_shiftwork.isna() | (job_involves_shiftwork == 'Never/rarely')")
    hypertension_cases = case_status.query("PHECODE == '401'")
    results = longitudinal_statistics.predictive_tests_cox(d, phecode_info, hypertension_cases, OUTDIR=None)
    print(f"Main analysis but excluding all the subjects that report shiftwork:")
    print(results)

    ## Compare workdays to weekends
    likely_workers = data.query("age_at_actigraphy < 65")[['weekend_temp_amplitude', 'weekday_temp_amplitude']].dropna()
    likely_workers = likely_workers[(likely_workers < 10).all(axis=1)]
    print("Weekend vs Weekday")
    print(likely_workers.describe())
    test = scipy.stats.ttest_rel(likely_workers['weekend_temp_amplitude'], likely_workers['weekday_temp_amplitude'])
    print(test)
