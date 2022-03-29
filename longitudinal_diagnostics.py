import math
import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic
import pylab

from longitudinal_statistics import COVARIATES

import rpy2.robjects as robjects
import rpy2.robjects.packages
from rpy2.robjects import pandas2ri
robjects.r("memory.limit(10000)")

# P value cutoff to use for cox.zph results to include factors as time-varying
ZPH_PS_CUTOFF = 0.01
# Number of bins to use for time-interaction to approximate the continuous time interaction
NUM_TIME_BINS = 20

utils = robjects.packages.importr("utils")
packnames = ['survival', 'timereg']
names_to_install = [x for x in packnames if not robjects.packages.isinstalled(x)]
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
if len(names_to_install) > 0:
    utils.install_packages(robjects.vectors.StrVector(names_to_install))
survival = robjects.packages.importr('survival')
timereg = robjects.packages.importr('timereg')


def time_interaction(data, vars, entry_var, time_var, event_var, time_bins):
    ''' expand data out to multiple instance, one per time bin so that Cox PH model
    can be run with a time-interaction on each of the columns named in `vars`

    `entry_var` is the column denoting entry time (left censor)
    `time_var` is the column denoting event time (or right censor)
    `event_var` is the column containing the events, with 'censored' indicating censorship
    data will be expanded so that each entry is split into left-and-right censored segments
    that are each contained inside one bin of time_bins '''
    time_mean = data[time_var].mean()
    datas = []
    for bin in time_bins:
        start = bin.left
        end = bin.right
        d = data.copy()
        # Restrict to time bin
        d['start'] = numpy.clip(d[entry_var], start, end)
        d['end'] = numpy.clip(d[time_var], start, end)
        # Drop segments that do not occur during the time bin
        d = d[d.start != d.end]
        # Censored if we cut the interval short
        d.loc[d.end == end, event_var] = 'censored'
        d['time'] = (bin.mid - time_mean)
        for var in vars:
            d[var+"_time"] =  d.time * d[var]
        datas.append(d)
    return pandas.concat(datas)

def diagnostics(data, case_status, top_phenotypes, OUTDIR, variable="temp_amplitude"):
    last_date = case_status.first_date.max()
    variable_SD = data[variable].std()

    d = data[[variable, 'date_of_death', 'birth_year_dt', 'age_at_actigraphy'] + COVARIATES].copy()

    results = {}
    for diagnosis in top_phenotypes:
        diagnoses = case_status[case_status.PHECODE == diagnosis].set_index('ID')
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
        d2 = d[d.use].reset_index()[['time_to_event', 'case_status', 'index', 'event', variable] + COVARIATES].dropna(how="any").rename(columns={"index": "ID"})

        # Run it in R
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            d2_r = robjects.conversion.py2rpy(d2)
        robjects.globalenv['d2'] = d2_r

        # First, we have the standard base model
        rfit = robjects.r(f'''
        res <- coxph(
            Surv(time_to_event, event == 'diagnosed') ~ {variable} + BMI + age_at_actigraphy_cat + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
            data=d2,
            id=ID,
            cluster=ID,
        )
        print(summary(res))
        res
        ''')

        # Next, we check for proportinoal hazards assumption met (i.e. hazard ratio is constant in time)
        zph = robjects.r('''
        zph <- cox.zph(res)
        print(zph)
        zph
        ''')

        # Extract results from R into summary objects
        zph_ps = pandas.Series(
            numpy.array(robjects.r("zph$table[,'p']")),
            index = robjects.r("rownames(zph$table)"),
        )
        coeffs = pandas.Series(
            numpy.array(robjects.r("coefficients(res)")),
            index = numpy.array(robjects.r("names(coefficients(res))"))
        )
        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res)$coefficients)')),
        )
        amp_summary = {
                'p': summary.loc['temp_amplitude', 'Pr(>|z|)'],
                'logHR': summary.loc['temp_amplitude', 'coef'],
                'logHR_se': summary.loc['temp_amplitude', 'robust se'],
                'std_logHR': summary.loc['temp_amplitude', 'coef'] * variable_SD,
                'std_logHR_se': summary.loc['temp_amplitude', 'robust se'] * variable_SD,
        }
        results[diagnosis] = {
            'coeffs': coeffs,
            'summary': summary,
            'amp': amp_summary,
            'zph_ps': zph_ps,
        }


        # Plot zph results
        # for manually checking linearity of non-prop. hazards effects
        fig_dir = OUTDIR / 'diagnosis_figs'
        fig_dir.mkdir(exist_ok=True)
        for i, var in enumerate(zph_ps.index):
            if var == "GLOBAL":
                continue
            output_img = str(fig_dir /  f'{diagnosis}.{var}.png')
            rcode = f'''
            png({repr(output_img)})
            plot(zph[{i+1}], col=2)
            dev.off()
            '''
            robjects.r(rcode)

        # Find the variables that cox.zph identifies as time-varying
        # by comparing p-values to ZPH_PS_CUTOFF
        time_varying = []
        for var in ["temp_amplitude", "BMI", "townsend_deprivation_index"]:
            if any(zph_ps[zph_ps.index.str.startswith(var)] < ZPH_PS_CUTOFF):
                time_varying.append(var)
        time_varying_cat = []
        for var in ["age_at_actigraphy_cat", "sex", "overall_good_health", "smoking_ever", "college_education", "ethnicity_white", "alcohol"]:
            if any(zph_ps[zph_ps.index.str.startswith(var)] < ZPH_PS_CUTOFF):
                time_varying_cat.append(var)

        #### Run with time-interaction as identified by cox.zph above
        # Manually expand the dataset with `time_interaction` with a limited number of time bins
        # so that the memory explosion is manageable - coxph's tt support for this crashes with our large dataset even after down-sampling
        # this approximates time interactions with a more manageable number of intervals (NUM_TIME_BINS)
        time_bins = pandas.cut(d2.time_to_event, NUM_TIME_BINS).unique()
        d2['start'] = 0
        d2_time = time_interaction(d2, time_varying, 'start', "time_to_event", "event", time_bins)
        d2_time['start_state'] = 'censored'
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            d2_time_r = robjects.conversion.py2rpy(d2_time)
        robjects.globalenv['d2_time'] = d2_time_r

        # Time interactions model with manually expanded dataset
        non_time_varying = " ".join([f"+ {cov}" for cov in COVARIATES if cov not in time_varying_cat])
        time_varying_factors = ' '.join(f"+ {var}_time" for var in time_varying)
        strata = ', '.join(time_varying_cat)
        strata = f"+ strata({strata})" if len(time_varying_cat) > 0 else ''
        rfit2 = robjects.r(f'''
        res2 <- coxph(
            Surv(start, end, event == 'diagnosed') ~ {variable} {non_time_varying}  {time_varying_factors} {strata},
            data=d2_time,
            id=ID,
            cluster=ID,
            istate=start_state,
        )
        print(summary(res2))
        res2
        ''')

        # Extract results of time-varying datasets
        tv_summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res2)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res2)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res2)$coefficients)')),
        )
        tv_amp_summary = {
                'p': tv_summary.loc['temp_amplitude', 'Pr(>|z|)'],
                'logHR': tv_summary.loc['temp_amplitude', 'coef'],
                'logHR_se': tv_summary.loc['temp_amplitude', 'robust se'],
                'std_logHR': tv_summary.loc['temp_amplitude', 'coef'] * variable_SD,
                'std_logHR_se': tv_summary.loc['temp_amplitude', 'robust se'] * variable_SD,
        }
        if 'temp_amplitude' in time_varying:
            tv_amp_summary.update({
                'p_time': tv_summary.loc['temp_amplitude_time', 'coef'], # p-value of slope of amp effect over time
                'logHR_time': tv_summary.loc['temp_amplitude_time', 'coef'], # change per year from mean diagnosis time
                'logHR_time_se': tv_summary.loc['temp_amplitude_time', 'robust se'],
                'std_logHR_time': tv_summary.loc['temp_amplitude_time', 'coef'] * variable_SD,
                'std_logHR_time_se': tv_summary.loc['temp_amplitude_time', 'robust se'] * variable_SD,
            })
        results[diagnosis].update({
            "time_varying_summary":  tv_summary,
            "time_varying_amp_summary": tv_amp_summary,
            "time_varying_factors": time_varying,
        })


        ### Run with non-linear temp_amplitude effects
        # Fit a spline over temp_amplitude
        rfit3 = robjects.r('''
        res3 <- coxph(
            Surv(time_to_event, event=='diagnosed') ~ pspline(temp_amplitude, df=3) + BMI + age_at_actigraphy_cat + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
            data=d2,
            id=ID,
            cluster=ID,
        )
        print(summary(res3))
        res3
        ''')

        # Plot the nonlinearity for manual inspection
        nonlinear_figs = fig_dir / "nonlinear"
        nonlinear_figs.mkdir(exist_ok=True)
        nonlinear_plot = repr(str(nonlinear_figs/ f'{diagnosis}.nonlinear.temp_amplitude.png'))
        robjects.r(f'''
        png({nonlinear_plot})
        termplot(res3, term=1, se=TRUE)
        dev.off()
        ''')

        # Extract nonlinear results
        nonlinear_summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res3)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res3)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res3)$coefficients)')),
        )
        results[diagnosis].update({
            "nonlinear_summary": nonlinear_summary,
        })


        ### Run with competing-outcomes effects (i.e. death vs diagnosis)
        rfit4 = robjects.r('''
        res4 <- coxph(
            Surv(time_to_event, event) ~ temp_amplitude + BMI + age_at_actigraphy_cat + sex + overall_health_good + smoking_ever + college_education + ethnicity_white + alcohol + townsend_deprivation_index,
            data=d2,
            id=ID,
            cluster=ID,
        )
        print(summary(res4))
        res4
        ''')

        # Extract results from R into summary objects
        summary = pandas.DataFrame(
            numpy.array(robjects.r('summary(res4)$coefficients')),
            index = numpy.array(robjects.r('rownames(summary(res4)$coefficients)')),
            columns = numpy.array(robjects.r('colnames(summary(res4)$coefficients)')),
        )
        amp_summary = {
                'p': summary.loc['temp_amplitude_1:2', 'Pr(>|z|)'],
                'logHR': summary.loc['temp_amplitude_1:2', 'coef'],
                'logHR_se': summary.loc['temp_amplitude_1:2', 'robust se'],
                'std_logHR': summary.loc['temp_amplitude_1:2', 'coef'] * variable_SD,
                'std_logHR_se': summary.loc['temp_amplitude_1:2', 'robust se'] * variable_SD,
        }
        results[diagnosis].update({
            'competing_outcomes_amp_summary': amp_summary,
            'competing_outcomes_summary': summary,
        })



        robjects.r('gc()') # Helps with memory use to clear between diagnoses


    # Aggregate results across the different diagnoses investigated
    amp_summary = pandas.DataFrame({
        diagnosis: res['amp']
        for diagnosis, res in results.items()
    })
    time_varying_amp_summary = pandas.DataFrame({
        diagnosis: res['time_varying_amp_summary']
        for diagnosis, res in results.items()
    })
    competing_outcomes_amp_summary = pandas.DataFrame({
        diagnosis: res['competing_outcomes_amp_summary']
        for diagnosis, res in results.items()
    })
    def label(df, diag):
        df = df.copy()
        df['diagnosis'] = diag
        return df
    summary = pandas.concat([
        label(res['summary'], diagnosis)
        for diagnosis, res in results.items()
    ])

    zph_ps = pandas.DataFrame({
        diagnosis: res['zph_ps']
        for diagnosis, res in results.items()
    })

    amp_summary.to_csv(fig_dir / "amp_summary.txt", sep="\t")
    time_varying_amp_summary.to_csv(fig_dir / "time_varying_amp_summary.txt", sep="\t")
    competing_outcomes_amp_summary.to_csv(fig_dir / "competing_outcomes_amp_summary.txt", sep="\t")
    summary.to_csv(fig_dir / "model.summary.txt", sep="\t")
    zph_ps.to_csv(fig_dir / "zph.ps.txt", sep="\t")

    return amp_summary, time_varying_amp_summary, competing_outcomes_amp_summary, zph_ps, results