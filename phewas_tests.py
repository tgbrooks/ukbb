import scipy
import numpy
import statsmodels.formula.api as smf
import pylab
import pandas

from util import BH_FDR
import fields_of_interest

# List of covariates we will controll for in the linear model
covariates = ["sex", "ethnicity_white", "overall_health_good", "high_income", "smoking_ever", "age_at_actigraphy", "BMI", "college_education"]
survival_covariates = ["BMI", "smoking_ever"]

# OLS method is either 'svd' or 'qr'
# Using 'qr' since intermittent problems wtih SVD convergence
OLS_METHOD = 'qr'
def OLS(*args, **kwargs):
    for _ in range(5):
        try:
            return smf.ols(*args, **kwargs).fit()#method=OLS_METHOD)
        except numpy.linalg.LinAlgError:
            print("Failed regression:")
            print(args)
            print(kwargs)
            print("Attempt number {i}")
            continue
    raise numpy.linalg.LinAlgError


def compute_phecode_test(activity_variable, phecode, data):
    covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
    fit = OLS(f"{activity_variable} ~ Q({phecode}) + sex * ({covariate_formula})",
                 data=data)
    p = fit.pvalues[f"Q({phecode})"]
    coeff = fit.params[f"Q({phecode})"]
    std_effect = coeff / data[activity_variable].std()
    N_cases = data.loc[~data[activity_variable].isna(), phecode].sum()
    return {"phecode": phecode,
             "activity_var": activity_variable,
             "p": p,
             "coeff": coeff,
             "std_effect": std_effect,
             "N_cases": N_cases,
    }, fit 

def phecode_tests(data, phecode_groups, activity_variables, activity_variable_descriptions, phecode_info, OUTDIR, RECOMPUTE=True):
    if not RECOMPUTE:
        try:
            phecode_tests = pandas.read_csv(OUTDIR+"phecodes.txt", sep="\t")
            phecode_tests_by_sex = pandas.read_csv(OUTDIR+"/all_phenotypes.by_sex.txt", sep="\t")
            return phecode_tests, phecode_tests_by_sex
        except FileNotFoundError:
            pass

    print("Computing phecode associations")
    ## Compute the phecode associations
    phecode_tests_list = []
    for group in phecode_groups:
        print(group, )
        N = data[group].sum()
        if N < 50:
            print(f"Skipping {group} - only {N} cases found")
            continue
        
        for activity_variable in activity_variables:
            summary, fit = compute_phecode_test(activity_variable, group, data)
            phecode_tests_list.append(summary)
    phecode_tests = pandas.DataFrame(phecode_tests_list)

    phecode_tests['q'] = BH_FDR(phecode_tests.p)
    phecode_tests["phecode_meaning"] = phecode_tests.phecode.map(phecode_info.phenotype)
    phecode_tests["phecode_category"] = phecode_tests.phecode.map(phecode_info.category)

    phecode_tests['Activity Category'] = phecode_tests.activity_var.map(activity_variable_descriptions["Category"])
    phecode_tests['Activity Subcategory'] = phecode_tests.activity_var.map(activity_variable_descriptions["Subcategory"])
    phecode_tests['Activity Units'] = phecode_tests.activity_var.map(activity_variable_descriptions["Units"])

    phecode_tests.to_csv(OUTDIR+f"phecodes.txt", sep="\t", index=False)

    print("Computing phecode tests by sex")
    phecode_tests_by_sex_list = []
    sex_covariate_formula = ' + '.join(c for c in covariates if c != 'sex')

    for group in phecode_groups:
        N = data[group].sum()
        N_male = numpy.sum(data[group].astype(bool) & (data.sex == "Male"))
        N_female = numpy.sum(data[group].astype(bool) & (data.sex == "Female"))
        if N_male <= 50 or N_female < 50:
            print(f"Skipping {group} - only {N_male} M and  {N_female} F cases found")
            continue
            
        if False: #phecode_tests.loc[group, "q"] > 0.01:
            # Skip test, not significant
            print(f"Skipping {group} since q > 0.01")
            continue
        
        for activity_variable in activity_variables:
            fit = OLS(f"{activity_variable} ~ 0 + C(sex, Treatment(reference=-1)) : ({sex_covariate_formula} +  Q({group}))",
                             data=data)


            female_coeff = fit.params[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']
            male_coeff = fit.params[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            p_female = fit.pvalues[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']
            p_male = fit.pvalues[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            diff_test = fit.t_test(f'C(sex, Treatment(reference=-1))[Male]:Q({group}) = C(sex, Treatment(reference=-1))[Female]:Q({group})')
            p_diff = diff_test.pvalue
            conf_ints = fit.conf_int()
            male_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Male]:Q({group})']
            female_conf_int = conf_ints.loc[f'C(sex, Treatment(reference=-1))[Female]:Q({group})']

            male_std = data.loc[data.sex == "Male", activity_variable].std()
            female_std = data.loc[data.sex == "Female", activity_variable].std()
            
            phecode_tests_by_sex_list.append({
                "phecode": group,
                "activity_var": activity_variable,
                "std_male_coeff": float(male_coeff) / male_std,
                "std_female_coeff": float(female_coeff) /  female_std,
                "p_male": float(p_male),
                "p_female": float(p_female),
                "p_diff": float(p_diff),
                "N_male": N_male,
                "N_female": N_female,
                "std_male_coeff_low": float(male_conf_int[0]) / male_std,
                "std_male_coeff_high": float(male_conf_int[1]) / male_std,
                "std_female_coeff_low": float(female_conf_int[0]) / female_std,
                "std_female_coeff_high": float(female_conf_int[1]) / female_std,
            })

    phecode_tests_by_sex = pandas.DataFrame(phecode_tests_by_sex_list)

    phecode_tests_by_sex["phecode_meaning"] = phecode_tests_by_sex.phecode.map(phecode_info.phenotype)
    phecode_tests_by_sex["phecode_category"] = phecode_tests_by_sex.phecode.map(phecode_info.category)
    phecode_tests_by_sex = phecode_tests_by_sex.join(phecode_tests.q, how="left")
    phecode_tests_by_sex['q_diff'] = BH_FDR(phecode_tests_by_sex.p_diff)
    phecode_tests_by_sex['differential_std_coeff'] = phecode_tests_by_sex.std_male_coeff - phecode_tests_by_sex.std_female_coeff
    phecode_tests_by_sex.sort_values(by="p_diff", inplace=True)

    phecode_tests_by_sex['Activity Category'] = phecode_tests_by_sex.activity_var.map(activity_variable_descriptions["Category"])
    phecode_tests_by_sex['Activity Subcategory'] = phecode_tests_by_sex.activity_var.map(activity_variable_descriptions["Subcategory"])
    phecode_tests_by_sex['Activity Units'] = phecode_tests_by_sex.activity_var.map(activity_variable_descriptions["Units"])

    phecode_tests_by_sex.to_csv(OUTDIR+"/all_phenotypes.by_sex.txt", sep="\t", index=False)
    return phecode_tests, phecode_tests_by_sex

def quantitative_tests(data, quantitative_variables, activity_variables, activity_variable_descriptions, quantitative_variable_descriptions, OUTDIR, RECOMPUTE=True):
    if not RECOMPUTE:
        try:
            quantitative_tests = pandas.read_csv(OUTDIR+"/quantitative_traits.txt", sep="\t")
            quantitative_age_tests = pandas.read_csv(OUTDIR+"/quantitative_traits.by_age.txt", sep="\t")
            quantitative_sex_tests = pandas.read_csv(OUTDIR+"/quantitative_traits.by_sex.txt", sep="\t")
            return quantitative_tests, quantitative_age_tests, quantitative_sex_tests
        except FileNotFoundError:
            pass

    print("Computing quantitative tests")
    quantitative_tests_list = []
    quantitative_sex_tests_list = []
    quantitative_age_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
    for phenotype in quantitative_variables:
        if phenotype in covariates:
            # Can't regress a variable that is also a exogenous variable (namely, BMI)
            continue

        N = data[phenotype].count()
        if N < 50:
            print(f"Skipping {phenotype} - only {N} cases found")
            continue
        phenotype_std = data[phenotype].std()
        for activity_var in activity_variables:
            N = (~data[[activity_var, phenotype]].isna().any(axis=1)).sum()
            fit = OLS(f"{phenotype} ~ {activity_var} + sex * ({covariate_formula})",
                         data=data)
            p = fit.pvalues[activity_var]
            coeff = fit.params[activity_var]
            activity_var_std = data[activity_var].std()
            std_effect = coeff * activity_var_std / phenotype_std
            quantitative_tests_list.append({"phenotype": phenotype,
                                    "activity_var": activity_var,
                                    "p": p,
                                    "coeff": coeff,
                                    "std_effect": std_effect,
                                    "N": N,
                                })

            # Fit the by-sex fit
            sex_fit = OLS(f"{phenotype} ~ 0 + C(sex, Treatment(reference=-1)) : ({activity_var} +  {covariate_formula})",
                             data=data)
            _, sex_difference_p, _ = sex_fit.compare_f_test(fit)
            female_coeff = sex_fit.params[f'C(sex, Treatment(reference=-1))[Female]:{activity_var}']
            male_coeff = sex_fit.params[f'C(sex, Treatment(reference=-1))[Male]:{activity_var}']
            p_female = sex_fit.pvalues[f'C(sex, Treatment(reference=-1))[Female]:{activity_var}']
            p_male = sex_fit.pvalues[f'C(sex, Treatment(reference=-1))[Male]:{activity_var}']
            #diff_test = sex_fit.t_test(f'C(sex, Treatment(reference=-1))[Male]:{activity_var} = C(sex, Treatment(reference=-1))[Female]:{activity_var}')
            #p_diff = diff_test.pvalue
            male_std_ratio = data.loc[data.sex == "Male", activity_var].std() / data.loc[data.sex == "Male", phenotype].std()
            female_std_ratio = data.loc[data.sex == "Female", activity_var].std() / data.loc[data.sex == "Female", phenotype].std()
            quantitative_sex_tests_list.append({"phenotype": phenotype,
                                    "activity_var": activity_var,
                                    "sex_difference_p": sex_difference_p,
                                    "p_male": p_male,
                                    "p_female": p_female,
                                    "std_male_coeff": male_coeff * male_std_ratio,
                                    "std_female_coeff": female_coeff * female_std_ratio,
                                    "N": N,
                                   })

            #By-age association
            age_fit = OLS(f"{phenotype} ~ {activity_var} * age_at_actigraphy + sex * ({covariate_formula})",
                             data=data)
            age_difference_p = age_fit.pvalues[f"{activity_var}:age_at_actigraphy"]
            main_coeff = age_fit.params[activity_var]
            age_coeff = age_fit.params[f"{activity_var}:age_at_actigraphy"]
            std_age_effect = age_coeff * activity_var_std / phenotype_std
            age_55_pvalue = age_fit.f_test(f"{activity_var}:age_at_actigraphy*55 + {activity_var}").pvalue
            age_70_pvalue = age_fit.f_test(f"{activity_var}:age_at_actigraphy*70 + {activity_var}").pvalue
            quantitative_age_tests_list.append({"phenotype": phenotype,
                                    "activity_var": activity_var,
                                    "age_difference_p": age_difference_p,
                                    "age_main_coeff": main_coeff,
                                    "age_effect_coeff": age_coeff,
                                    "std_age_effect": std_age_effect,
                                    "N": N,
                                    "age_55_p": age_55_pvalue,
                                    "age_70_p": age_70_pvalue,
                                   })

    # Final prep of the overall quantitative associations data frame
    quantitative_tests = pandas.DataFrame(quantitative_tests_list)
    quantitative_tests['q'] = BH_FDR(quantitative_tests.p)
    quantitative_tests['Activity Category'] = quantitative_tests.activity_var.map(activity_variable_descriptions["Category"])
    quantitative_tests['Activity Subcategory'] = quantitative_tests.activity_var.map(activity_variable_descriptions["Subcategory"])
    quantitative_tests['Activity Units'] = quantitative_tests.activity_var.map(activity_variable_descriptions["Units"])

    def base_name(x):
        if "_V" in x:
            return x.split("_V")[0]
        return x
    base_variable_name = quantitative_tests.phenotype.apply(base_name)
    quantitative_tests['ukbb_field'] = base_variable_name.map(fields_of_interest.all_fields)
    quantitative_tests['Functional Category'] = quantitative_tests.phenotype.map(quantitative_variable_descriptions['Functional Categories'])
    quantitative_tests.to_csv(OUTDIR+"/quantitative_traits.txt", sep="\t", index=False)

    # Final prep of age effects dataframe
    quantitative_age_tests = pandas.DataFrame(quantitative_age_tests_list)
    stds = data[activity_variables].std()
    phenotype_stds = data[quantitative_variables].std()
    quantitative_age_tests['age_55_std_effect'] = (quantitative_age_tests["age_main_coeff"] + quantitative_age_tests['age_effect_coeff'] * 55) * quantitative_age_tests.activity_var.map(stds) / quantitative_age_tests.phenotype.map(phenotype_stds)
    quantitative_age_tests['age_70_std_effect'] = (quantitative_age_tests["age_main_coeff"] + quantitative_age_tests['age_effect_coeff'] * 70) * quantitative_age_tests.activity_var.map(stds) / quantitative_age_tests.phenotype.map(phenotype_stds)
    quantitative_age_tests['age_55_q'] = BH_FDR(quantitative_age_tests['age_55_p'])
    quantitative_age_tests['age_70_q'] = BH_FDR(quantitative_age_tests['age_70_p'])
    quantitative_age_tests['age_difference_q'] = BH_FDR(quantitative_age_tests['age_difference_p'])
    quantitative_age_tests['Activity Category'] = quantitative_age_tests.activity_var.map(activity_variable_descriptions["Category"])
    quantitative_age_tests['Activity Subcategory'] = quantitative_age_tests.activity_var.map(activity_variable_descriptions["Subcategory"])
    quantitative_age_tests['Activity Units'] = quantitative_age_tests.activity_var.map(activity_variable_descriptions["Units"])
    base_variable_name = quantitative_age_tests.phenotype.apply(base_name)
    quantitative_age_tests['ukbb_field'] = base_variable_name.map(fields_of_interest.all_fields)
    quantitative_age_tests['Functional Category'] = quantitative_age_tests.phenotype.map(quantitative_variable_descriptions['Functional Categories'])
    quantitative_age_tests.to_csv(OUTDIR+"/quantitative_traits.by_age.txt", sep="\t", index=False)

    # Final prep of sex difference dataframe
    quantitative_sex_tests = pandas.DataFrame(quantitative_sex_tests_list)
    quantitative_sex_tests['sex_difference_q'] = BH_FDR(quantitative_sex_tests['sex_difference_p'])
    quantitative_sex_tests['q_male'] = BH_FDR(quantitative_sex_tests['p_male'])
    quantitative_sex_tests['q_female'] = BH_FDR(quantitative_sex_tests['p_female'])
    quantitative_sex_tests['Activity Category'] = quantitative_sex_tests.activity_var.map(activity_variable_descriptions["Category"])
    quantitative_sex_tests['Activity Subcategory'] = quantitative_sex_tests.activity_var.map(activity_variable_descriptions["Subcategory"])
    quantitative_sex_tests['Activity Units'] = quantitative_sex_tests.activity_var.map(activity_variable_descriptions["Units"])
    base_variable_name = quantitative_sex_tests.phenotype.apply(base_name)
    quantitative_sex_tests['ukbb_field'] = base_variable_name.map(fields_of_interest.all_fields)
    quantitative_sex_tests['Functional Category'] = quantitative_sex_tests.phenotype.map(quantitative_variable_descriptions['Functional Categories'])
    quantitative_sex_tests.to_csv(OUTDIR+"/quantitative_traits.by_sex.txt", sep="\t", index=False)

    return quantitative_tests, quantitative_age_tests, quantitative_sex_tests

##### Age-associations
def age_tests(data, phecode_groups, activity_variables, activity_variable_descriptions, phecode_info, OUTDIR, RECOMPUTE):
    if not RECOMPUTE:
        try:
            age_tests = pandas.read_csv(OUTDIR+"phecodes.age_effects.txt", sep="\t")
            return age_tests
        except FileNotFoundError:
            pass

    print("Computing age associations")
    age_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if (c != 'birth_year'))
    for group in phecode_groups:
        N = data[group].sum()
        if N < 200:
            print(f"Skipping {group} - only {N} cases found")
            continue
        
        for activity_variable in activity_variables:
            #if not phecode_tests[(phecode_tests.phecode == group)
            #                     & (phecode_tests.activity_var == activity_variable)].q_significant.any():
            #    continue # Only check for age-effects in significant main-effects variables

            fit = OLS(f"{activity_variable} ~ Q({group}) * age_at_actigraphy + ({covariate_formula})",
                         data=data)
            p = fit.pvalues[f"Q({group}):age_at_actigraphy"]
            main_coeff = fit.params[f"Q({group})"]
            age_coeff = fit.params[f"Q({group}):age_at_actigraphy"]
            std_effect = age_coeff / data[activity_variable].std()
            age_55_pvalue = fit.f_test(f"Q({group}):age_at_actigraphy*55 + Q({group})").pvalue
            age_70_pvalue = fit.f_test(f"Q({group}):age_at_actigraphy*70 + Q({group})").pvalue
            age_tests_list.append({"phecode": group,
                                    "activity_var": activity_variable,
                                    "p": p,
                                    "main_coeff": main_coeff,
                                    "age_effect_coeff": age_coeff,
                                    "std_age_effect": std_effect,
                                    "N_cases": N,
                                    "age_55_p": age_55_pvalue,
                                    "age_70_p": age_70_pvalue,
                                   })
    age_tests = pandas.DataFrame(age_tests_list)

    age_tests['q'] = BH_FDR(age_tests.p)
    stds = data[activity_variables].std()
    age_tests['age_55_std_effect'] = (age_tests["main_coeff"] + age_tests['age_effect_coeff'] * 55) / age_tests.activity_var.map(stds)
    age_tests['age_70_std_effect'] = (age_tests["main_coeff"] + age_tests['age_effect_coeff'] * 70) / age_tests.activity_var.map(stds)
    age_tests['age_55_q'] = BH_FDR(age_tests.age_55_p)
    age_tests['age_70_q'] = BH_FDR(age_tests.age_70_p)
    age_tests["phecode_meaning"] = age_tests.phecode.map(phecode_info.phenotype)
    age_tests["phecode_category"] = age_tests.phecode.map(phecode_info.category)

    age_tests['Activity Category'] = age_tests.activity_var.map(activity_variable_descriptions["Category"])
    age_tests['Activity Subcategory'] = age_tests.activity_var.map(activity_variable_descriptions["Subcategory"])
    age_tests['Activity Units'] = age_tests.activity_var.map(activity_variable_descriptions["Units"])

    age_tests.to_csv(OUTDIR+f"phecodes.age_effects.txt", sep="\t", index=False)
    return age_tests

def age_sex_interaction_tests(data, phecode_groups, activity_variables, phecode_info, OUTDIR, RECOMPUTE):
    if not RECOMPUTE:
        try:
            sex_age_tests = pandas.read_csv(OUTDIR+"phecodes.sex_and_age_effects.txt", sep="\t")
            return sex_age_tests
        except FileNotFoundError:
            pass

    print("Computing sex-age associations")
    sex_age_tests_list = []
    covariate_formula = ' + '.join(c for c in covariates if (c != 'birth_year'))
    for group in phecode_groups:
        N_male = data.loc[data.sex == "Male", group].sum()
        N_female = data.loc[data.sex == "Female", group].sum()
        if N_male < 500 or N_female < 500:
            print(f"Skipping {group} - only {N_male} and {N_female} cases found in M/F")
            continue
        else:
            print(f"Running {group}")
        
        for activity_variable in activity_variables:
            fit = OLS(f"{activity_variable} ~ Q({group}) * age_at_actigraphy * sex + sex*({covariate_formula})",
                         data=data)
            p_age = fit.pvalues[f"Q({group}):age_at_actigraphy"]
            p_sex = fit.pvalues[f"Q({group}):sex[T.Male]"]
            p_age_sex_interaction = fit.pvalues[f"Q({group}):age_at_actigraphy:sex[T.Male]"]
            main_coeff = fit.params[f"Q({group})"]
            age_coeff = fit.params[f"Q({group}):age_at_actigraphy"]
            sex_coeff = fit.params[f"Q({group}):sex[T.Male]"]
            age_sex_interaction_coeff = fit.params[f"Q({group}):age_at_actigraphy:sex[T.Male]"]
            std =  data[activity_variable].std()
            sex_age_tests_list.append({"phecode": group,
                                    "activity_var": activity_variable,
                                    "p_age": p_age,
                                    "p_sex": p_sex,
                                    "p_age_sex_interaction": p_age_sex_interaction,
                                    "main_coeff": main_coeff,
                                    "sex_coeff": sex_coeff,
                                    "male_age_coeff": age_coeff + age_sex_interaction_coeff,
                                    "female_age_coeff": age_coeff,
                                    "std_main_ceoff": main_coeff / std,
                                    "std_sex_coeff": sex_coeff / std,
                                    "std_male_age_coeff": (age_coeff + age_sex_interaction_coeff) / std,
                                    "std_female_age_coeff": age_coeff /std,
                                    "N_male_cases": N_male,
                                    "N_female_cases": N_female,
                                   })
    sex_age_tests = pandas.DataFrame(sex_age_tests_list)

    sex_age_tests['q_age_sex_interaction'] = BH_FDR(sex_age_tests.p_age_sex_interaction)
    sex_age_tests["phecode_meaning"] = sex_age_tests.phecode.map(phecode_info.phenotype)
    sex_age_tests["phecode_category"] = sex_age_tests.phecode.map(phecode_info.category)

    sex_age_tests.to_csv(OUTDIR+f"phecodes.sex_and_age_effects.txt", sep="\t", index=False)
    return sex_age_tests

def survival_tests(data, activity_variables, activity_variable_descriptions, OUTDIR, RECOMPUTE=True):
    if not RECOMPUTE:
        try:
            survival_tests = pandas.read_csv(OUTDIR+"survival.by_activity_variable.txt", sep="\t")
            return survival_tests 
        except FileNotFoundError:
            pass

    print("Computing survival tests")
    covariate_formula = ' + '.join(survival_covariates)
    survival_tests_data = []
    for var in activity_variables:
        print(var)
        for method in ['newton', 'cg']:# Try two convergence methods - some work for only one method
            try:
                formula = f"age_at_death_censored ~ {var} + sex + {covariate_formula}"
                result = smf.phreg(formula=formula,
                                    data=data,
                                    status=data.uncensored,
                                    entry=data.entry_age,
                                    ).fit(method=method)
                print('.')
                interaction_formula = f"age_at_death_censored ~ {var} * sex + {covariate_formula}"
                interaction_result = smf.phreg(formula=interaction_formula,
                                    data=data,
                                    status=data.uncensored,
                                    entry=data.entry_age,
                                    ).fit(method=method)
            except numpy.linalg.LinAlgError:
                print(f"MLE fit method {method} failed, trying alternative")
                continue
            break

        pvalues = pandas.Series(result.pvalues, index=result.model.exog_names)
        params = pandas.Series(result.params, index=result.model.exog_names)
        interaction_pvalues = pandas.Series(interaction_result.pvalues, index=interaction_result.model.exog_names)
        interaction_params = pandas.Series(interaction_result.params, index=interaction_result.model.exog_names)
        survival_tests_data.append({
            "activity_var": var,
            "p": pvalues[var],
            "log Hazard Ratio": params[var],
            "standardized log Hazard Ratio": params[var] * data[var].std(),
            "sex_difference_p": interaction_pvalues[f"{var}:sex[T.Male]"],
            "male_logHR": interaction_params[f"{var}:sex[T.Male]"] + interaction_params[f"{var}"],
            "female_logHR": interaction_params[f"{var}"],
        })

    survival_tests = pandas.DataFrame(survival_tests_data)
    survival_tests['q'] = BH_FDR(survival_tests.p)
    survival_tests['sex_difference_q'] = BH_FDR(survival_tests.sex_difference_p)
    survival_tests = pandas.merge(survival_tests, activity_variable_descriptions[["Category", "Subcategory", "Units"]], left_on="activity_var", right_index=True)
    survival_tests.to_csv(OUTDIR+"survival.by_activity_variable.txt", sep="\t", index=False)
    return survival_tests

def beyond_RA_tests(data, activity_variables, activity_variable_descriptions, OUTDIR, RECOMPUTE=True):
    ### Assess what variables add to acceleration_RA the most
    if not RECOMPUTE:
        try:
            beyond_RA_tests = pandas.read_csv(OUTDIR+"beyond_RA_tests.txt", sep="\t")
            return beyond_RA_tests
        except FileNotFoundError:
            pass

    print("Computing beyond RA tests")
    covariate_formula = ' + '.join(survival_covariates)
    beyond_RA_tests_list = []
    for var in activity_variables:
        if var == "acceleration_RA":
            continue
        formula = f"age_at_death_censored ~ acceleration_RA + {var} + sex + {covariate_formula}"
        try:
            results = smf.phreg(formula=formula,
                                data=data,
                                status=data.uncensored,
                                entry=data.entry_age,
                                ).fit()
        except numpy.linalg.LinAlgError:
            print(f"Failed regression on {var} - skipping")
            continue
        pvalues = pandas.Series(results.pvalues, index=results.model.exog_names)
        params = pandas.Series(results.params, index=results.model.exog_names)
        beyond_RA_tests_list.append({
            "activity_var": var,
            "p": pvalues[var],
            "p_RA": pvalues["acceleration_RA"],
            "standardized log Hazard Ratio": params[var] * data[var].std(),
            "standardized log Hazard Ratio RA": params['acceleration_RA'] * data['acceleration_RA'].std(),
        })
    beyond_RA_tests = pandas.DataFrame(beyond_RA_tests_list)
    beyond_RA_tests = pandas.merge(beyond_RA_tests, activity_variable_descriptions[["Category", "Subcategory"]],
                            left_on="activity_var",
                            right_index=True)
    beyond_RA_tests.to_csv(OUTDIR+"beyond_RA_tests.txt", sep="\t", index=False)
    return beyond_RA_tests
