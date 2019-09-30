import pandas
import numpy
import pylab
import scipy
from statsmodels.multivariate import cancorr
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

PERFORM_PERMUTATION_TEST = False #NOTE: slow! So I've left it disabled

CONDITIONS = ["all", "control", "depressed", "anxious", "depressed_anxious"]

# Load the data
ukbb_data = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)

# Grab accelerometer summary values but only those we deem useful
# They include as output about ~900 values, most of which are like "sleep-hourOfWeekday-5-avg" which is of questionable use
acc_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", sep="\t", index_col=0)
def select_columns(col):
    return any(x in col
                for x in ["field-", "-overall-",  "quality-"])
acc_summary = acc_summary[[c for c in acc_summary.columns if select_columns(c)]]

activity_features = pandas.read_csv("../processed/activity_features_aggregate.txt", sep="\t", index_col = 0)
#Drop this column since it is linearly dependent upon several others and makes CCA impossible
activity_features.drop(columns=["total_sleep_avg", "total_sleep_std", "main_sleep_duration_avg", "main_sleep_duration_std"], inplace=True)

data = ukbb_data.join(acc_summary, how="inner").join(activity_features, how="inner")

# Determine which subset to use
data['quality_actigraphy'] = (data['quality-goodCalibration'] == 1) & ~(data['quality-daylightSavingsCrossover'] == 1) & (data['quality-goodWearTime'] == 1)

data['has_questionnaire'] = ~data.date_of_mental_health_questionnaire.isna()

data['use'] = data.quality_actigraphy & data.has_questionnaire

# Binarized depression value
data['lifetime_depression'] = (data.ever_prolonged_depression == 1) | (data.ever_prolonged_loss_of_interest == 1)

data['lifetime_anxiety'] =  (data.ever_worried_much_more == 1) | (data.longest_period_worried >= 6) | (data.longest_period_worried == -999)

all_data = data
for CONDITION in CONDITIONS:
    if CONDITION == "depressed":
        data = all_data[all_data.lifetime_depression == 1].copy()
    elif CONDITION == "anxious":
        data = all_data[all_data.lifetime_anxiety == 1].copy()
    elif CONDITION == "depressed_anxious":
        data = all_data[(all_data.lifetime_anxiety == 1) & (all_data.lifetime_depression == 1)].copy()
    elif CONDITION == "control":
        data = all_data[(all_data.lifetime_anxiety == 0) & (all_data.lifetime_depression == 0)].copy()

    # Give estimated values for the '-999' values meaning "Too many to count"
    # assign 99% percentile to those
    # TODO: should this be done in gather_fields.py??
    data.loc[data.longest_period_worried == -999, "longest_period_worried"] = data.longest_period_worried.quantile(0.99)
    data.loc[data.number_depressed_periods == -999, "number_depressed_periods"] = data.number_depressed_periods.quantile(0.99)

    ## Do Univariate correlations on all the variables
    corr = data.corr()
    # Restrict to correlations between mental-health and activity values (not within)
    corr = corr.reindex(labels=ukbb_data.columns, columns=list(activity_features.columns) + list(acc_summary.columns))
    # Put corr into long form, so we can sort for all correlation
    corr['first_var'] = corr.index
    corr_long = corr.melt(id_vars=["first_var"])
    corr_long = corr_long.dropna().sort_values(by="value", ascending=False)
    corr_long.rename(columns={"variable": "second_var", "value":"correlation"}, inplace=True)

    # Compute p-values for these correlations
    # however they N varies for each set of two variables due to NaNs
    def corr_p(row):
        N = numpy.sum( ~(data[row.first_var].isna() | data[row.second_var].isna()) )
        t = row.correlation * numpy.sqrt((N - 2) / (1 - row.correlation**2))
        return scipy.stats.distributions.t(df=N-2).sf(abs(t)) * 2
    corr_long['p'] = corr_long.apply(corr_p, axis=1)
    corr_long.to_csv(f"../processed/mental_health/univariate_correlations.{CONDITION}.txt", sep="\t", index=False)


    def plot_corr(A,B, title='', order=False):
        # Show correlations between two matrices A, B
        A_is_B = False
        if (A is B):
            A_is_B = True

        if isinstance(A, pandas.DataFrame):
            A = A.dropna()
            A = A._get_numeric_data()
        if isinstance(B, pandas.DataFrame):
            B = B.dropna()
            B = B._get_numeric_data()

        A = A - A.mean(axis=0)
        B = B - B.mean(axis=0)

        if order:
            A_noise = numpy.random.normal(size=A.shape)*1e-8
            A_order = hierarchy.leaves_list(hierarchy.linkage((A+A_noise).T, metric="correlation", optimal_ordering=True))
            A = A.iloc[:,A_order]
            if not A_is_B:
                B_noise = numpy.random.normal(size=B.shape)*1e-8
                B_order = hierarchy.leaves_list(hierarchy.linkage((B+B_noise).T, metric="correlation", optimal_ordering=True))
                B = B.iloc[:,B_order]
            else:
                B = A

        fig = pylab.figure(figsize=(14,14))
        ax = fig.add_subplot(111)
        cov = (A.T @ B).values # Covariance matrix between A and B
        cov_A = (A.T @ A).values
        cov_B = (B.T @ B).values
        corr =  cov / numpy.sqrt(cov_A.diagonal()[:,numpy.newaxis]) / numpy.sqrt(cov_B.diagonal()[numpy.newaxis,:])

        corr[numpy.isnan(corr)] = 0 # Why not?

        ax.set_xticklabels(B.columns, rotation=90)#, horizontalalignment="right")
        ax.set_xticks(numpy.arange(len(B.columns)))
        ax.set_yticklabels(A.columns)
        fig.subplots_adjust(bottom=0.3, left=0.3, top=0.95, right=0.95)
        ax.set_yticks(numpy.arange(len(A.columns)))
        im = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1)
        pylab.colorbar(im, ax=ax)

        pylab.title(title)

        pylab.show()
        return corr

    mental_health = data[data.use][ukbb_data.columns].select_dtypes(include=["number"])
    activity = data[data.use][list(acc_summary.columns) + list(activity_features.columns)].select_dtypes(include=["number"])

    # We fill in this values so that it's not throwing out all depressive males!!!
    # TODO: we should either handle this earlier in gather_fields.py OR find another way to handle it (drop it?)
    mental_health.loc[mental_health.depression_related_to_childbirth.isna(), "depression_related_to_childbirth"] = 0

    # Drop rows with NAs and columns with no variation (can't do correlation with no variation)
    # And drop some columns with low numbers of non-nan responses
    # eg: due to only being asked conditional on other questions and having no good default fillvalue
    mental_health = mental_health.drop(columns=['age_at_first_episode', 'age_at_last_episode']).dropna()
    activity = activity.dropna()
    both_index = mental_health.index.intersection(activity.index)
    mental_health = mental_health.loc[both_index]
    activity = activity.loc[both_index]

    # Control for sex effects
    # by subtracting the mean for the sex
    # TODO: do this earlier and add age as a variable
    original_mental_health = mental_health.copy()
    original_activity = activity.copy()
    mental_health_M_mean = mental_health.loc[mental_health.sex == 1].mean()
    mental_health_F_mean = mental_health.loc[mental_health.sex == 0].mean()
    mental_health_sex_diff = mental_health_M_mean - mental_health_F_mean
    mental_health.loc[mental_health.sex == 1] -= mental_health_sex_diff
    activity_M_mean = activity.loc[mental_health.sex == 1].mean()
    activity_F_mean = activity.loc[mental_health.sex == 0].mean()
    activity_sex_diff = activity_M_mean - activity_F_mean
    activity.loc[mental_health.sex == 1] -= activity_sex_diff

    # Drop rows with no variance (everyone has same value, correlation undefined)
    activity = activity.loc[:,activity.std() > 0]
    mental_health = mental_health.loc[:,mental_health.std() > 0]

    # Order variables hierarchically by correlation
    mental_health_order = hierarchy.leaves_list(hierarchy.linkage(mental_health.T, metric="correlation", optimal_ordering=True))
    activity_order = hierarchy.leaves_list(hierarchy.linkage(activity.T, metric="correlation", optimal_ordering=True))
    mental_health = mental_health.iloc[:,mental_health_order]
    activity = activity.iloc[:,activity_order]

    # Plot univariate correlations between the two datasets
    plot_corr(mental_health, mental_health, f"Mental Health Correlations {CONDITION}")
    pylab.savefig(f"../processed/mental_health/mental_health_correlations.{CONDITION}.svg")
    plot_corr(activity, activity, f"Activity Correlations {CONDITION}")
    pylab.savefig(f"../processed/mental_health/activity_correlations.{CONDITION}.svg")
    plot_corr(mental_health, activity, f"Activity - Mental Health Correlations {CONDITION}")
    pylab.savefig(f"../processed/mental_health/activity_mental_health_correlation.{CONDITION}.svg")

    #### Perform CCA on activity vs mental health questionnaire
    # Perform CCA
    cca_res = cancorr.CanCorr(mental_health, activity)
    cca_test = cca_res.corr_test()
    print(cca_test.summary())
    cca_corr = cca_test.stats['Canonical Correlation']
    cca_pvals = cca_test.stats['Pr > F']
    cca_corr = cca_corr[cca_pvals < 0.05]

    ## Do permutation test of the CCA results
    def permutation_test(A,B,N = 10_000, num_correlations = 1):
        perm_correlations = numpy.empty(N, num_correlations)
        for i in range(N):
            rows = numpy.random.permutation(B.shape[0])
            perm_B = B.values[rows]
            perm_result = cancorr.CanCorr(A, perm_B)
            perm_corr = perm_result.corr_test().stats["Canonical Correlation"][:num_correlations]
            perm_correlations[i] = perm_corr
            if n % 100 == 0:
                print(".",sep='')
        permutation_p = (numpy.sum(perm_correlations > cca_corr) + 1) / (N + 1)
        return permutation_p, perm_correlations

    if PERFORM_PERMUTATION_TEST:
        # NOTE: this is somewhat slow
        print(f"Starting permutation test for CCA significant (N = {N}). This may take some time.")
        N = 10_000
        perm_p, perm_correlations = permutation_test(mental_health, activity, N)
        print(f"Done permutation test.\nTop CCA factor has  p = {perm_p}")
        # When I ran this to N = 4020 I got
        # p = 0.00025 for the top component, which is the lowest possible (i.e. no permutations exceeded our correlation)

    # Examine the top CCA component
    # Compute the correlations of each univariate variable with the scores
    mental_health_coefficients = pandas.DataFrame(cca_res.y_cancoef[:,cca_pvals < 0.05], index=mental_health.columns)
    activity_coefficients = pandas.DataFrame(cca_res.x_cancoef[:,cca_pvals < 0.05], index=activity.columns)
    mental_health_scores = mental_health @ mental_health_coefficients
    activity_scores = activity @ activity_coefficients

    cca_labels = [f"{corr:0.1%}" for corr in cca_test.stats['Canonical Correlation'][cca_pvals < 0.05]]
    mental_health_scores.columns = cca_labels
    activity_scores.columns = cca_labels

    plot_corr(mental_health_scores, mental_health, f"Mental Health Variable Correlations with Canonical Components {CONDITION}")
    pylab.savefig(f"../processed/mental_health/mental_health_corr_with_top_cca.{CONDITION}.svg")
    plot_corr(activity_scores, activity, f"Activity Variable Correlations with Canonical Components {CONDITION}")
    pylab.savefig(f"../processed/mental_health/activity_corr_with_top_cca.{CONDITION}.svg")
    #plot_corr(mental_health_scores, A, "Mental Health CCA Scores vs Activity Measures") #The cross correlations
    #plot_corr(activity_scores, B, "Activity CCA Scores vs Mental Health Measures")


    ### PERFORM PCA within just the actigraphy values
    pca = PCA(n_components=2)
    activity_data = data.loc[data.use, list(activity_features.columns) + list(acc_summary.columns)].dropna()
    zscore_activity = (activity_data - activity_data.mean(axis=0)) / activity_data.std(axis=0)
    zscore_activity.dropna(axis=1, inplace=True)
    activity_PCA_coords =  pca.fit_transform(zscore_activity)
    fig = pylab.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(*(activity_PCA_coords.T))
    ax.set_xlabel(f"PCA1 {pca.explained_variance_ratio_[0]:0.1%}")
    ax.set_ylabel(f"PCA2 {pca.explained_variance_ratio_[1]:0.1%}")
    ax.set_title(f"PCA of Activity Variables - {CONDITION}")
    fig.savefig(f"../processed/mental_health/activity_PCA.{CONDITION}.png")
