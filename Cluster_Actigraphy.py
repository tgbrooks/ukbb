#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('qtconsole', '# Run for debugging purposes')


# In[209]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import scipy
import numpy
from IPython.display import display, HTML
import statsmodels.api as sm
import seaborn as sns
import re


# In[963]:


import pandas
full_activity = pandas.read_csv("../processed/activity_features_aggregate.txt", index_col=0, sep="\t")
activity_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", index_col=0, sep="\t")
ukbb = pandas.read_hdf("../processed/ukbb_data_table.h5")
full_mental_health = pandas.read_hdf("../processed/ukbb_mental_health.h5")


# In[218]:


# Remove the activity variables that we don't want to use
bad_columns = ["_IV$", "^temp_", "^light_"]
good_columns = []
for c in activity.columns:
    fail = False
    for bad in bad_columns:
        if re.search(bad, c):
            fail = True
    if not fail:
        good_columns.append(c)
activity = activity[good_columns]


# In[314]:


# drop activity for people who fail basic QC
[c for c in activity_summary.columns if 'quality' in c]
okay = activity_summary['quality-goodCalibration'].astype(bool) & (~activity_summary['quality-daylightSavingsCrossover'].astype(bool)) & (activity_summary['quality-goodWearTime'].astype(bool))
activity = activity[okay]
print(f"Dropping {(~okay).sum()} entries out of {len(okay)} due to bad quality or wear-time")


# In[886]:


covariates = ["sex", "ethnicity", "overall_health", "household_income", "smoking", "birth_year", "BMI",
               #'education_Prefer_not_to_answer', # This answer causes problems for some reason
               'education_None_of_the_above',
               'education_College_or_University_degree',
               'education_A_levels/AS_levels_or_equivalent', 
               'education_O_levels/GCSEs_or_equivalent',
               'education_CSEs_or_equivalent',
               'education_NVQ_or_HND_or_HNC_or_equivalent',
               'education_Other_professional_qualifications_eg:_nursing,_teaching',
                ]
#covariates = ["BMI"]

data = activity.copy()
data = data.join(ukbb[covariates], how="inner")
print(f"Data starting size: {data.shape}")

# Down sample for testing
numpy.random.seed(0)
selected_ids = numpy.random.choice(data.index, size=(25_000), replace=False)
data = data.loc[selected_ids]
print(f"Data size after selecting test set: {data.shape}")


# In[887]:


data.shape


# In[888]:


# Model the covariates' effect on the activity values
covariate_data = pandas.get_dummies(data[covariates])
missing = covariate_data.isna().any(axis=1) | data[activity.columns].isna().any(axis=1)
model = sm.OLS(exog=covariate_data[~missing], endog=data[activity.columns][~missing], missing='drop').fit()
# Control the activity data
controlled_data = data.copy()
controlled_data[activity.columns] -= pandas.DataFrame(model.predict(covariate_data[~missing])).rename(columns={i:c for i,c in enumerate(activity.columns)})
controlled_data[activity.columns] += data.mean(axis=0) # Add back the mean so that the numbers are meaningful


# In[985]:


# Set up to model the covariates' effect on the mental health values
missing = covariate_data.isna().any(axis=1)
#missing = covariate_data.isna()
mental_health_columns_to_control = full_mental_health.columns.difference(covariates).difference(["date_of_mental_health_questionnaire", "assessment_center"])
mental_health_dummies = pandas.get_dummies(full_mental_health[mental_health_columns_to_control]).loc[data.index][~missing]


# In[986]:


#Control the mental health data
mental_health = mental_health_dummies.copy()
for col in mental_health.columns:
    model = sm.OLS(exog=covariate_data[~missing], endog = mental_health_dummies[col], missing="drop").fit()
    fit = model.predict(covariate_data[~missing])#.rename(columns={i:c for i,c in enumerate(mental_health_dummies.columns)})
    mental_health[col] -= fit
    mental_health[col] += mental_health_dummies[col].mean()


# In[987]:



#mental_health_dummies[~missing].count().T.sort_values().to_dict()
mental_health.describe()
#fit.describe()
#model.params
#missing.sum()
#covariate_data.count().idxmin()
#covariate_data.describe().T
#covariate_data.isna().sum(axis=0)


# In[999]:


# Group the columns of mental_health by which fields they correspond to
from fields_of_interest import mental_health_fields
mental_health_questions = {field:[c for c in mental_health.columns if c.startswith(field+"_")]
                                    for field in mental_health_fields.keys()}
mental_health_questions = {field:cols for field,cols in mental_health_questions.items() if len(cols) != 0} # Drop empty questions

# It's possible that there are problematic fields where one is a substring of the other and so we would grab columns for both
# which is a problem. So we check for that here - no column should be used more than once
import collections
col_counts = collections.Counter([c for cols in mental_health_questions.values() for c in cols])
assert max(col_counts.values()) == 1


# 

# In[632]:


from sklearn.decomposition import PCA
N_PCA_COMPONENTS = 25
pca = PCA(n_components=25)


for_pca = controlled_data[activity.columns].drop(columns=[c for c in data.columns if "_IS" in c]) # IS variables throw off PCA
for_pca = for_pca.dropna()
zscored = (for_pca - for_pca.mean(axis=0))/for_pca.std(axis=0)
pca_coords = pca.fit_transform(zscored)


# In[633]:


import pylab
fig = pylab.figure(figsize=(10,5))
ax = fig.add_subplot(121)
color = [{"Male":"b", "Female":"r"}[s] for s in for_pca.index.map(ukbb.sex)] # Color by sex
ax.scatter(*(pca_coords[:,:2].T), s=1, alpha=0.5, c=color)
pylab.title("PCA Components 1+2")
ax = fig.add_subplot(122)
ax.scatter(*(pca_coords[:,1:3].T), s=1, alpha=0.5, c=color)
pylab.title("PCA Components 2+3")
pylab.show()
pca_coords


# In[558]:


for_pca.describe()


# In[634]:


pca.explained_variance_ratio_


# In[635]:


# PCA Loadings
pca_loadings = pandas.DataFrame(pca.components_.T, index=for_pca.columns)
display(HTML(pca_loadings.abs().sort_values(by=2).to_html()))
#pylab.figure()
#pylab.imshow(pca_loadings.abs().sort_values(by=0).T, )
#pylab.show()


# In[667]:


# Bootstrap PCA components
pca_bootstrap_N = 500
pca_bootstrap = PCA(n_components=N_PCA_COMPONENTS)
bootstrap_components = []
for i in range(pca_bootstrap_N):
    zscored_bootstrap = zscored.iloc[numpy.random.choice(zscored.shape[0], size=zscored.shape[0], replace=True)]
    pca_coords = pca_bootstrap.fit_transform(zscored_bootstrap)
    bootstrap_components.append(pca_bootstrap.components_)
bootstrap_components = numpy.array(bootstrap_components)


# In[670]:


bootstrap_components.shape


# In[668]:


# Display relational matrix of the various PCA components
# I.e. given our computed ith component and a randomly chosen jth component, what is cos(angle) between them?
# If PCA components are robust under bootstrap, then this will be diagonal 1s, or nearly so
pca_component_relations = numpy.empty((N_PCA_COMPONENTS, N_PCA_COMPONENTS))
for i in range(N_PCA_COMPONENTS):
    for j in range(N_PCA_COMPONENTS):
        cos_angles = pca.components_[i] @ bootstrap_components[:,j,:].T
        pca_component_relations[i,j] = numpy.mean(numpy.abs(cos_angles))
pylab.figure()
pylab.imshow(pca_component_relations, vmin=0,vmax=1)
pylab.title("PCA Component - Bootstrap Robustness")


# In[776]:


# Display the PCA loading bootstrap values
x = numpy.broadcast_to(numpy.arange(components.shape[1]).reshape((1,-1)), components.shape)

bounds = [-0.45,0.45]
def plot_pca_loadings(components, bootstrap_components, components_to_plot):
    ordering = numpy.argsort((components[0]))
    fig, axes = pylab.subplots(ncols=len(components_to_plot), figsize=(8,15))
    pylab.subplots_adjust(wspace=1)
    for i, ax in enumerate(axes):
        j = components_to_plot[i]
        
        bs_components = bootstrap_components[:,j,ordering]
        ax.scatter(bs_components, x, s=1, alpha=0.1)
        ax.scatter(components[j, ordering], x[0], s=3, alpha=1, c='k')

        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.xaxis.set_ticks_position('top')
        ax.set_xlim(bounds)

        if i < len(components_to_plot) - 1:
            ax.set_yticklabels(for_pca.columns[ordering])
            ax.set_yticks(numpy.arange(len(ordering)))
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_tick_params(length=0)
        else:
            ax.yaxis.set_ticks([])
        ax.set_title(f"PCA {j+1}")
plot_pca_loadings(pca.components_, bootstrap_components, [0,1])
plot_pca_loadings(pca.components_, bootstrap_components, [2,3])


# In[773]:


# Correlations of the components in the i-th PCA components
i = 1
#pca_loading_correlations = numpy.corrcoef(bootstrap_components[:,i,:].T)
pca_loading_correlations = numpy.corrcoef(bootstrap_components.reshape((-1, bootstrap_components.shape[2])).T) # Correlations along all the components
print(pca_loading_correlations.shape)
fig, ax = pylab.subplots(figsize=(10,10))
ax.imshow(pca_loading_correlations, cmap="bwr", vmin=-1,vmax=1)
ax.set_yticklabels(for_pca.columns)
ax.set_yticks(numpy.arange(len(for_pca.columns)));
ax.set_title("Correlations of PCA loading values")


# In[792]:


PCA_COMPONENTS_TO_USE = 2
pca_coords_full = pca_coords.copy()
pca_coords = pca_coords[:,:PCA_COMPONENTS_TO_USE]


# In[793]:


# Find an ellipse in PCA coords containing ~99% of the data
center = pca_coords[:,:PCA_COMPONENTS_TO_USE].mean(axis=0) # By defintion, the means are 0
radii = pca_coords.std(axis=0) * 3 # 3 Standard Deviations from mean
# note covariance is also 0 by definition

# Collect 'outlier' points far from the mean
outliers = numpy.sum(((pca_coords - center) / radii)**2, axis=1) > 1
outliers = pandas.Series(outliers, index=for_pca.index)

fig = pylab.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.scatter(*(pca_coords[~outliers,:2].T), s=2, alpha=0.5, c="k")
ax.scatter(*(pca_coords[outliers,:2].T), s=2, alpha=0.5, c="r")
ellipse = pylab.mpl.patches.Ellipse(center[:2], 2*radii[0], 2*radii[1], edgecolor="k", facecolor='none')
ellipse.set_transform(ax.transData)
ax.add_patch(ellipse)
pylab.xlabel("PCA Component 1")
pylab.ylabel("PCA Component 2")
pylab.title(f"Activity PCA Outlier Detection \n Found {outliers.sum()} outliers ({outliers.sum() / len(outliers):.1%})")
pylab.show()


# In[ ]:





# In[1035]:


# Check for mental health differences between outliers and controls
mental_health_outliers = mental_health.loc[outliers.index]#.select_dtypes(include=[numpy.number])

def G_cat(A,B):
    # G statistic for case where A is boolean and B is categorical
    G = 0
    for condition in [0,1]:
        fraction_A = (A == condition).sum() / A.shape[0]
        for cat in B.cat.categories:
            observed = ((B == cat) & (A == condition)).sum(axis=0)
            expected = (B == cat).sum(axis=0) * fraction_A # Expected num under null hypothesis (independence)
            if expected > 0 and observed > 0:
                # Note: never should have expected == 0 and observed > 0 but we check in case
                G += 2 * observed * numpy.log(observed/expected)
    return G

def G(condition,dummies):
    # G statistic for case where 'condition' is a boolean and
    # 'dummies' is a dataframe with all the columns encoding answers to one question
    G = 0
    for cond in [0,1]:
        fraction = (condition == cond).sum() / condition.shape[0]
        for col in dummies.columns:
            observed = (dummies[col] * (condition == cond).astype(float)).sum(axis=0)
            expected = dummies[col].sum(axis=0) * fraction
            if expected > 0 and observed > 0:
                # Note: never should have expected == 0 and observed > 0 but we check in case
                G += 2 * observed * numpy.log(observed/expected)
    return G

def correlate(A,B):
    # Assumes that A is binary 0/1 but B could be anything
    def corr_(A, col):
        if hasattr(col, 'cat'):
            # Categorical column
            # Compute the G statistic
            return G(A, col)
        elif hasattr(col, 'str'):
            # String column, treat same as categorical
            return G(A, col.astype("category"))
        elif numpy.issubdtype(col.dtype, numpy.number):
            # Numeric column, use Pearson's correlation
            return A.corr(col)
    A = pandas.Series(A, index=B.index)
    return pandas.Series({col_name: corr_(A,col) for col_name, col in B.items()})

def correlate_questions(A,B, questions):
    # Assumes that A is binary and B is a dataframe of dummy variables
    # and questions is a dictionary mapping fieldnames to the corresponding dummy columns
    A = pandas.Series(A, index=B.index)
    return pandas.Series({field: G(A,B[questions[field]])
                            for field in questions})

corr = correlate_questions(outliers, mental_health_outliers, mental_health_questions)
N = mental_health_outliers.count()


# In[ ]:


# Bootstrap ps
bootstrap_corrs = []
bootstrap_N = 200
for n in range(bootstrap_N):
    corr_perm = correlate_questions(pandas.Series(numpy.random.permutation(outliers), index=outliers.index),
                                    mental_health_outliers, mental_health_questions)
    bootstrap_corrs.append(corr_perm)
bootstrap_corrs = numpy.array(bootstrap_corrs)
bootstrap_ps = (numpy.sum(numpy.abs(bootstrap_corrs) >= numpy.abs(corr).to_numpy().reshape((1,-1)), axis=0) + 1) / (bootstrap_N + 1)


# In[1052]:





# In[1047]:


def get_highest_enriched_value(col, outliers):
    if hasattr(col, 'str'):
        col = col.astype('category')
    if hasattr(col, 'cat'):
        best_category = ""
        best_enrichment = -1
        for category in col.cat.categories:
            observed = ((col == category) & outliers).sum()
            expected = (col == category).sum() * outliers.sum() / len(outliers) 
            enrichment = observed / expected
            if enrichment > best_enrichment:
                best_enrichment = enrichment
                best_category = category
        return {"value": best_enrichment, "category": best_category}
    else:
        return {"value": "-", "category": "-"}
def get_enrichments(outliers, data, questions):
    fraction = outliers.sum() / outliers.shape[0]
    results = {}
    for field in questions:
        cols = questions[field]
        best_category = ""
        best_enrichment = -1
        for col in cols:
            label = col[len(field)+1:].replace("_"," ")
            if label in ["Prefer not to answer", "Do not know"]: # Skip the non-answers, usually too few to be interesting
                continue
            observed = (data[col] * outliers.astype(float)).sum(axis=0)
            expected = data[col].sum(axis=0) * fraction
            enrichment = observed / expected
            if enrichment > best_enrichment:
                best_enrichment = enrichment
                best_category = label
        results[field] = {"most_enriched":best_category, "enrichment":enrichment}
    return pandas.DataFrame(results).T


# In[1059]:


#highest_enriched_value = pandas.DataFrame({colname: get_highest_enriched_value(col, outliers)
#                                for colname, col in mental_health_outliers.items()}).T
highest_enriched_value = get_enrichments(outliers, mental_health_outliers, mental_health_questions)
#outlier_correlations = pandas.DataFrame( {"stat": corr, "bootstrap_p":bootstrap_ps, "N":N}).join(highest_enriched_value).sort_values(by="bootstrap_p")
outlier_correlations = highest_enriched_value.join(pandas.DataFrame({"G": corr, "p": bootstrap_ps})).sort_values("G", ascending=False)
display(HTML(outlier_correlations.to_html()))


# In[310]:





# In[976]:


# Enrichment by standard deviation from the center
# We want to divide people by how many STDs they are from the center of the actigraphy PCA and see
# how that correlates with their having various mental health questionairre answers
dist_from_pca_center = numpy.sqrt(numpy.sum((pca_coords / pca_coords.std(axis=0).reshape((1,-1)))**2, axis=1))

def correlate_continuous(A,B):
    # Assumes that A is a continuous variable, but 'col' could be anything
    def corr_(A, col):
        if hasattr(col, 'str'):
            # String column, treat same as categorical
            col = col.astype("category")
            
        if hasattr(col, 'cat'):
            # Categorical column
            # ANOVA
            good = ~(pandas.isna(A) | pandas.isna(col))
            stat, p = scipy.stats.f_oneway(*[A[(col == cat) & good] for cat in col.cat.categories])
            grand_mean = numpy.sum(A[good]) / len(A[good])
            means = {cat: numpy.sum(A[(col == cat) & good]) / numpy.sum((col == cat) & good) 
                            for cat in col.cat.categories}
            enrichments = {cat: means[cat] - grand_mean
                               for cat in col.cat.categories}
            N_by_category = {cat: numpy.sum((col == cat) & good) for cat in col.cat.categories}
            most_enriched = max(enrichments.items(), key=lambda x: x[1] if x[0] not in ["Prefer not to answer", "Do not know"] else float("-inf"))[0]
            N_in_most = N_by_category[most_enriched]
            #SS_res = numpy.sum([numpy.sum((A[(col == cat) & good] - means[cat])**2) for cat in col.cat.categories])
            #SS_tot = numpy.sum((A[good] - grand_mean)**2)
            #R2 = 1 - (SS_res/SS_tot)
            return {"stat":stat, "type": "F", "p": p, "most_enriched": most_enriched, "enrichment": enrichments[most_enriched], "N_enriched": N_in_most}
        elif numpy.issubdtype(col.dtype, numpy.number):
            # Numeric column, use Pearson's correlation
            good = ~(pandas.isna(A) | pandas.isna(col))
            corr, p = scipy.stats.pearsonr(A[good], col[good])
            return {"stat": corr, "type":"R", "p": p}
    A = pandas.Series(A, index=B.index)
    return pandas.DataFrame({col_name: corr_(A,col) for col_name, col in B.items()})

correlate_with_dist = correlate_continuous(dist_from_pca_center, mental_health.loc[for_pca.index]).T.sort_values('p')


# In[977]:


display(HTML(correlate_with_dist.to_html()))


# In[978]:


# Correlation mental health with PCA components inside the outlier groups
correlate_with_PCA2 = correlate_continuous(pca_coords[outliers,0], mental_health.loc[for_pca.index].loc[outliers]).T.sort_values('p')
display(HTML(correlate_with_PCA2.to_html()))


# In[979]:


pylab.figure()
pylab.scatter(correlate_with_dist.N_enriched, correlate_with_dist.enrichment)
pylab.xlabel("N")
pylab.ylabel("Enrichment")
pylab.show()


# In[ ]:


pylab.figure(figsize=(7,7))
var = "physically_abused_by_family_as_child"
mental_health_for_pca = mental_health.loc[for_pca.index]
by_category = [dist_from_pca_center[mental_health_for_pca[var] == cat] for cat in mental_health_for_pca[var].cat.categories]
pylab.boxplot(by_category, labels=mental_health[var].cat.categories, notch=True)
pylab.xticks(rotation=45)
pylab.subplots_adjust(bottom=0.3)
pylab.ylabel("Distance from PCA center")
pylab.xlabel(var)
pylab.show()


# In[ ]:


# High-scoring depression
depression = (((mental_health.mental_health_problems_diagnosed_Depression == 1.0)
                & (mental_health['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] == 0.0)
                & (mental_health.mental_health_problems_diagnosed_Anorexia_nervosa == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Psychological_over-eating_or_binge-eating'] == 0.0))
    & ((mental_health.ever_attempted_suicide == 'Yes')
       | ((mental_health.number_depressed_periods >= 3)
          & ((mental_health.impact_on_normal_roles_worst_episode == 'A lot')
             | (mental_health.impact_on_normal_roles_worst_episode == 'Somewhat'))))
    & ~(mental_health.ever_addicted_drugs == 'Yes'))
depression = depression.loc[for_pca.index]

activity_depression = for_pca.loc[depression.index]#.select_dtypes(include=[numpy.number])

corr_depression = correlate(depression, activity_depression)
N_depression = activity_depression.count()


# In[ ]:


# Bootstrap ps
bootstrap_corrs_depression = []
bootstrap_N_depression = 1000
for n in range(bootstrap_N_depression):
    corr_perm = correlate(pandas.Series(numpy.random.permutation(depression), index=depression.index), activity_depression)
    bootstrap_corrs_depression.append(corr_perm)
bootstrap_corrs_depression = numpy.array(bootstrap_corrs_depression)
bootstrap_ps_depression = (numpy.sum(numpy.abs(bootstrap_corrs_depression) >= numpy.abs(corr_depression).to_numpy().reshape((1,-1)), axis=0) + 1) / (bootstrap_N_depression + 1)


# In[ ]:


# Display depression <-> dist_from_pca_center correlations
highest_enriched_value_depression = pandas.DataFrame({colname: get_highest_enriched_value(col, depression )
                                for colname, col in activity_depression.items()}).T
depression_correlations = pandas.DataFrame( {"stat": corr_depression, "bootstrap_p":bootstrap_ps_depression, "N":N_depression, "most_enriched":highest_enriched_value_depression.category, "enrichment":highest_enriched_value_depression.value}).sort_values(by="bootstrap_p")
display(HTML(depression_correlations.to_html()))


# In[ ]:


# Define the other conditions
anxiety = (((mental_health['mental_health_problems_diagnosed_Depression'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] == 1.0)
                & (mental_health['mental_health_problems_diagnosed_Anorexia_nervosa'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Psychological_over-eating_or_binge-eating'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Schizophrenia'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Any_other_type_of_psychosis_or_psychotic_illness'] == 0.0)
                & (mental_health[ 'mental_health_problems_diagnosed_Mania,_hypomania,_bipolar_or_manic-depression'] == 0.0))
            & ~(mental_health.ever_addicted_drugs == 'Yes')
            & ((mental_health.longest_period_worried > 1) | (mental_health.longest_period_worried == -999))
          )
anxiety = anxiety.loc[for_pca.index]

mania = (((mental_health['mental_health_problems_diagnosed_Depression'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Anorexia_nervosa'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Psychological_over-eating_or_binge-eating'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Schizophrenia'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Any_other_type_of_psychosis_or_psychotic_illness'] == 0.0)
                & (mental_health[ 'mental_health_problems_diagnosed_Mania,_hypomania,_bipolar_or_manic-depression'] == 1.0))
            & ~(mental_health.ever_addicted_drugs == 'Yes')
            & ((mental_health.ever_mania == 'Yes'))
        )
mania = mania.loc[for_pca.index]
         
psychosis = (((mental_health['mental_health_problems_diagnosed_Depression'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Anorexia_nervosa'] == 0.0)
                & (mental_health['mental_health_problems_diagnosed_Psychological_over-eating_or_binge-eating'] == 0.0)
                & ((mental_health['mental_health_problems_diagnosed_Schizophrenia'] == 1.0)
                    | (mental_health['mental_health_problems_diagnosed_Any_other_type_of_psychosis_or_psychotic_illness'] == 1.0))
                & (mental_health[ 'mental_health_problems_diagnosed_Mania,_hypomania,_bipolar_or_manic-depression'] == 0.0))
            & ~(mental_health.ever_addicted_drugs == 'Yes')
            )
psychosis = psychosis.loc[for_pca.index]

control = (((mental_health['mental_health_problems_diagnosed_Depression'] != 1.0)
                & (mental_health['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] != 1.0)
                & (mental_health['mental_health_problems_diagnosed_Anorexia_nervosa'] != 1.0)
                & (mental_health['mental_health_problems_diagnosed_Psychological_over-eating_or_binge-eating'] != 1.0)
                & (mental_health['mental_health_problems_diagnosed_Schizophrenia'] != 1.0)
                & (mental_health['mental_health_problems_diagnosed_Any_other_type_of_psychosis_or_psychotic_illness'] != 1.0)
                & (mental_health[ 'mental_health_problems_diagnosed_Mania,_hypomania,_bipolar_or_manic-depression'] != 1.0))
         & (~mental_health.date_of_mental_health_questionnaire.isna())
         & (~(mental_health.ever_addicted_drugs == 'Yes'))
         & (mental_health.ever_mania == 'No')
         & (mental_health.ever_prolonged_depression == 'No')
)
control = control.loc[for_pca.index]


# In[ ]:


categories = {"psychosis": psychosis, "mania": mania, "anxiety": anxiety, "depression": depression, "control": control}
def get_category(i):
    for cat, vector in categories.items():
        if vector[i]:
            return cat
    return float("NaN")
category = pandas.Series([get_category(i) for i in for_pca.index],
    index = for_pca.index).astype("category")
category = category.cat.reorder_categories(["control", "depression", "anxiety", "mania", "psychosis"])
mental_health_category = pandas.DataFrame({"category": category})
for i, coord in enumerate(pca_coords.T):
    mental_health_category[f'pca_{i}'] = coord
mental_health_category['dist_from_pca_center'] = dist_from_pca_center
import collections; collections.Counter(category)


# In[ ]:


pylab.figure()
sns.scatterplot(x="pca_0", y="pca_1", hue="category", data=mental_health_category[mental_health_category.category != 'control'])
ax = pylab.gca()
ellipse = pylab.mpl.patches.Ellipse(center[:2], 2*radii[0], 2*radii[1], edgecolor="k", facecolor='none')
ellipse.set_transform(ax.transData)
ax.add_patch(ellipse)
pylab.show()


# In[ ]:


pylab.figure(figsize=(7,7))
var = "category"
mental_health_for_pca = mental_health.loc[for_pca.index]
labels = mental_health_category[var].dropna().unique()
by_category = [dist_from_pca_center[mental_health_category[var] == cat] for cat in labels]
pylab.boxplot(by_category, labels=labels, notch=True)
pylab.xticks(rotation=45)
pylab.subplots_adjust(bottom=0.3)
pylab.ylabel("Distance from PCA center")
pylab.show()


# In[ ]:





# In[ ]:


# Check correlation of condition with PCA distance
category_v_pca_dist = sm.OLS.from_formula(formula="dist_from_pca_center ~ category", data=mental_health_category).fit()
category_v_pca_dist.summary()


# In[ ]:


# Find differences between the categories
def compare(cat1, cat2):
    in_cat1 = mental_health_category.category == cat1
    in_cat2 = mental_health_category.category == cat2
    either = in_cat1 | in_cat2
    endog = for_pca[either]
    results = {}
    for var in endog:
        constant = numpy.ones(in_cat1[either].shape)
        exog = numpy.vstack( [constant, in_cat1[either].astype('int')] ).T
        res = sm.OLS(endog = endog[var], exog = exog).fit()
        contrast = [0,1] # Difference between categories is 0
        test = res.f_test(contrast)
        mean1 = endog.loc[in_cat1[either], var].mean()
        mean2 = endog.loc[in_cat2[either], var].mean()
        total_mean = for_pca[var].mean()
        results[var] = {"p": test.pvalue, f"{cat1}_mean":mean1, f"{cat2}_mean":mean2, "overall_mean":total_mean}
    return pandas.DataFrame(results).T.sort_values(by="p")
res = compare("control", "depression")
display(HTML(res.to_html()))


# In[ ]:


res = compare("depression", "anxiety")
display(HTML(res.to_html()))


# In[ ]:


total = (~mental_health.loc[for_pca.index].date_of_mental_health_questionnaire.isna()).sum()
for c in [col for col in mental_health.columns if ("mental_health_problems_diagnosed" in col)]:
    print(c, mental_health.loc[for_pca.index, c].sum())


# In[ ]:


(mental_health.loc[for_pca.index]['mental_health_problems_diagnosed_Anxiety,_nerves_or_generalized_anxiety_disorder'] *
 mental_health.loc[for_pca.index]['mental_health_problems_diagnosed_Depression']).sum()
control.sum()


# In[ ]:


import glob
import re
activity_files = glob.glob("../processed/acc_analysis/*_90001_0_0-timeSeries.csv.gz")
non_outlier_IDs = []
outlier_IDs = []
for filename in activity_files:
    ID = int(re.search("([\d]*)_90001_0_0-timeSeries.csv.gz", filename).groups()[0])
    if ID in outliers.index:
        if outliers[ID]:
            outlier_IDs.append(ID)
        else:
            non_outlier_IDs.append(ID)
#",".join([f"{x}_90001_0_0-timeSeries.csv.gz" for x in outliers.index[numpy.where(outliers)[0]]])
len(outlier_IDs)


# In[ ]:


# Generate PDFs of the outlier vs non-outlier plots
#NOTE: slowish!
n = 20 # Number of plots to include
pylab.ioff()# Hide plots
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("../figures/activity_non_outliers.pdf")
for ID in non_outlier_IDs[:10]:
    visualize.visualize(f"../processed/acc_analysis/{ID}_90001_0_0-timeSeries.csv.gz", figsize=(9,7), show=False)
    pdf.savefig(fig=pylab.gcf())
pdf.close()
pdf = matplotlib.backends.backend_pdf.PdfPages("../figures/activity_outliers.pdf")
for ID in outlier_IDs[:10]:
    visualize.visualize(f"../processed/acc_analysis/{ID}_90001_0_0-timeSeries.csv.gz", figsize=(9,7), show=False)
    pdf.savefig(fig=pylab.gcf())
    pylab.close()
pdf.close()
pylab.ion()


# In[ ]:


# Display a random plot of activity
import visualize
ID = numpy.random.choice(outlier_IDs)
visualize.visualize(f"../processed/acc_analysis/{ID}_90001_0_0-timeSeries.csv.gz", figsize=(9,7))
i = numpy.where(outliers.index == ID)[0][0]
print("dist_from_pca_center:", dist_from_pca_center[i])


# In[ ]:


# View details of a single individual by ID
ID = 2055584
i = list(for_pca.index).index(ID)
print(f"PCA Coordinates:{pca_coords[i]}")
pandas.DataFrame({"value": for_pca.loc[ID], "zscore": ((for_pca - for_pca.mean(axis=0))/for_pca.std(axis=0)).loc[ID], "mean": for_pca.mean(axis=0), "std": for_pca.std(axis=0)})


# In[ ]:


pylab.figure()
outliers.rename("outlier", inplace=True)
with_outlier = for_pca.join(outliers)
sns.boxplot(y="main_sleep_offset_avg", data=with_outlier, x=category, notch=True)


# In[ ]:


full_activity.main_sleep_onset_avg.describe()


# In[ ]:


pylab.figure()
sns.violinplot(y="main_sleep_onset_avg", data=full_activity, x=full_activity.index.map(ukbb.sex))
#sns.violinplot(y=pandas.Series(dist_from_pca_center, index=for_pca.index), data=for_pca, x=for_pca.index.map(ukbb.sex))
pylab.show()


# In[ ]:




