import pandas
import pylab
import numpy

from sklearn.decomposition import PCA
import scipy.stats
from scipy.cluster import hierarchy

ukbb_data = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)
has_questionnaire = ~ukbb_data.date_of_mental_health_questionnaire.isna()
numeric_data = ukbb_data.select_dtypes(include=['number', 'category', 'bool'])[has_questionnaire]

#Drop, for now, any -999 entries which are "too many to count" values
numeric_data[numeric_data == -999] = float("NaN")

# Select only those columns that are highly present
# this mostly just drops a handful of collumns that I don't
# know how to handle correctly yet
numeric_data = numeric_data.loc[:,numeric_data.count() > 100_000]

# Drop, for now, any rows that didn't answer any question
numeric_data = numeric_data[~numeric_data.isna().any(axis=1)]

zscores = (numeric_data - numeric_data.mean(axis=0)) / numeric_data.std(axis=0)




pca = PCA(n_components=5)
coords = pca.fit_transform(zscores)


# Visually confirmed that
#PCA1 is depression + anxiety spectrum
#PCA2 is depression vs. anxiety spectrum

fig = pylab.figure(figsize=(12,8))
ax = fig.add_subplot(111)
im = ax.scatter(*(coords[:,:2].T), c=numeric_data.general_happiness)
ax.set_title("Depression Variables PCA colored by 'general happiness'")
fig.colorbar(im, ax=ax)
fig.savefig("../processed/figures/depression/depression_pca.png")


pca1_correlations = pandas.Series([scipy.stats.pearsonr(coords[:,0], numeric_data[column])[0] for column in numeric_data], index=numeric_data.columns)


dist = scipy.spatial.distance.pdist(zscores.T, metric="correlation")
dist = 1 - numpy.abs(1 - dist)
linkage = hierarchy.linkage(dist)
fig = pylab.figure(figsize=(14,8))
dendro = hierarchy.dendrogram(linkage, labels=zscores.columns, leaf_font_size=10)
fig.subplots_adjust(bottom=0.5)
fig.savefig("../processed/figures/depression/depression_hierarchical.png")

pylab.show()
