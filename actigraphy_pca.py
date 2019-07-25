#!/usr/bin/env python
import pandas
import numpy
from sklearn.decomposition import PCA

import pylab

summary = pandas.read_csv("results/actigraphy_summary.txt", sep="\t", index_col=0)

# Expect L5 times to be around midnight, so convert them to be in "time since noon" not "time since midnight"
summary.L5_time = (summary.L5_time - 12) % 24

zscore_summary = (summary - summary.mean(axis=0)) / summary.std(axis=0)
valid = ~zscore_summary.isna().any(axis=1)
pca = PCA(n_components=2)
coords = pca.fit_transform(zscore_summary[valid])

def on_pick(event):
    ind = event.ind
    print(f"EID: {summary.index[ind]}")

for var in summary.columns:
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.scatter(*(coords.T), c=summary[valid][var], picker=True)

    fig.canvas.mpl_connect("pick_event", on_pick)
    pylab.title(f"Colored by {var}")

    pylab.show()
