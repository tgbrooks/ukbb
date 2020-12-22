"""
Create plots for daily 24 hour average traces (of acceleration values or temperature, etc)
"""
import pathlib

import pandas
import numpy
import pylab

import activity_features

def get_ids_of_traces_available(directory="../processed/acc_analysis"):
    dir = pathlib.Path(directory)
    return [int(f.name.split("_")[0]) for f in dir.glob("*_90001_0_0-timeSeries.csv.gz")]

def plot_average_trace(ids, var="acceleration", directory="../processed/acc_analysis/", transform = lambda x: x, normalize_mean=False, ax=None):
    average_traces = []
    for id in ids:
        tracefile = directory+str(id)+"_90001_0_0-timeSeries.csv.gz"
        tracedata = activity_features.load_activity_file(tracefile)

        if tracedata is None:
            print("Skipping {id}")
            continue

        # Time since midnight
        timeofday = (tracedata.index - tracedata.index.normalize())
        average_trace = tracedata.set_index(timeofday).resample("1min").mean()
        average_traces.append(transform(average_trace[[var]]))
    if normalize_mean:
        # Make sure everyone's mean is the same
        grand_mean = pandas.DataFrame([trace.mean() for trace in average_traces]).mean()
        average_traces = [trace - trace.mean() + grand_mean for trace in average_traces]
    resampled = pandas.concat(average_traces).resample("1min")[var]
    grand_average = resampled.mean()
    low_average = resampled.quantile(0.25)
    high_average = resampled.quantile(0.75)


    if ax is None:
        fig, ax = pylab.subplots()
    ax.plot(grand_average.index/ pandas.to_timedelta("1H"), grand_average, label="mean", c="k")
    ax.fill_between(low_average.index / pandas.to_timedelta("1H"),
                       low_average,
                       high_average,
                       color="#ccc",
                       )
    #ax.plot(low_average.index / pandas.to_timedelta("1H"), low_average, label="75th percentile")
    #ax.plot(high_average.index / pandas.to_timedelta("1H"), high_average, label="25th percentile")
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time of Day")
    ticks = numpy.arange(0,25,4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick}:00" for tick in ticks])
    return average_traces, grand_average, ax