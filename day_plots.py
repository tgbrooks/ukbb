"""
Create plots for daily 24 hour average traces (of acceleration values or temperature, etc)

While visualize.py does individual trace plots, this averages over many individuals
"""
import pathlib
import functools

import pandas
import numpy
import pylab

import activity_features

def get_ids_of_traces_available(directory="../processed/acc_analysis"):
    dir = pathlib.Path(directory)
    return [int(f.name.split("_")[0]) for f in dir.glob("*_90001_0_0-timeSeries.csv.gz")]

@functools.lru_cache(None)
def get_tracedata(tracefile):
        return activity_features.load_activity_file(tracefile)

def plot_average_trace(ids, var="acceleration", directory="../processed/acc_analysis/", transform = lambda x: x,
            normalize_mean=False, set_mean=None, ax=None, color="k", label=None, show_variance=True,
            show_confidence_intervals=False):
    average_traces = []
    for id in ids:
        tracefile = directory+str(id)+"_90001_0_0-timeSeries.csv.gz"
        tracedata = get_tracedata(tracefile)

        if tracedata is None:
            print(f"Skipping {id}")
            continue

        # Trim the first and last 6 hours of data
        # due to apparent discontinuities around them
        tracedata = tracedata[tracedata.index.min() + pandas.to_timedelta("6H"):
                              tracedata.index.max() - pandas.to_timedelta("6H")]

        # Time since midnight
        timeofday = (tracedata.index - tracedata.index.normalize())
        average_trace = tracedata.set_index(timeofday).resample("1min").mean()
        average_traces.append(transform(average_trace[[var]]))
    if normalize_mean:
        # Make sure everyone's mean is the same
        if set_mean is not None:
            grand_mean = set_mean
        else:
            grand_mean = pandas.DataFrame([trace.mean() for trace in average_traces]).mean()
        average_traces = [trace - trace.mean() + grand_mean for trace in average_traces]
    resampled = pandas.concat(average_traces).resample("1min")[var]
    grand_average = resampled.mean()


    if ax is None:
        fig, ax = pylab.subplots()
    ax.plot(grand_average.index/ pandas.to_timedelta("1H"), grand_average, c=color, label=label)

    if show_variance:
        low_average = resampled.quantile(0.25)
        high_average = resampled.quantile(0.75)
        ax.fill_between(low_average.index / pandas.to_timedelta("1H"),
                        low_average,
                        high_average,
                        facecolor=color,
                        alpha=0.25,
                        )

    if show_confidence_intervals:
        sem = resampled.sem()
        low = grand_average - 1.96 * sem
        high = grand_average + 1.96 * sem
        ax.fill_between(low.index / pandas.to_timedelta("1H"),
                        low,
                        high,
                        facecolor=color,
                        alpha=0.25,
                        )

    #ax.plot(low_average.index / pandas.to_timedelta("1H"), low_average, label="75th percentile")
    #ax.plot(high_average.index / pandas.to_timedelta("1H"), high_average, label="25th percentile")
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time of Day")
    ticks = numpy.arange(0,25,4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick}:00" for tick in ticks])
    return average_traces, grand_average, ax
