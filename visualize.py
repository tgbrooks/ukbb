import argparse
import math

import pandas
import numpy
import pylab

import util

def visualize(filename):
    if data.endswith("csv"):
        data = pandas.read_csv(filename, index_col=0)
    else:
        data = pandas.read_csv(filename, sep="\t", index_col=0)
    data = data.set_index(pandas.to_datetime(data.index))

    data_10min = data.resample("10Min").mean()

    # Group by time of day and
    data_10min["time"] = data_10min.index.time
    mask = util.mask_inactivity(data_10min)

    grouped = data_10min[mask].groupby("time")
    day_max = grouped.max()
    day_min = grouped.min()
    day_avg = grouped.mean()

    #### Make figures
    fig = pylab.figure()

    ax1 = fig.add_subplot(211)
    ax1.stackplot(data_10min.index, util.mask_inactivity(data_10min))
    #ax.stackplot(data_10min.index, numpy.log10(1+data_10min["Lux"]))
    #ax.stackplot(data_10min.index, data_10min["Inclinometer Off"], data_10min["Inclinometer Standing"], data_10min["Inclinometer Sitting"], data_10min["Inclinometer Lying"])

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(data_10min['Vector Magnitude'], color="k")

    # stacked days, double-plotted
    num_days = math.ceil((data.index[-1] - data.index[0]).total_seconds() / (24*60*60)) + 1
    fig, axes = pylab.subplots(nrows=num_days, sharex=True)
    for i, ax in enumerate(axes):
        start = pandas.to_datetime(data_10min.index[0].date()) + i * pandas.to_timedelta("1 day")
        end = start + pandas.to_timedelta("2 day")
        day = data_10min['Vector Magnitude'].loc[start:end]
        day.index = (day.index - start).total_seconds() / (60*60)

        #ax = fig.add_subplot(num_days, 1, i+1, sharex=last_ax)
        ax.plot(day.index, day, color="k")
        last_ax = ax

        ax.set_yticks([],[])
        ax.set_ylabel(start.strftime("%a"))
        ax.set_ylim(0,data_10min['Vector Magnitude'].max())

    # 24Hour compressed data
    fig2 = pylab.figure()

    ax = fig2.add_subplot(111)
    ax.plot(day_max['Vector Magnitude'])
    ax.plot(day_avg['Vector Magnitude'])
    ax.plot(day_min['Vector Magnitude'])
    ax.set_title("Activity through Day")

    pylab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Visualize an actigraphy data, preferably downsampled to be, say, 1minute intervals")
    parser.add_argument("-f", "--filename", required=True, help="path to file to load")
    args = parser.parse_args()

    visualize(args.filename)
