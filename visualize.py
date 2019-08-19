import argparse
import math

import pandas
import numpy
import pylab

import util

def visualize(filename):
    if filename.endswith("csv"):
        data = util.read_actigraphy(filename)
    else:
        data = pandas.read_csv(filename, sep="\t", index_col=0, parse_dates=[0])

    data_10min = data.resample("10Min").mean()

    # Group by time of day and
    data_10min["time"] = data_10min.index.time
    mask = numpy.ones(data_10min.index.shape).astype(bool)

    grouped = data_10min[mask].groupby("time")
    day_max = grouped.max()
    day_min = grouped.min()
    day_avg = grouped.mean()

    #### Make figures
    fig = pylab.figure()

    ax1 = fig.add_subplot(211)
    ax1.stackplot(data_10min.index, [data_10min.sleeping, data_10min.walking, data_10min.sedentary, data_10min['light-tasks'], data_10min['moderate'])

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(data_10min['acceleration'])

    # stacked days, double-plotted
    num_days = math.ceil((data.index[-1] - data.index[0]).total_seconds() / (24*60*60)) + 1
    fig, axes = pylab.subplots(nrows=num_days, sharex=True)
    for i, ax in enumerate(axes):
        start = pandas.to_datetime(data_10min.index[0].date()) + i * pandas.to_timedelta("1 day")
        end = start + pandas.to_timedelta("2 day")
        day = data_10min['acceleration'].loc[start:end]
        index = (day.index - start).total_seconds() / (60*60)

        #ax = fig.add_subplot(num_days, 1, i+1, sharex=last_ax)
        ax.plot(index.values, day.values)
        last_ax = ax

        ax.set_yticks([],[])
        ax.set_ylabel(start.strftime("%a"))
        ax.set_ylim(0,data_10min['acceleration'].max())

    # 24Hour compressed data
    fig2 = pylab.figure()

    ax = fig2.add_subplot(111)
    ax.plot(day_max['acceleration'])
    ax.plot(day_avg['acceleration'])
    ax.plot(day_min['acceleration'])
    ax.set_title("Activity through Day")

    pylab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Visualize an actigraphy data, preferably downsampled to be, say, 1minute intervals")
    parser.add_argument("-f", "--filename", required=True, help="path to file to load")
    args = parser.parse_args()

    visualize(args.filename)
