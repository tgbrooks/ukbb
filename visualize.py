import argparse
import math

import pandas
import numpy
import pylab

import util

def visualize(filename):
    if filename.endswith("csv") or filename.endswith(".csv.gz"):
        data = util.read_actigraphy(filename)
    else:
        data = pandas.read_csv(filename, sep="\t", index_col=0, parse_dates=[0])

    #data = data.resample("10Min").mean()

    # Group by time of day and
    #data["time"] = data.index.time
    mask = numpy.ones(data.index.shape).astype(bool)

    grouped = data[mask].groupby(data.index.time)
    day_max = grouped.max()
    day_min = grouped.min()
    day_avg = grouped.mean()

    #### Make figures
    activities = ['sleep', 'sedentary', 'walking', 'tasks-light', 'moderate', 'imputed']
    activity_colors = ["b", "r", "g", "y", "c", "k"]

    # Long plot
    #fig = pylab.figure()

    #ax1 = fig.add_subplot(211)
    #ax1.stackplot(data.index, [data[act] for act in activities], labels=activities)

    #ax2 = fig.add_subplot(212, sharex=ax1)
    #ax2.plot(data['acceleration'])
    #fig.legend()

    # stacked days, double-plotted
    num_days = math.ceil((data.index[-1] - data.index[0]).total_seconds() / (24*60*60)) + 1
    fig, axes = pylab.subplots(nrows=num_days, sharex=True)
    for i, ax in enumerate(axes):
        start = pandas.to_datetime(data.index[0].date()) + i * pandas.to_timedelta("1 day")
        end = start + pandas.to_timedelta("2 day")
        day = data.loc[start:end]
        index = (day.index - start).total_seconds() / (60*60)

        # Shade regions based off the inferred activity
        shading_rects = []
        for act, color in zip(activities, activity_colors):
            changes = numpy.diff(numpy.concatenate([[0], (day[act] > 0).astype(int), [0]]))
            starts, = numpy.where(changes > 0)
            stops, = numpy.where(changes < 0)
            for (a, b) in zip(starts, stops):
                first = index[a]
                last = index[min(b, len(day.index)-1)]
                rect = ax.axvspan(first, last, facecolor=color, alpha=0.5)
            # Grab one rect per activity for legend
            # only do this for the first day
            shading_rects.append(rect)

        # Plot acceleration
        ax.plot(index.values, day.acceleration.values, c='k')

        ax.set_yticks([],[])
        ax.set_ylabel(start.strftime("%a"))
        ax.set_ylim(0,data['acceleration'].max())
        ax.set_xticks([0,6,12,18,24,30,36,42,48])

    fig.legend(shading_rects, activities)

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
