import argparse
import math
import pathlib

import pandas
import numpy
import pylab
import matplotlib

import util
import activity_features

def visualize(filename):
    filename = pathlib.Path(filename)

    # Get activity features to plot the 'main sleep' times
    data, results, by_day  = activity_features.run(filename, None)

    mask = numpy.ones(data.index.shape).astype(bool)

    grouped = data[mask].groupby(data.index.time)
    day_max = grouped.max()
    day_min = grouped.min()
    day_avg = grouped.mean()

    #### Make figures
    activities = ['main_sleep', 'other_sleep', 'sedentary', 'walking', 'tasks-light', 'moderate', 'imputed']
    activity_colors = ["b", "c", "r", "g", "y", "m", "k"]


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
        # Start at midnight morning of the day until midnight at night of second day
        start = data.index[0].floor("1D") + i * pandas.to_timedelta("1 day")
        end = start + pandas.to_timedelta("2 day")
        day = data.loc[start:end]
        index = (day.index - start).total_seconds() / (60*60)

        # Shade regions based off the inferred activity
        shading_rects = []
        shading_activities = []
        for act, color in zip(activities, activity_colors):
            changes = numpy.diff(numpy.concatenate([[0], (day[act] > 0).astype(int), [0]]))
            starts, = numpy.where(changes > 0)
            stops, = numpy.where(changes < 0)

            for (a, b) in zip(starts, stops):
                first = index[a]
                last = index[min(b, len(day.index)-1)]
                rect = ax.axvspan(first, last, facecolor=color, alpha=0.5)

        # Plot acceleration
        ax.plot(index.values, day.acceleration.values, c='k')

        ax.set_yticks([],[])
        ax.set_ylabel(start.strftime("%a"))
        ax.set_ylim(0,data['acceleration'].max())
        ax.set_xticks([0,6,12,18,24,30,36,42,48])

    axes[0].set_title(filename.name)

    patches = [matplotlib.patches.Patch(color=color, alpha=0.5) for color in activity_colors]
    fig.legend(patches, activities)

    # 24Hour compressed data
    #fig2 = pylab.figure()

    #ax = fig2.add_subplot(111)
    #ax.plot(day_max['acceleration'])
    #ax.plot(day_avg['acceleration'])
    #ax.plot(day_min['acceleration'])
    #ax.set_title(f"Activity through Day\n{filename.name}")

    pylab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Visualize an actigraphy data, preferably downsampled to be, say, 1minute intervals")
    parser.add_argument("-f", "--filename", required=True, help="path(s) to file(s) to load", nargs="+")
    args = parser.parse_args()

    for filename in args.filename:
        visualize(filename)
