import argparse
import math
import pathlib

import pandas
import numpy
import pylab
import matplotlib

import util
import activity_features

def visualize(filename, show=True, **kwargs):
    filename = pathlib.Path(filename)

    # Get activity features to plot the 'main sleep' times
    data, results, by_day  = activity_features.run(filename, None)

    mask = numpy.ones(data.index.shape).astype(bool)

    grouped = data[mask].groupby(data.index.time)
    day_max = grouped.max()
    day_min = grouped.min()
    day_avg = grouped.mean()

    #### Make figures
    activities = ['main_sleep', 'other_sleep', 'sedentary', 'walking', 'tasks_light', 'moderate', 'imputed']
    activity_colors = ["b", "c", "r", "g", "y", "m", "k"]

    # stacked days, double-plotted
    num_days = (data.index[-1].date() - data.index[0].date()).days + 1
    fig, axes = pylab.subplots(nrows=num_days, sharex=True, **kwargs)
    for i, ax in enumerate(axes):
        # Start at midnight morning of the day until midnight at night of second day
        # If daylight savings crossover happens then affected days are 1 hour shorter or longer
        # (due to double-plotting this affects two days)
        start = pandas.to_datetime(data.index[0].date() + i * pandas.to_timedelta("1 day")).tz_localize(data.index.tz)
        end = start + pandas.to_timedelta("2 day")
        day = data.loc[start:end]
        index = (day.index - day.index[0].normalize()).total_seconds() / (60*60)

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
        ax.plot(index.values, numpy.log10(day.acceleration.values + 1), c='k')

        # Plot cosinor fit
        #cosinor = numpy.cos((index - results['phase']) * 2 * numpy.pi / 24) * results['amplitude'] + results['mesor']
        #ax.plot(index.values, numpy.log10(cosinor + 1), c='k', linestyle="--")
        cosinor = numpy.cos((index - results['phase']) * 2 * numpy.pi / 24) * results['amplitude'] + results['mesor']
        ax.plot(index.values, cosinor, c='k', linestyle="--")

        # Plot light
        ax_light = ax.twinx()
        ax_light.plot(index.values, day.light.values, c='w')
        ax_light.set_yticks([],[])
        ax_temp = ax.twinx()
        ax_temp.plot(index.values, day.temp.values, c='grey')
        ax_temp.set_yticks([],[])

        ax.set_yticks([],[])
        ax.set_ylabel(start.strftime("%a") + "\n" + start.strftime("%y-%m-%d"))
        #ax.set_ylim(0,data['acceleration'].max())
        ax.set_xticks([0,6,12,18,24,30,36,42,48])

    axes[0].set_title(filename.name)

    painters = [matplotlib.patches.Patch(color=color, alpha=0.5, label=act) for color, act in zip(activity_colors, activities)]
    painters.append(matplotlib.lines.Line2D([0],[0], color="k", label="accel"))
    painters.append(matplotlib.lines.Line2D([0],[0], color="w", label="light"))
    painters.append(matplotlib.lines.Line2D([0],[0], color="grey", label="temp"))
    fig.legend(painters, activities + ['accel', 'light', 'temp'])

    # 24Hour compressed data
    #fig2 = pylab.figure()

    #ax = fig2.add_subplot(111)
    #ax.plot(day_max['acceleration'])
    #ax.plot(day_avg['acceleration'])
    #ax.plot(day_min['acceleration'])
    #ax.set_title(f"Activity through Day\n{filename.name}")

    if show:
        pylab.show()

    return data, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Visualize an actigraphy data, preferably downsampled to be, say, 1minute intervals")
    parser.add_argument("-f", "--filename", required=True, help="path(s) to file(s) to load", nargs="+")
    args = parser.parse_args()

    for filename in args.filename:
        data, results = visualize(filename)
