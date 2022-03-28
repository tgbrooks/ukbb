'''
Script to generate an actigraphy trace plot of one individual
Shows acceleration, light, temperature levels and highlights
based off of the classified activity types (sleep, walking, etc.)
'''
import argparse
import math
import pathlib

import pandas
import numpy
import pylab
import matplotlib

import util
import activity_features

def visualize(filename, show=True, M10_view=False, temperature=False, **kwargs):
    filename = pathlib.Path(filename)

    # Get activity features to plot the 'main sleep' times
    data, results, by_day  = activity_features.run(filename, None)

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
        index = (day.index - day.index[0].normalize()) / pandas.to_timedelta('1H')

        if not temperature:
            # Plot acceleration
            ax.plot(index.values, numpy.log10(day.acceleration.values + 1), c='k')

            # Plot cosinor fit
            #cosinor = numpy.cos((index - results['phase']) * 2 * numpy.pi / 24) * results['amplitude'] + results['mesor']
            #ax.plot(index.values, numpy.log10(cosinor + 1), c='k', linestyle="--")
            cosinor = numpy.cos((index - results['phase']) * 2 * numpy.pi / 24) * results['amplitude'] + results['mesor']
            ax.plot(index.values, cosinor, c='k', linestyle="--")
        else:
            # Plot temperatures
            ax.plot(index.values, day.temp.values, c='k')
            # Cosinor tempereature fit
            cosinor = numpy.cos((index - results['temp_phase']) * 2 * numpy.pi / 24) * results['temp_amplitude'] + results['temp_mesor']
            ax.plot(index.values, cosinor, c='grey', linestyle="--")

        if not M10_view:
            # Shade regions based off the inferred activity
            for act, color in zip(activities, activity_colors):
                changes = numpy.diff(numpy.concatenate([[0], (day[act] > 0).astype(int), [0]]))
                starts, = numpy.where(changes > 0)
                stops, = numpy.where(changes < 0)

                for (a, b) in zip(starts, stops):
                    first = index[a]
                    last = index[min(b, len(day.index)-1)]
                    rect = ax.axvspan(first, last, facecolor=color, alpha=0.5)

            # Plot light
            ax_light = ax.twinx()
            ax_light.plot(index.values, day.light.values, c='w')
            ax_light.set_yticks([],[])
            ax_light.set_yticks([],minor=[])
            ax_temp = ax.twinx()
            ax_temp.plot(index.values, day.temp.values, c='grey')
            ax_temp.set_yticks([],minor=[])

            ax.set_yticks([],minor=[])
            ax.set_ylabel(start.strftime("%a") + "\n" + start.strftime("%y-%m-%d"))
            ax.set_xticks([0,6,12,18,24,30,36,42,48])
            ax.set_xlim([0,48])

        if M10_view:
            # Plot the M10/L5 parts
            M10_start = results['acceleration_M10_time'] - 5
            M10_end = results['acceleration_M10_time'] + 5
            L5_start = results['acceleration_L5_time'] - 2.5
            L5_end = results['acceleration_L5_time'] + 2.5
            M10_color = (1.0, 0.4980392156862745, 0.054901960784313725, 1.0)
            L5_color = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
            ax.axvspan(M10_start,
                        M10_end,
                        facecolor=M10_color)
            ax.axvspan(L5_start,
                        L5_end,
                        facecolor=L5_color)
            ax.axvspan(M10_start + 24,
                        M10_end + 24,
                        facecolor=M10_color)
            ax.axvspan(L5_start + 24,
                        L5_end + 24,
                        facecolor=L5_color)

            hour_plots_args = {
                'linestyle':"-",
                'color':"white",
                'alpha':1.0,
                'zorder':1,
            }
            for i in range(11):
                ax.axvline(M10_start + i, **hour_plots_args)
                ax.axvline(M10_start + i + 24, **hour_plots_args)

            for i in range(6):
                ax.axvline(L5_start + i, **hour_plots_args)
                ax.axvline(L5_start + i + 24, **hour_plots_args)

            ax.set_yticks([],minor=[])
            #ax.set_ylabel(start.strftime("%a") + "\n" + start.strftime("%y-%m-%d")) # With dates
            ax.set_ylabel(start.strftime("%a")) # Just day-of-week
            ax.set_xticks([0,4,8,12,16,20,24])
            ax.set_xlim([0,24])


    axes[0].set_title(filename.name)

    # Legends
    if not M10_view:
        painters = [matplotlib.patches.Patch(color=color, alpha=0.5, label=act) for color, act in zip(activity_colors, activities)]
        painters.append(matplotlib.lines.Line2D([0],[0], color="k", label="accel"))
        painters.append(matplotlib.lines.Line2D([0],[0], color="w", label="light"))
        painters.append(matplotlib.lines.Line2D([0],[0], color="grey", label="temp"))
        fig.legend(painters, activities + ['accel', 'light', 'temp'])
    else:
        label = "accel" if not temperature else "temp"
        painters = [matplotlib.patches.Patch(color=M10_color, label="M10"),
                    matplotlib.patches.Patch(color=L5_color, label="L5"),]
        painters.append(matplotlib.lines.Line2D([0],[0], color="k", label=label))
        fig.legend(painters, ['M10', 'L5', label])

        ## Annotations
        #ax.annotate("Between-day SD",
        #            xy = (0.9,0.2), xycoords="figure fraction",
        #            xytext=(10,0), textcoords="offset pixels",
        #            verticalalignment='center',
        #            arrowprops={
        #                'arrowstyle': '-[, widthB=2.0',
        #            })


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
    parser.add_argument("--temperature", required=False, default=False, action="store_const", const=True, dest="temperature", help="Emphasize temperature not activity")
    parser.add_argument("--M10_view", required=False, default=False, action="store_const", const=True, dest="M10_view", help="Highlight M10/L5 areas")
    args = parser.parse_args()

    for filename in args.filename:
        data, results = visualize(filename, M10_view=args.M10_view, temperature=args.temperature)
