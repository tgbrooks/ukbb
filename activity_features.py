import pandas
import numpy

SAMPLE_RATE = 30 # in Seconds
HOURS_TO_COUNTS = 60*60//SAMPLE_RATE
COUNTS_PER_DAY = 24*HOURS_TO_COUNTS
SLEEP_PERIOD = 8*HOURS_TO_COUNTS

MIN_PERIOD_LENGTH = 10 # 5 minutes of sleep consecutively to count as a sleep period
JUMP_RATIO = 0.5 # Length of longest gap to jump over as a ratio of the size of the adjacent sleep periods
RIGHT_JUMP_RATIO = 1.5
MAX_GAP_LENGTH = 2.0 * HOURS_TO_COUNTS
GOOD_COUNT_THRESHOLD = 0.75
SLEEP_THRESHOLD = 0.9 # May not be necessary - 30second intervals seem to generally have sleep=0 or 1

def hours_since_noon(timeseries):
    ''' Return time since the last noon, in hours

    NOTE: if a clock change happens due to daylight savings change
    then this will still give the total number of hours since noon.'''
    timeseries = pandas.to_datetime(timeseries)
    noon_same_day = timeseries.dt.round("1D") + pandas.to_timedelta("0.5D")
    difference = (timeseries - noon_same_day).dt.total_seconds() / 60 / 60

    noon_last_day = noon_same_day - pandas.to_timedelta("1D")
    difference_last_day = (timeseries - noon_last_day).dt.total_seconds() / 60 / 60
    difference[difference < 0] = difference_last_day[difference < 0]
    return difference

def extract_main_sleep_period(day_data):
    sleep = day_data.sleep
    # Binarize slepe (it seems to be 0 or 1 already aynway)
    # and this also converts NaNs to 1, i.e. to sleeping which seems to be a common
    # mistake of the calling algorithm, to give non-wear time to sleep episodes during the night
    # (only will matter for computing the main sleep period, not for computing total sleep)
    sleeping = (~(sleep.values < SLEEP_THRESHOLD)).astype('int')

    # Find periods of sleep
    diff = numpy.diff(numpy.concatenate([[0], sleeping, [0]]))
    onset_times, = numpy.where(diff > 0)
    offset_times, = numpy.where(diff < 0)


    # Skip any sleep periods of very short length
    #period_lengths = offset_times - onset_times
    #skip = period_lengths < MIN_PERIOD_LENGTH
    #onset_times = onset_times[~skip]
    #offset_times = offset_times[~skip]

    if len(onset_times) == 0:
        # Never slept at all
        return dict(onset=pandas.to_datetime("NaT"), offset=pandas.to_datetime("NaT"), num_wakings=float("NaN"), WASO=float("NaN"), acceleration_during_main_sleep=float("NaN"))

    # Lengths of each sleep period
    period_lengths = offset_times - onset_times

    # Time between consecutive periods
    gap_lengths = onset_times[1:] - offset_times[:-1]

    # Join sleep periods by:
    # iterating to the right (later) from each period, grabbing any period
    # that is sufficiently close. The total non-sleep time between joined periods
    # must never be more than MAX_GAP_LENGTH. And the lengths of the periods being
    # joined together must be large enough compared to the gap
    gaps_to_join = numpy.zeros(shape=gap_lengths.shape, dtype=bool)
    for i in range(len(gap_lengths)):
        joined_period_length = period_lengths[i]

        furthest_jump = i
        jump_length = 0
        for j in range(i+1, len(period_lengths)):
            # Accumulate the jump length by this gap
            jump_length += gap_lengths[j-1]

            if jump_length > MAX_GAP_LENGTH or jump_length >= JUMP_RATIO * joined_period_length:
                # Searched past the max gap length, so need to stop
                break

            # Add this period to the lengths of those joined so far
            # Note: we may not actually end up joining this period
            # but we need to include its length in the next step
            joined_period_length += period_lengths[j]

            if jump_length >= RIGHT_JUMP_RATIO * period_lengths[j]:
                # Too short a period to jump to
                continue

            # We've accepted the jump to at least this far now
            furthest_jump = j

        # Join all gaps (possibly 0, possibly many) that span between these two periods
        gaps_to_join[i:furthest_jump] = True

    # Join together all periods whose gaps match the above criteria
    onset_times = numpy.concatenate([onset_times[:1], onset_times[1:][~gaps_to_join]])
    offset_times = numpy.concatenate([offset_times[:-1][~gaps_to_join], offset_times[-1:]])

    # Extract main (longest) period
    period_lengths = offset_times - onset_times
    best_period = numpy.argmax(period_lengths)

    onset_index = onset_times[best_period]
    offset_index = offset_times[best_period]

    onset = sleep.index[0] + pandas.to_timedelta( str(onset_index / HOURS_TO_COUNTS) + "H")
    offset = sleep.index[0] + pandas.to_timedelta( str(offset_index / HOURS_TO_COUNTS) + "H")

    # Number of times woken up
    # counted as number of transitions from sleep
    num_wakings = numpy.sum(numpy.diff(numpy.concatenate([[0], sleeping[onset_index:offset_index] > SLEEP_THRESHOLD, [0]])) < 0) - 1
    WASO = numpy.sum(1 - sleeping[onset_index:offset_index]) / HOURS_TO_COUNTS

    # Average acceleration duing main sleep
    acceleration_during_main_sleep = day_data.acceleration[onset_times[best_period]:offset_times[best_period]].mean()

    return dict(onset=onset,
                offset=offset,
                num_wakings=num_wakings,
                WASO=WASO,
                acceleration_during_main_sleep=acceleration_during_main_sleep)

def RA(activity):
    ''' Compute Relative Amplitude of any activity feature

    (M10 - L5) / (M10 + L5)
    where M10 = average activity in highest 10 hours of day
    L5 = average activity in lowest 5 hours of day
    NOTE that this may differ from RA computations done by others
    because it uses the AVERAGE over all days and then looks for the
    M10 and L5 of that, versus looking for the M10, L5 of each day and
    then using those.This avoids the difficult questions of when one day
    ends and another begins - which is generally not well specified in
    descriptions of RA. However, this makes the RA value dependent upon
    the number of days of data collected (going down with more days
    in general, until very large numbers of days are available).
    '''

    # Average over all days of the activity at a specific time-of-day
    average = activity.groupby(activity.index.time).mean()
    # Pad so that the rolling average can 'wrap around' midnights
    average = pandas.concat([average, average, average], axis=0)

    M10 = average.rolling(10 * HOURS_TO_COUNTS).mean().max()
    L5 = average.rolling(5* HOURS_TO_COUNTS).mean().min()

    return (M10 - L5) / (M10 + L5)

def IS(activity):
    '''
    Interdaily stability value for the given measure

    Calculated as from https://www.sciencedirect.com/science/article/pii/0006322390905235
    '''
    hourly_avg = activity.groupby(activity.index.hour).mean() # Average activity in a specific time of day
    IS = ((hourly_avg - activity.mean())**2).mean() / hourly_avg.var(ddof=0)
    return IS

def IV(activity):
    '''
    Intradaily variability value of the given measure

    Calculated as from https://www.sciencedirect.com/science/article/pii/0006322390905235
    '''

    hourly_avg = activity.groupby([activity.index.hour, activity.index.date]).mean() # Average activity within a specific hour
    IV = (hourly_avg.diff(1)**2).mean() / hourly_avg.var(ddof=0)
    return IV


def activity_features(data):
    ''' return dictionary of result summary values '''

    results = {}

    ### Sleep Features
    if 'sleep' in data:
        stretches = data.rolling(SLEEP_PERIOD, center=True).mean()

        # Sleep onset
        # For, say, Monday, we look for sleep periods starting between noon Monday and noon Tuesday
        # since we want to put a sleep-onset time of 1AM on Tuesday as being the onset "for Monday"
        # this is done with the "base=0.5" parameter, offsets the 1day samples by 12 hours
        sleep_peak_time = stretches.resample("1D", base=0.5).sleep.idxmax()
        sleep_peak_quality = stretches.sleep.resample("1D", base=0.5).max()
        sleep_peak_data_available = stretches.resample("1D", base=0.5).sleep.count() > (COUNTS_PER_DAY * GOOD_COUNT_THRESHOLD)

        # Define the MAIN SLEEP PERIOD to be the period of time such that
        # there are no periods > 1 HR of time without sleep
        # and that maximizes the total amount of sleep in that period over the entire day (ie. of ones starting in a noon-to-noon)
        by_day = data.resample("1D", base=0.5)
        main_sleep_period = by_day.apply(extract_main_sleep_period)
        main_sleep_period_df = pandas.DataFrame(main_sleep_period.tolist(), index=main_sleep_period.index)
        main_sleep_onset, main_sleep_offset, main_sleep_wakings, main_sleep_WASO, acceleration_during_main_sleep = [main_sleep_period_df[col].copy() for col in main_sleep_period_df.columns]

        main_sleep_duration = (main_sleep_offset - main_sleep_onset).dt.total_seconds()/60/60
        main_sleep_ratio = (main_sleep_duration - main_sleep_WASO)/ main_sleep_duration

        # Define sleep as either 'main sleep' or 'other sleep'
        data['main_sleep_period'] = False
        for day in main_sleep_onset.index:
            if pandas.isna(main_sleep_onset[day]):
                continue
            data.loc[main_sleep_onset[day]:main_sleep_offset[day], 'main_sleep_period'] = True
        data['main_sleep'] = (data.sleep > 0) & data.main_sleep_period
        data['other_sleep'] = (data.sleep > 0) & (~data.main_sleep_period)
        # For computing 'other' times, we want the data to be for a midnight-to-midnight day
        # not the noon-to-noon day that we use for sleep periods
        by_midnight_day = data.resample("1D", base=0.0)

        # Collect day-level results
        results_by_day = main_sleep_period_df
        results_by_day['main_sleep_duration'] = by_day.main_sleep.sum() / HOURS_TO_COUNTS
        results_by_day['other_sleep'] = (by_midnight_day.other_sleep.sum() / HOURS_TO_COUNTS).values
        # Total sleep is napping between that days midnight-midnight period
        # plus actual sleep in the main sleep from the noon-noon period
        results_by_day['total_sleep'] = results_by_day.main_sleep_duration - results_by_day.WASO + results_by_day.other_sleep
        results_by_day['main_sleep_ratio'] = (results_by_day.main_sleep_duration - results_by_day.WASO) / results_by_day.main_sleep_duration

        # Throw out days without nearly all of the hours
        # eg: if the data starts at Monday 10:00am, we don't want to consider Sunday noon - Monday noon a day
        # should have 2880 for a complete day
        days_invalid = (by_day.sleep.count() < 2500)
        for measure in [main_sleep_onset, main_sleep_offset, main_sleep_wakings, main_sleep_WASO, main_sleep_duration, main_sleep_ratio, acceleration_during_main_sleep]:
            measure[days_invalid] = float("NaN")


        # Collect summary level results, across all days
        results.update(dict(
            # hours past 0:00 AM (usually of the day prior to the peak sleep)
            sleep_peak_time_avg = hours_since_noon(sleep_peak_time).mean() + 12,
            sleep_peak_time_std = hours_since_noon(sleep_peak_time).std(),

            # Onset/offset times, from 0:00AM morning of the first day
            onset_time_avg = hours_since_noon(main_sleep_onset).mean() + 12,
            onset_time_std = hours_since_noon(main_sleep_onset).std(),
            offset_time_avg = hours_since_noon(main_sleep_offset).mean() + 12,
            offset_time_std = hours_since_noon(main_sleep_offset).std(),

            # Duration of main period of sleep NOT the total amount of time they slept - may have many waking periods
            main_sleep_duration_avg = main_sleep_duration.mean(),
            main_sleep_duration_std = main_sleep_duration.std(),

            # Number of times waking up during main sleep period
            main_sleep_wakings_avg = main_sleep_wakings.mean(),
            main_sleep_wakings_std = main_sleep_wakings.std(),

            # Total time in hours spent not sleeping during main sleep period
            main_sleep_WASO_avg = main_sleep_WASO.mean(),
            main_sleep_WASO_std = main_sleep_WASO.std(),

            # Sleep ratio: fraction of main sleep period spent sleeping
            main_sleep_ratio_avg = main_sleep_ratio.mean(),
            main_sleep_ratio_std = main_sleep_ratio.std(),

            # Time spent sleeping NOT in main sleep period
            other_sleep_duration_avg = by_midnight_day.other_sleep.sum().mean() / HOURS_TO_COUNTS,
            other_sleep_duration_std = by_midnight_day.other_sleep.sum().std() / HOURS_TO_COUNTS,

            #Average acceleration during the main sleep period
            acceleration_during_main_sleep_avg = acceleration_during_main_sleep.mean(),
            acceleration_during_main_sleep_std = acceleration_during_main_sleep.std()
        ))

    # Add RA/IS/IV values for each measure
    for activity in ['sleep', 'walking', 'sedentary', 'moderate', 'acceleration', 'tasks-light', 'MET', 'temp', 'light']:
        try:
            results.update({
                activity + "_RA": RA(data[activity]),
                activity + "_IS": IS(data[activity]),
                activity + "_IV": IV(data[activity]),
            })
        except KeyError:
            #Some datasets do not have MET or other columns
            #so we just drop them here
            #these will end up as NaNs in the aggregated spreadsheet
            pass

    # Light and temperature values
    for value in ['light', 'temp']:
        results.update({
            value + "_avg": data[value].mean(),
            value + "_std": data[value].std(),
        })

    # Attempt to estimate daylight
    # Since light sensors require calibration which we do not know we estimate a threshold for daylight
    # by using the minimum and highest values observed
    # since daylight is generally orders of magnitude brighter than anything else this should be reliable
    # so long as at least one 30second epoch was exposed to daylight
    # though there is the concern that some will get daylight and others will get direct sunlight which could be much much brighter
    # NOTE: I have decided to not attempt to infer daylight for now
    # the data here shows that the (uncalibrated) lux range of participants is roughly 15 - 500.
    # Real sunlight should be closer to 10000 or more. The enclosure blocks some amount of light but this is more atenuated than expeceted
    # If there is that little difference then we can't reliably differentiate sunlight from office light

    #daylight_threshold = (data.light.max() - data.light.min()) / 10 + data.light.min()

    return results, results_by_day

def run(input, output):
    '''Compute and output activity features'''
    # Load data
    data = pandas.read_csv(input, parse_dates=[0])
    if 'time' not in data:
        print(f"Processed timeseries file {input} has no 'time' index collumn and is being skipped")
        results = {}
    else:
        # Process the timezones from UTC to London
        # TODO: determine if this is the right starting timezone or if, in fact, they start in Europe/London
        data = data.set_index(data.time.dt.tz_localize("UTC").dt.tz_convert("Europe/London"))

        # Rename for convenience - this column name contains redundant information we don't need
        data = data.rename(columns={data.columns[1]: "acceleration"})

        # Remove data when imputed. We don't like that very much
        data[data.imputed == 1] = float("NaN")

        # Run
        results, by_day = activity_features(data)

    import json
    if output is not None:
        json.dump(results, open(output, 'w'))

    return data, results, by_day

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute activity features from processed actigraphy timeseries")
    parser.add_argument("input", help="input timeSeries.csv.gz file as output by biobankAccelerometerAnalysis/accProcess.py")
    parser.add_argument("output", help="output file to sleep features write to")

    args = parser.parse_args()

    run(args.input, args.output)
