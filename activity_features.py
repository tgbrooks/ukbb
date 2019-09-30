import pandas
import numpy
import pytz

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
    # by most sleep (not counting the imputed times as sleep since sometimes that will call
    # a giant imputed block at the start of the readings as the main sleep period)
    sleep_by_period = [sleep.iloc[onset:offset].sum() for onset,offset in zip(onset_times, offset_times)]
    best_period = numpy.argmax(sleep_by_period)

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

        # Sleep onset
        # For, say, Monday, we look for sleep periods starting between noon Monday and noon Tuesday
        # since we want to put a sleep-onset time of 1AM on Tuesday as being the onset "for Monday"
        # this is done with the "base=0.5" parameter, offsets the 1day samples by 12 hours
        stretches = data.rolling(SLEEP_PERIOD, center=True, win_type="gaussian").mean(std=SLEEP_PERIOD)
        sleep_peak_time = stretches.resample("1D", base=0.5).sleep.idxmax()
        sleep_peak_quality = stretches.sleep.resample("1D", base=0.5).max()

        # Define the MAIN SLEEP PERIOD to be the period of time such that
        # there are no periods > 1 HR of time without sleep
        # and that maximizes the total amount of sleep in that period over the entire day (ie. of ones starting in a noon-to-noon)
        by_day = data.resample("1D", base=0.5)
        main_sleep_period = by_day.apply(extract_main_sleep_period)
        results_by_day = pandas.DataFrame(main_sleep_period.tolist(), index=main_sleep_period.index)

        # Define sleep as either 'main sleep' or 'other sleep'
        data['main_sleep_period'] = False
        for day in results_by_day.index:
            if pandas.isna(results_by_day.onset[day]):
                continue
            data.loc[results_by_day.onset[day]:results_by_day.offset[day], 'main_sleep_period'] = True
        data['main_sleep'] = (data.sleep > 0) & data.main_sleep_period
        data['other_sleep'] = (data.sleep > 0) & (~data.main_sleep_period)

        # Collect more day-level results
        results_by_day['sleep_peak_time'] = hours_since_noon(sleep_peak_time) + 12
        results_by_day['sleep_peak_quality'] = sleep_peak_quality

        # convert the index to be 'day-at-0:00' not 'day-at-noon'
        # so that we can use the base=0.0 resampling too
        results_by_day.set_index(results_by_day.index.normalize(), inplace=True)

        # For computing 'other' times, we want the data to be for a midnight-to-midnight day
        # not the noon-to-noon day that we use for sleep periods
        by_midnight_day = data.resample("1D", base=0.0)

        results_by_day.onset = hours_since_noon(results_by_day.onset) + 12
        results_by_day.offset = hours_since_noon(results_by_day.offset) + 12
        results_by_day['other_sleep'] = by_midnight_day.other_sleep.sum() / HOURS_TO_COUNTS
        results_by_day['main_sleep_duration'] = results_by_day.offset - results_by_day.onset
        # Total sleep is napping between that days midnight-midnight period
        # plus actual sleep in the main sleep from the noon-noon period
        results_by_day['total_sleep'] = results_by_day.main_sleep_duration - results_by_day.WASO + results_by_day.other_sleep
        results_by_day['main_sleep_ratio'] = (results_by_day.main_sleep_duration - results_by_day.WASO) / results_by_day.main_sleep_duration

        # Light and Temperature values
        # Note that these were not calibrated so they might not be very useful
        results_by_day['total_light'] = by_midnight_day.light.mean()
        results_by_day['light_90th'] = by_midnight_day.light.quantile(0.9)
        results_by_day['light_10th'] = by_midnight_day.light.quantile(0.1)
        results_by_day['temp'] = by_midnight_day.temp.mean()
        results_by_day['temp_90th'] = by_midnight_day.temp.quantile(0.9)
        results_by_day['temp_10th'] = by_midnight_day.temp.quantile(0.1)

        while_sleep = data[data.main_sleep].resample("1D", base=0.5)
        light_while_main_sleep = while_sleep.light.mean()
        light_while_main_sleep.set_axis(light_while_main_sleep.index.normalize(), axis=0, inplace=True)
        results_by_day['light_while_main_sleep'] = light_while_main_sleep

        temp_while_main_sleep = while_sleep.temp.mean()
        temp_while_main_sleep.set_axis(temp_while_main_sleep.index.normalize(), axis=0, inplace=True)
        results_by_day['temp_while_main_sleep'] = temp_while_main_sleep

        # Throw out days without nearly all of the hours
        # eg: if the data starts at Monday 10:00am, we don't want to consider Sunday noon - Monday noon a day
        # should have 2880 for a complete day
        days_invalid = (by_day.sleep.count() < 2500)
        results_by_day = results_by_day[~days_invalid.set_axis(days_invalid.index.normalize(), axis=0, inplace=False)]

        # Give better column names
        results_by_day.rename(columns={"onset": "main_sleep_onset", "offset": "main_sleep_offset"}, inplace=True)

        # Now fix the index by converting to dates instead of datetimes
        results_by_day.set_axis(results_by_day.index.date, axis=0, inplace=True)

        # Collect summary level results, across all days
        results.update( {col + "_avg": results_by_day[col].mean() for col in results_by_day})
        results.update( {col + "_std": results_by_day[col].std() for col in results_by_day})
    else:
        results_by_day = pandas.DataFrame([])

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

    return results, results_by_day

def run(input, output=None, by_day_output=None):
    '''Compute and output activity features'''
    # Load data
    data = pandas.read_csv(input, parse_dates=[0])
    if 'time' not in data:
        print(f"Processed timeseries file {input} has no 'time' index collumn and is being skipped")
        results = {}
        by_day = pandas.DataFrame([])
    else:
        # Process the timezones from UTC to London
        # Taking into account how they are reported

        # Times reported to use from processing of the CWA file are in a confusing state
        # where they are neither Europe/London nor UTC
        # Essentially, they are either in UTC if not in daylight savings time
        # and are in British Summer Time if in daylight savings time
        # and maintain whatever timezone they are in even if the DST crossover happens!!
        # Moreover - and impossible to fix with what we know - some of those that start
        # shortly after (somewhere around the next 3-4 days) the DST crossover
        # remain in the old time zone.
        if data.time[0].tz_localize("Europe/London").dst():
            time = data.time.dt.tz_localize(pytz.FixedOffset(60)) # British Summer Time
        else:
            time = data.time.dt.tz_localize("UTC") # non-DST London time
        time = time.dt.tz_convert("Europe/London")
        data = data.set_index(time)

        # Rename for convenience - this column name contains redundant information we don't need
        data = data.rename(columns={data.columns[1]: "acceleration"})

        # Remove data when imputed. We don't like that very much
        imputed = data.imputed.copy()
        data[imputed == 1] = float("NaN")
        data.imputed = imputed

        # Run
        results, by_day = activity_features(data)

    import json
    if output is not None:
        json.dump(results, open(output, 'w'))

    if by_day_output is not None:
        by_day.to_csv(by_day_output, sep="\t")

    return data, results, by_day

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute activity features from processed actigraphy timeseries")
    parser.add_argument("input", help="input timeSeries.csv.gz file as output by biobankAccelerometerAnalysis/accProcess.py")
    parser.add_argument("output", help="output file to write activity features summary to")
    parser.add_argument("--by_day", help="output file to write day-level statistics to", default=None)

    args = parser.parse_args()

    run(args.input, args.output, args.by_day)
