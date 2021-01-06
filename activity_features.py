#!/usr/bin/env python
import datetime
import pandas
import numpy
import pytz
import statsmodels.api as sm

SAMPLE_RATE = 30 # in Seconds
HOURS_TO_COUNTS = 60*60//SAMPLE_RATE
COUNTS_PER_DAY = 24*HOURS_TO_COUNTS
SLEEP_PERIOD = 8*HOURS_TO_COUNTS
ACTIVITY_WINDOW = 8*HOURS_TO_COUNTS
ACTIVITY_WINDOW_STD = 1*HOURS_TO_COUNTS
ALL_COLUMNS = ['sleep', 'walking', 'sedentary', 'moderate', 'acceleration', 'tasks_light', 'MET', 'temp', 'light', 'MVPA', 'VPA']
LONGEST_IMPUTED_STRETCH = 1.75 * HOURS_TO_COUNTS

JUMP_RATIO = 0.5 # Length of longest gap to jump over as a ratio of the size of the adjacent sleep periods
RIGHT_JUMP_RATIO = 1.5
MAX_GAP_LENGTH = 2.0 * HOURS_TO_COUNTS
SLEEP_THRESHOLD = 0.9 # May not be necessary - 30second intervals seem to generally have sleep=0 or 1
MIN_COUNTS_IN_DAY = 20 * HOURS_TO_COUNTS

def noon_same_day(time):
    '''return Noon of the same day as `time`.
    Handles DST and timezones'''
    noon = datetime.datetime(year=time.year, month=time.month, day=time.day, hour=12)
    return time.tz.localize(noon)

def hours_since_midnight_before_last_noon(timeseries):
    ''' Return time since most recent midnight that was
    at least one noon ago. (11:00 gives 35 hours)

    NOTE: if a clock change happens due to daylight savings change
    then this will still give the total number of hours since noon.'''
    timeseries = pandas.to_datetime(timeseries)

    # Find the timezone for the timeseries
    tz = None
    for i in timeseries:
        if not pandas.isna(i):
            tz = i.tz
            break
    if tz == None:
        return float("NaN")

    midnight_same_day = timeseries.dt.floor("D")
    difference = (timeseries - midnight_same_day).dt.total_seconds() / 60 / 60

    midnight_last_day = midnight_same_day - pandas.DateOffset(days=1)
    difference_last_day = (timeseries - midnight_last_day).dt.total_seconds() / 60 / 60

    noon_same_day = pandas.to_datetime(dict(year=timeseries.dt.year, month=timeseries.dt.month, day=timeseries.dt.day, hour=12)).dt.tz_localize(tz)
    difference[noon_same_day > timeseries] = difference_last_day[noon_same_day > timeseries]

    return difference

def hours_since_midnight(timeseries):
    ''' Return time since the last midnight, in hours

    NOTE: if a clock change happens due to daylight savings change
    then this will still give the total number of hours since midnight.'''
    timeseries = pandas.to_datetime(timeseries)
    midnight = timeseries.dt.floor("1D")
    difference = (timeseries - midnight).dt.total_seconds() / 60 / 60
    return difference

def extract_main_sleep_periods(data):
    sleep = data.sleep

    # Binarize sleep (it seems to be 0 or 1 already aynway)
    # this also converts NaNs (form non-wear stretches) to 'sleeping'
    # since the non-wear calling seems to trigger in periods of sleep
    # and messes with main sleep period extraction
    sleeping = (~(sleep.values < SLEEP_THRESHOLD)).astype('int')

    # Find periods of sleep
    diff = numpy.diff(numpy.concatenate([[0], sleeping, [0]]))
    onset_times, = numpy.where(diff > 0)
    offset_times, = numpy.where(diff < 0)

    if len(onset_times) == 0:
        # Never slept at all
        return pandas.DataFrame([], columns="onset onset_time offset offset_time num_wakings WASO acceleration_during_main_sleep count".split(' '))

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
    i = 0
    while i < len(gap_lengths):
        joined_period_length = period_lengths[i]

        furthest_jump = i
        jump_length = 0
        for j in range(i+1, len(period_lengths)):
            # Accumulate the jump length by this gap
            jump_length += gap_lengths[j-1]

            if jump_length > MAX_GAP_LENGTH or jump_length >= JUMP_RATIO * joined_period_length:
                # Searched past the max gap length, so need to stop
                break

            imputed_counts = data.imputed[onset_times[i]:offset_times[j]].sum()
            if imputed_counts > LONGEST_IMPUTED_STRETCH:
                # We only allow a certain amount of imputed region
                # in our main sleep period. Since non-wear time triggers
                # during sleep but we also do not want to grab, say, two
                # days of non-wear time
                print(f"Stopping due to long imputed region {imputed_counts} at {onset_times[i]/HOURS_TO_COUNTS}:{offset_times[j]/HOURS_TO_COUNTS}")
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

        i = furthest_jump + 1

    # Join together all periods whose gaps match the above criteria
    onset_times = numpy.concatenate([onset_times[:1], onset_times[1:][~gaps_to_join]])
    offset_times = numpy.concatenate([offset_times[:-1][~gaps_to_join], offset_times[-1:]])

    # Calculate total sleep time in each period
    sleep_by_period = numpy.array([sleep.iloc[onset:offset].sum() for onset,offset in zip(onset_times, offset_times)])

    # Break the data into noon-noon days to find the main sleep period of each day
    days = pandas.unique(data.index.normalize())
    noon_to_noon_ranges = [slice(noon_same_day(day), noon_same_day(day + pandas.DateOffset(days=1)))
                            for day in days]
    results = {}
    for noon_to_noon, day in zip(noon_to_noon_ranges, days):
        start = numpy.searchsorted(data.index, noon_to_noon.start)
        stop = numpy.searchsorted(data.index, noon_to_noon.stop)

        # Count the number of datapoints where we have non-nan values in the noon-to-noon day
        count = data.sleep[noon_to_noon].count()

        in_range = (onset_times >= start) & (onset_times < stop)
        if not any(in_range):
            results[day] = dict(onset=float("NaN"),
                                onset_time=pandas.to_datetime("NaT"),
                                offset=float("NaN"),
                                offset_time=pandas.to_datetime("NaT"),
                                num_wakings=float("NaN"),
                                WASO=float("NaN"),
                                acceleration_during_main_sleep=float("NaN"),
                                count=count)
            continue

        # Extract main (longest) period
        # by most sleep (not counting the imputed times as sleep since sometimes that will call
        # a giant imputed block at the start of the readings as the main sleep period)
        indexes_in_range = numpy.where(in_range)[0] #Indexes into the list of sleep times
        best_period = indexes_in_range[numpy.argmax(sleep_by_period[in_range])]

        onset_index = onset_times[best_period]
        offset_index = offset_times[best_period]

        onset_time = sleep.index[0] + pandas.to_timedelta( str(onset_index / HOURS_TO_COUNTS) + "H")
        offset_time = sleep.index[0] + pandas.to_timedelta( str(offset_index / HOURS_TO_COUNTS) + "H")

        # Number of times woken up
        # counted as number of transitions from sleep
        num_wakings = numpy.sum(numpy.diff(numpy.concatenate([[0], sleeping[onset_index:offset_index] > SLEEP_THRESHOLD, [0]])) < 0) - 1
        WASO = numpy.sum(1 - sleeping[onset_index:offset_index]) / HOURS_TO_COUNTS

        # Average acceleration duing main sleep while actually sleeping
        main_sleep = data[onset_index:offset_index]
        # just the sleeping part of the main sleep period, counting imputed areas
        # we'll use acceleration = 0 for imputed areas since they were low enough to trigger
        # the non-wear-time assessment
        main_sleep = main_sleep[main_sleep.sleep != 0]
        acceleration_during_main_sleep = main_sleep.acceleration.fillna(0).mean()

        # Compute the time in hours since midnight of onset/offset
        onset = (onset_time - day).total_seconds() / 60 / 60
        offset = (offset_time - day).total_seconds() / 60 / 60

        results[day] = dict(onset=onset,
                        offset=offset,
                        onset_time=onset_time,
                        offset_time=offset_time,
                        num_wakings=num_wakings,
                        WASO=WASO,
                        acceleration_during_main_sleep=acceleration_during_main_sleep,
                        count=count)

    return pandas.DataFrame(results).T

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

def cosinor(activity):
    '''
    Cosine-fit phase, amplitude and mesor for a given measure
    '''

    start_midnight = activity.index[0].normalize()
    time = (activity.index - start_midnight).total_seconds() / 60 / 60 / 24
    cos_term = numpy.cos(time*2*numpy.pi)
    sin_term = numpy.sin(time*2*numpy.pi)
    const = numpy.ones(len(activity))
    exog = numpy.array([const, cos_term, sin_term]).T

    #values = activity.values
    values = numpy.log10(activity.values+1)
    try:
        results = sm.OLS(values, exog, missing="drop").fit()
        reduced = sm.OLS(values, const, missing="drop").fit()
    except Exception as e:
        if 'args' in globals():
            print(f"Exception occured on cosinor during {globals()['args'].input} - skipping cosinor")
        print(e)
        return {"mesor":float("NaN"), "amplitude": float("NaN"), "phase":float("NaN"), "cosinor_pvalue":float("NaN"), "cosinor_rsquared": float("NaN")}

    mesor, alpha, beta = results.params
    amplitude = numpy.sqrt(alpha**2 + beta**2)
    phase = numpy.arctan2(beta, alpha) * 24 / (2 * numpy.pi)
    return {"mesor": mesor,
            "amplitude": amplitude,
            "phase":phase,
            "cosinor_pvalue":results.compare_f_test(reduced)[1],
            "cosinor_rsquared":results.rsquared}

def detailed_stats(activity, M10_time=None, L5_time=None):
    '''  Gives detailed statistics on variability by day and by hour and with M10/L5 periods

    If M10/L5 times (mid-points) are not provided, then:
    M10/L5 windows are the 10 (resp. 5) contiguous hours of a day during which
    the average activity value is highest (resp. lowest).

    From these, we define:
    overall_M10 = mean activity during M10 window, across all days
    hourly_SD_M10 = standard deviation across hourly binned activity in the M10 window of all days
    within_day_SD_M10 = mean across days of the standard deviation across hourly binned activity in the M10 window
    between_day_SD_M10 = standard deviation across days of the mean across hourly binned activity in the M10 window
    M10_time = midpoint the M10 window
    (And analogously for L5.)
    The same set of variables are also computed for the entire day interval, not just the M10/L5 intervals
    And finally,
    RA = (overall_M10 - overall_L5)/(overall_M10 + overall_L5),
    '''

    # Timezone
    TZ = activity.index[0].tz

    # Number of days of data
    num_days = int((activity.index[-1] - activity.index[0]) / pandas.to_timedelta("1D")) + 1
    # Midnight of first day
    begin = pandas.to_datetime(activity.index[0].date()).tz_localize(TZ)

    # Average over all days of the activity at a specific time-of-day
    time_since_midnight = activity.index - pandas.to_datetime(activity.index.date).tz_localize(TZ)
    average = activity.groupby(time_since_midnight).mean()
    # Pad so that the rolling average can 'wrap around' midnights
    average = pandas.concat([average, average, average], axis=0)

    # Times of the M10/L5 windows
    # We subtract 1 millisecond from end times since pandas uses CLOSED intervals
    # when selecting a range of time-series but we want open (on right) intervals
    if M10_time is None or L5_time is None:
        M10_end = average.rolling(10 * HOURS_TO_COUNTS).sum().idxmax() - pandas.to_timedelta("1ms")
        M10_start = M10_end - pandas.to_timedelta("10H")
        L5_end = average.rolling(5 * HOURS_TO_COUNTS).sum().idxmin() - pandas.to_timedelta("1ms")
        L5_start = L5_end - pandas.to_timedelta("5H")
    else:
        M10_start = M10_time * pandas.to_timedelta('1H') - pandas.to_timedelta('5H')
        M10_end = M10_time * pandas.to_timedelta('1H') + pandas.to_timedelta('5H')
        L5_start = L5_time * pandas.to_timedelta('1H') - pandas.to_timedelta('2.5H')
        L5_end = L5_time * pandas.to_timedelta('1H') + pandas.to_timedelta('2.5H')

    def range_statistics(start, end):
        ''' compute statistics for a range of times (relative to midnight) '''
        day_std = []
        day_mean = []
        hourly = []
        for day in range(-1,num_days+1):
            interval = activity[begin + start + day * pandas.to_timedelta("24H"):
                                begin + end + day * pandas.to_timedelta("24H")]
            if interval.count() < 0.5 * (start - end) / pandas.to_timedelta("1H") * HOURS_TO_COUNTS:
                # Throw out any intervals with less than half the data present
                continue
            by_hour = interval.resample("1H", origin="start").mean() 
            day_std.append(by_hour.std())
            day_mean.append(by_hour.mean())
            hourly.append(by_hour)
        hourly = pandas.concat(hourly)
        day_std = pandas.Series(day_std)
        day_mean = pandas.Series(day_mean)
        return numpy.mean(day_mean), hourly.std(), numpy.mean(day_std), numpy.std(day_mean),

    overall_M10, hourly_SD_M10, within_day_SD_M10, between_day_SD_M10 = range_statistics(M10_start, M10_end)
    overall_L5, hourly_SD_L5, within_day_SD_L5, between_day_SD_L5 = range_statistics(L5_start, L5_end)
    overall, hourly_SD, within_day_SD, between_day_SD = range_statistics(pandas.to_timedelta("0H"),
                                                                         pandas.to_timedelta("24H"))
    return {
        "overall": overall,
        "hourly_SD": hourly_SD,
        "within_day_SD": within_day_SD,
        "between_day_SD": between_day_SD,
        "overall_M10": overall_M10,
        "hourly_SD_M10": hourly_SD_M10,
        "within_day_SD_M10": within_day_SD_M10,
        "between_day_SD_M10": between_day_SD_M10,
        "M10_time": (M10_start + pandas.to_timedelta("5H")) / pandas.to_timedelta("1H"),
        "overall_L5": overall_L5,
        "hourly_SD_L5": hourly_SD_L5,
        "within_day_SD_L5": within_day_SD_L5,
        "between_day_SD_L5": between_day_SD_L5,
        "L5_time": (L5_start + pandas.to_timedelta("2.5H")) / pandas.to_timedelta("1H"),
        "RA": (overall_M10 - overall_L5)/(overall_M10 + overall_L5),
    }

def activity_features(data):
    ''' return dictionary of result summary values '''

    results = {}

    ### Sleep Features
    if 'sleep' in data:

        # Sleep onset
        # For, say, Monday, we look for sleep periods starting between noon Monday and noon Tuesday
        # since we want to put a sleep-onset time of 1AM on Tuesday as being the onset "for Monday"
        # this is done with the "base=0.5" parameter, offsets the 1day samples by 12 hours
        # min_periods=1 so that we essentially ignore NaNs
        stretches = data.rolling(SLEEP_PERIOD, center=True, win_type="gaussian", min_periods=1).mean(std=SLEEP_PERIOD)
        sleep_peak_time = stretches.resample("1D", offset="12H").sleep.idxmax()
        sleep_peak_time.index = sleep_peak_time.index.normalize()
        sleep_peak_quality = stretches.sleep.resample("1D", offset="12H").max()
        sleep_peak_quality.index = sleep_peak_quality.index.normalize()


        # Inactivity peak
        # estimates the inactivity (i.e. sleep?) levels just from the raw data without
        # the use of the ML classifier and so may be more robust
        activity = data[['acceleration']].fillna(0)
        activity_avg = activity.rolling(SLEEP_PERIOD, center=True, win_type="gaussian", min_periods=1*HOURS_TO_COUNTS).mean(std=SLEEP_PERIOD/2)
        inactivity_peak_time = activity_avg.resample("1D", offset="12H").acceleration.idxmin()
        inactivity_peak_time.set_axis(inactivity_peak_time.index.normalize(), inplace=True, axis=0)
        inactivity_peak_value = activity_avg.resample("1D", offset="12H").acceleration.min()
        inactivity_peak_value.set_axis(inactivity_peak_value.index.normalize(), inplace=True, axis=0)

        # Find peak times for each feature type based off when the value maximizes in each noon-noon day
        # and the peak value, which is a weighted sum of the values over the window of size ACTIVITY_WINDOW
        feature_peak_times = {}
        feature_peak_values = {}
        # fillna(0) to make the imputed stretches become 0's
        window = data.fillna(0).rolling(ACTIVITY_WINDOW, center=True, win_type="gaussian", min_periods=1).mean(std=ACTIVITY_WINDOW_STD)
        peak_values = window.resample("1D", offset="0H").max()
        for feature in ALL_COLUMNS:
            if feature == 'sleep':
                # Skip 'sleep' since it is done separately above
                # needs to be midnight-to-midnight
                continue
            # For some reason, cannot do idxmax except on resampled DataFrame
            try:
                peak_time = window.resample("1D", offset="0H")[feature].idxmax()
                peak_value = peak_values[feature]
            except KeyError:
                # Lacking MET probably
                continue

            # If we literally have no data about the feature (it's all zero)
            # then we'll give a peak_time of NaN
            # NOTE: this works since all our features are non-negative
            peak_time[peak_value == 0.0] = float("NaN")

            feature_peak_times[feature]  = peak_time
            feature_peak_values[feature] = peak_value

        # Find the MAIN SLEEP PERIOD
        # within each noon-to-noon day
        # allowing a sleep period to go beyond noon the next day
        results_by_day = extract_main_sleep_periods(data)

        if len(results_by_day) == 0:
            # Quit early but first:
            # Give better column names
            results_by_day.rename(columns={"onset": "main_sleep_onset", "offset": "main_sleep_offset"}, inplace=True)
            results_by_day.drop(columns=["onset_time", "offset_time", "count"], inplace=True)

            return dict(), results_by_day

        # Define sleep as either 'main sleep' or 'other sleep'
        data['main_sleep_period'] = False
        for day in results_by_day.index:
            if pandas.isna(results_by_day.onset_time[day]):
                continue
            data.loc[results_by_day.onset_time[day]:results_by_day.offset_time[day], 'main_sleep_period'] = True
        data['main_sleep'] = (data.sleep > 0) & data.main_sleep_period
        data['other_sleep'] = (data.sleep > 0) & (~data.main_sleep_period)

        # Collect more day-level results
        results_by_day['sleep_peak_time'] = hours_since_midnight_before_last_noon(sleep_peak_time)
        results_by_day['sleep_peak_quality'] = sleep_peak_quality

        # convert the index to be 'day-at-0:00' not 'day-at-noon'
        # so that we can use the base=0.0 resampling too
        #results_by_day.set_index(results_by_day.index.normalize(), inplace=True)

        # For computing 'other' times, we want the data to be for a midnight-to-midnight day
        # not the noon-to-noon day that we use for sleep periods
        by_midnight_day = data.resample("1D", offset="0H")

        results_by_day['other_sleep'] = by_midnight_day.other_sleep.sum() / HOURS_TO_COUNTS
        results_by_day['main_sleep_duration'] = results_by_day.offset - results_by_day.onset
        # Total sleep is napping between that days midnight-midnight period
        # plus actual sleep in the main sleep from the noon-noon period
        results_by_day['total_sleep'] = results_by_day.main_sleep_duration - results_by_day.WASO + results_by_day.other_sleep
        results_by_day['main_sleep_ratio'] = (results_by_day.main_sleep_duration - results_by_day.WASO) / results_by_day.main_sleep_duration

        # Light and Temperature values
        # Note that these were not calibrated so they might not be very useful
        results_by_day['light_mean'] = by_midnight_day.light.mean()
        results_by_day['light_90th'] = by_midnight_day.light.quantile(0.9)
        results_by_day['light_10th'] = by_midnight_day.light.quantile(0.1)
        results_by_day['temp_mean'] = by_midnight_day.temp.mean()
        results_by_day['temp_90th'] = by_midnight_day.temp.quantile(0.9)
        results_by_day['temp_10th'] = by_midnight_day.temp.quantile(0.1)

        while_sleep = data[data.main_sleep].resample("1D", offset="12H")
        light_while_main_sleep = while_sleep.light.mean()
        light_while_main_sleep.set_axis(light_while_main_sleep.index.normalize(), axis=0, inplace=True)
        results_by_day['light_while_main_sleep'] = light_while_main_sleep

        temp_while_main_sleep = while_sleep.temp.mean()
        temp_while_main_sleep.set_axis(temp_while_main_sleep.index.normalize(), axis=0, inplace=True)
        results_by_day['temp_while_main_sleep'] = temp_while_main_sleep

        # Throw out noon-noon days without nearly all of the hours since we can't estimate sleep well
        # eg: if the data starts at Monday 10:00am, we don't want to consider Sunday noon - Monday noon a day
        days_valid = results_by_day['count'] > MIN_COUNTS_IN_DAY
        results_by_day = results_by_day[days_valid]
        results_by_day.drop(columns=["count"], inplace=True)

        # Gather the feature peak times/values
        for feature in ALL_COLUMNS:
            if feature == 'sleep':
                continue

            try:
                results_by_day[feature + "_peak_time"] = hours_since_midnight(feature_peak_times[feature])
                results_by_day[feature + "_peak_value"] = feature_peak_values[feature]
            except KeyError:
                # Lacking MET, probably
                continue

        results_by_day['inactivity_peak_time'] = hours_since_midnight_before_last_noon(inactivity_peak_time)
        results_by_day['inactivity_peak_value'] = inactivity_peak_value #Note: lower is less active

        # Give better column names
        results_by_day.rename(columns={"onset": "main_sleep_onset", "offset": "main_sleep_offset"}, inplace=True)
        results_by_day.drop(columns=["onset_time", "offset_time"], inplace=True)

        # Now fix the index by converting to dates instead of datetimes
        results_by_day.set_axis(results_by_day.index.date, axis=0, inplace=True)

        # Collect summary level results, across all days
        results.update( {col + "_mean": results_by_day[col].mean() for col in results_by_day})
        results.update( {col + "_SD": results_by_day[col].std() for col in results_by_day})
    else:
        results_by_day = pandas.DataFrame([])

    # Run cosinor on the activity variables
    results.update(cosinor(data.acceleration))

    # Compute the MVPA and VPA columns from MET
    if 'MET' in data.columns:
        data['MVPA'] = (data['MET'] >= 3) & (data['MET'] < 6)
        data['VPA'] = data['MET'] >= 6

    # Add RA/IS/IV values for each measure
    acc_stats = detailed_stats(data.acceleration)
    M10_time, L5_time = acc_stats['M10_time'], acc_stats['L5_time']
    acc_stats = {"acceleration_" + key: value for key,value in acc_stats.items()}
    results.update(acc_stats)
    results.update({
        "acceleration_IS": IS(data.acceleration),
        "acceleration_IV": IV(data.acceleration),
    })
    for activity in ALL_COLUMNS:
        if activity == 'acceleration':
            continue
        if activity not in data.columns:
            #Some datasets do not have MET or other columns
            #these will end up as NaNs in the aggregated spreadsheet
            print(f"Skipping {activity} - no data column of that name")
            continue
        stats = detailed_stats(data[activity], M10_time=M10_time, L5_time=L5_time)
        # append acitivity name to key, and remove redundant L5/M10 times (same for every variable)
        stats = {activity+"_"+key: value for key,value in stats.items()
                        if key not in ["L5_time", "M10_time"]}
        results.update(stats)
        results.update({
            activity + "_IS": IS(data[activity]),
            activity + "_IV": IV(data[activity]),
        })

    return results, results_by_day

def check_bounds(by_day):
    # Perform basic results quality assessments
    if any(by_day.main_sleep_offset < by_day.main_sleep_onset):
        print(f"Error in {input} observed main_sleep_offset < main_sleep_onset")
    if any(by_day.main_sleep_ratio > 1):
        print(f"Error in {input} observed main_sleep_ratio > 1")
    if any(by_day.main_sleep_ratio < 0):
        print(f"Error in {input} observed main_sleep_ratio < 0")
    if any(by_day.main_sleep_duration < 0):
        print(f"Error in {input} observed main_sleep_duration < 0")
    if any(by_day.total_sleep < 0):
        print(f"Error in {input} observed total_sleep < 0")
    if any(by_day.sleep_peak_quality > 1):
        print(f"Error in {input} observed sleep_peak_quality > 1")
    if any(by_day.sleep_peak_quality < 0):
        print(f"Error in {input} observed sleep_peak_quality < 0")
    if any(by_day.acceleration_during_main_sleep < 0):
        print(f"Error in {input} observed acceleration_during_main_sleep < 0")
    if any(by_day.main_sleep_onset > 37):
        print(f"Error in {input} observed main_sleep_onset > 37")
    if any(by_day.main_sleep_onset < 11):
        print(f"Error in {input} observed main_sleep_onset < 11")
    if any(by_day.sleep_peak_time > 37):
        print(f"Error in {input} observed sleep_peak_time > 37")
    if any(by_day.sleep_peak_time < 11):
        print(f"Error in {input} observed sleep_peak_time < 11")
    if any(by_day.walking_peak_time > 25):
        print(f"Error in {input} observed walking_peak_time > 25")
    if any(by_day.walking_peak_time < 0):
        print(f"Error in {input} observed walking_peak_time < 0")
    if any(by_day.sedentary_peak_time > 25):
        print(f"Error in {input} observed sedentary_peak_time > 25")
    if any(by_day.sedentary_peak_time < 0):
        print(f"Error in {input} observed sedentary_peak_time < 0")


# The data for the UKBB switches between two timezones, UTC+1 (British Summer Time) and UTC
# and it does so nearly but not quite on the daylight savings time transitions
# We record here manually observed transitions of these changes for the purposes of correcting
# the time zones
# Each entry is a (first_day, last_day, offset) tuples, where offset is in hours
# Data from other datasets would need to be processed differently
# NOTE: use average_over_timespan.py with --activation_date set to values around the time changes (British Summer Time)
# to visually inspect whether there is a timezone difference between different start days
# and see analyze_start_times.sh
TIMEZONES = [("2013-06-01", "2013-10-30", 1), # Original cohorts
             ("2013-10-31", "2014-04-03", 0),
             ("2014-04-04", "2014-10-29", 1),
             ("2014-10-30", "2015-04-01", 0),
             ("2015-04-02", "2015-10-28", 1),
             ("2015-10-29", "2015-12-29", 0),

             ("2017-11-01", "2018-03-28", 0), # Seasonal repeats
             ("2018-03-29", "2018-10-30", 1),
             ("2018-10-31", "2019-01-31", 0),
             ]

def load_activity_file(input):
    try:
        data = pandas.read_csv(input, parse_dates=[0])
    except (OSError, pandas.errors.EmptyDataError) as e:
         # Some rare files are bad, we just skip them and return nothing
        print(f"FAILED to read {input}. Invalid data?")
        print(e)
        return None

    if 'time' not in data:
        print(f"Processed timeseries file {input} has no 'time' index column and is being skipped")
        return None

    # Process the timezones from UTC to London
    # Taking into account how they are reported

    # Times reported to use from processing of the CWA file are in different timezones
    # based on when they start
    # We convert those all to Europe/London
    start_day = data.time[0].date()
    tz = None
    for first_day, last_day, offset in TIMEZONES:
        if (pandas.to_datetime(first_day).date() <= start_day and
                    pandas.to_datetime(last_day).date() >= start_day):
            tz = pytz.FixedOffset(60*offset)
    if tz == None:
        print(f"ERROR: could not find an appropriate timezone for file {input} which starts on date {repr(start_day)}\nTimezones must be manually annotated in activity_features.py based off the starting time of the recording")
    time = data.time.dt.tz_localize(tz)
    data = data.set_index(time.dt.tz_convert("Europe/London"))

    # Rename for convenience - this column name contains redundant information we don't need
    data.rename(columns={data.columns[1]: "acceleration"}, inplace=True)
    data.drop(columns=["time"], inplace=True)
    # Replace hyphen with underscore for consistency
    data.rename(columns={"tasks-light": "tasks_light"}, inplace=True)

    # Remove data when imputed. We don't like that very much
    imputed = data.imputed.copy()
    data[imputed == 1] = float("NaN")
    data.imputed = imputed
    return data


def run(input, output=None, by_day_output=None):
    '''Compute and output activity features'''
    # Load data
    print(f"Running with input={input}, output={output}, by_day_output={by_day_output}")

    data = load_activity_file(input)

    if data is None:
        results = {}
        by_day = pandas.DataFrame([])
    else:
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

    data, results, results_by_day = run(args.input, args.output, args.by_day)
