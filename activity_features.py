import argparse
parser = argparse.ArgumentParser(description="Compute activity features from processed actigraphy timeseries")
parser.add_argument("input", help="input timeSeries.csv.gz file as output by biobankAccelerometerAnalysis/accProcess.py")
parser.add_argument("output", help="output file to sleep features write to")

args = parser.parse_args()

import pandas
import numpy

SAMPLE_RATE = 30 # in Seconds
HOURS_TO_COUNTS = 60*60//SAMPLE_RATE
COUNTS_PER_DAY = 24*HOURS_TO_COUNTS
SLEEP_PERIOD = 8*HOURS_TO_COUNTS

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

def extract_main_sleep_period(sleep):
    sleeping = (sleep.values > SLEEP_THRESHOLD).astype('int')# Binarize

    # Find periods of sleep
    diff = numpy.diff(numpy.concatenate([[0], sleeping, [0]]))
    onset_times, = numpy.where(diff > 0)
    offset_times, = numpy.where(diff < 0)

    if len(onset_times) == 0:
        # Never slept at all
        return pandas.to_datetime("NaT"), pandas.to_datetime("NaT"), float("NaN"), float("NaN"), float("NaN")

    # Time between consecutive periods
    gap_lengths = onset_times[1:] - offset_times[:-1]

    # Join together all periods that are less than 1 Hour apart
    joined_onset_times = numpy.concatenate([onset_times[:1], onset_times[1:][gap_lengths > HOURS_TO_COUNTS]])
    joined_offset_times = numpy.concatenate([offset_times[:-1][gap_lengths > HOURS_TO_COUNTS], offset_times[-1:]])

    # Extract main (longest) period
    period_lengths = joined_offset_times - joined_onset_times
    best_period = numpy.argmax(period_lengths)

    onset = sleep.index[0] + pandas.to_timedelta( str(joined_onset_times[best_period] / HOURS_TO_COUNTS) + "H")
    offset = sleep.index[0] + pandas.to_timedelta( str(joined_offset_times[best_period] / HOURS_TO_COUNTS) + "H")

    num_wakings = numpy.sum(numpy.diff(numpy.concatenate([[0], sleep[onset:offset] > SLEEP_THRESHOLD, [0]])) < 0)
    WASO = numpy.sum(1 - sleep[onset:offset]) / HOURS_TO_COUNTS
    other_duration = (numpy.sum(sleep) - numpy.sum(sleep[onset:offset])) / HOURS_TO_COUNTS

    return onset, offset, num_wakings, WASO, other_duration

def get_main_sleep_onset(sleep):
    onset, offset, num_wakings, WASO, other_duration = extract_main_sleep_period(sleep)
    return onset
def get_main_sleep_offset(sleep):
    onset, offset, num_wakings, WASO, other_duration = extract_main_sleep_period(sleep)
    return offset
def get_main_sleep_wakings(sleep):
    onset, offset, num_wakings, WASO, other_duration = extract_main_sleep_period(sleep)
    return num_wakings
def get_main_sleep_WASO(sleep):
    onset, offset, num_wakings, WASO, other_duration = extract_main_sleep_period(sleep)
    return WASO
def get_other_sleep_duration(sleep):
    onset, offset, num_wakings, WASO, other_duration = extract_main_sleep_period(sleep)
    return other_duration

# Load data
data = pandas.read_csv(args.input, parse_dates=[0])

# Process the timezones from UTC to London
# TODO: determine if this is the right starting timezone or if, in fact, they start in Europe/London
data = data.set_index(data.time.dt.tz_localize("UTC").dt.tz_convert("Europe/London"))

# Rename for convenience - this column name contains redundant information we don't need
data = data.rename(columns={data.columns[1]: "acceleration"})
stretches = data.rolling(SLEEP_PERIOD, center=True).mean()

# Remove data when imputed. We don't like that very much
data[data.imputed == 1] = float("NaN")

# dictionary of result summary values

### Sleep Features

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
main_sleep_onset = data.sleep.resample("1D", base=0.5).apply(get_main_sleep_onset)
main_sleep_offset = data.sleep.resample("1D", base=0.5).apply(get_main_sleep_offset)
main_sleep_wakings = data.sleep.resample("1D", base=0.5).apply(get_main_sleep_wakings)
main_sleep_WASO = data.sleep.resample("1D", base=0.5).apply(get_main_sleep_WASO)
main_sleep_duration = (main_sleep_offset - main_sleep_onset).dt.total_seconds()/60/60
main_sleep_ratio = (main_sleep_duration - main_sleep_WASO)/ main_sleep_duration
other_sleep_duration = data.sleep.resample("1D", base=0.5).apply(get_other_sleep_duration)

results = dict(
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
    other_sleep_duration_avg = other_sleep_duration.mean(),
    other_sleep_duration_std = other_sleep_duration.std(),
)

import json
json.dump(results, open(args.input, 'w'))
