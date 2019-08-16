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

    # Time between consecutive periods
    gap_lengths = onset_times[1:] - offset_times[:-1]

    # Join together all periods that are less than 1 Hour apart
    joined_onset_times = numpy.concatenate([onset_times[:1], onset_times[1:][gap_lengths > HOURS_TO_COUNTS]])
    joined_offset_times = numpy.concatenate([offset_times[:-1][gap_lengths > HOURS_TO_COUNTS], offset_times[-1:]])

    # Extract main (longest) period
    period_lengths = joined_offset_times - joined_onset_times
    best_period = numpy.argmax(period_lengths)

    onset, offset, length = joined_onset_times[best_period], joined_offset_times[best_period], period_lengths[best_period]
    return onset / HOURS_TO_COUNTS, offset / HOURS_TO_COUNTS

def main_sleep_onset(sleep):
    onset, offset = extract_main_sleep_period(sleep)
    print(onset)
    return onset
def main_sleep_offset(sleep):
    onset, offset = extract_main_sleep_period(sleep)
    return offset

# Load data
data = pandas.read_csv(args.input, parse_dates=[0])

# Process the timezones from UTC to London
# TODO: determine if this is the right starting timezone or if, in fact, they start in Europe/London
data = data.set_index(data.time.dt.tz_localize("UTC").dt.tz_convert("Europe/London"))

# Rename for convenience - this column name contains redundant information we don't need
data = data.rename(columns={data.columns[1]: "acceleration"})
stretches = data.rolling(SLEEP_PERIOD, center=True).mean()

### Sleep Features

# Sleep onset
# For, say, Monday, we look for sleep periods starting between noon Monday and noon Tuesday
# since we want to put a sleep-onset time of 1AM on Tuesday as being the onset "for Monday"
# this is done with the "base=0.5" parameter, offsets the 1day samples by 12 hours
sleep_peak_time = stretches.resample("1D", base=0.5).sleep.idxmax()
sleep_peak_quality = stretches.sleep.resample("1D", base=0.5).max()
sleep_peak_data_available = stretches.resample("1D", base=0.5).sleep.count() > (COUNTS_PER_DAY * GOOD_COUNT_THRESHOLD)

# hours past 0:00 AM (usually of the day prior to the peak sleep)
average_sleep_peak_time = hours_since_noon(sleep_peak_time).mean() + 12

# Define the MAIN SLEEP PERIOD to be the period of time such that
# there are no periods > 1 HR of time without sleep
# and that maximizes the total amount of sleep in that period over the entire day (ie. of ones starting in a noon-to-noon)
main_sleep_onset = stretches.resample("1D", base=0.5).apply(main_sleep_onset)
main_sleep_offset = stretches.resample("1D", base=0.5).apply(main_sleep_offset)
