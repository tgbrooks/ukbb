import pandas
import numpy
import statsmodels

data = pandas.read_csv("../processed/activity_by_day.txt", sep="\t", index_col=0, parse_dates=[0])
ukbb = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)

#### FOR SPEED WHILE TESTING
# and as control
numpy.random.seed(0)
full_data = data
ids = list(full_data.ID.unique())
selected_ids = numpy.random.choice(ids, 5000)
data = full_data[[(ID in selected_ids) for ID in full_data.ID]].copy()
###

data['sex'] = data.ID.map(ukbb.sex)
data['time'] = data.index
start_year = data.ID.map(data.time.groupby(data.ID).min().dt.year)
birth_year = data.ID.map(ukbb.birth_year)
data['age'] = start_year - birth_year

shift_one = numpy.arange(len(data)) - 1
has_previous = ((data.iloc[shift_one].ID.values == data.ID.values) &
                (data.iloc[shift_one].time.values == data.time.values - pandas.to_timedelta("1D")))
def get_previous_value(data, column):
    return numpy.where(has_previous,
                       data.iloc[shift_one][column],
                       float("NaN"))
data['prev_main_sleep_duration'] = get_previous_value(data, "main_sleep_duration")
data['prev_main_sleep_onset'] = get_previous_value(data, "main_sleep_onset")
data['diff_main_sleep_duration'] = data.main_sleep_duration - data.prev_main_sleep_duration
data['diff_main_sleep_onset'] = data.main_sleep_duration - data.prev_main_sleep_duration


def lowess(data, column):
    fit = statsmodels.nonparametric.smoothers_lowess.lowess(data['main_sleep_onset'],  data.index)
    return pandas.Series(fit[:,1], index=pandas.to_datetime(fit[:,0])).drop_duplicates()

dates = data.groupby(level=0).time.min()
main_sleep_onset_lowess = lowess(data, "main_sleep_onset")
main_sleep_onset_std = data.groupby(level=0).main_sleep_onset.std()
main_sleep_onset_diff_zscore = (data.groupby(level=0).main_sleep_onset.mean() - main_sleep_onset_lowess) / main_sleep_onset_std
prev_main_sleep_onset_diff_zscore = main_sleep_onset_diff_zscore.shift(1,"D")

data['main_sleep_onset_diff_zscore'] = data.time.map(main_sleep_onset_diff_zscore)
data['prev_main_sleep_onset_diff_zscore'] = data.time.map(prev_main_sleep_onset_diff_zscore)

def day_difference_pvalues(value, category, data):
    # Take difference from one day to the next and
    # compute the significance of non-zero difference
    shifted = data[[value, "time", "ID"]]
    shifted['time'] = shifted.time.shift(1,"D")
    shifted.set_index(['time', 'ID'], inplace=True)

    data = data.set_index(["time", "ID"])
    data['time-ID'] = data.index
    previous_value = data['time-ID'].map(shifted[value])
    lksd
    return previous_value
