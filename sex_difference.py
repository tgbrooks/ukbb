import pandas
import statsmodels.formula.api as smf
import numpy

pandas.plotting.register_matplotlib_converters()

ukbb = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)
data = pandas.read_csv("../processed/activity_by_day.txt", sep="\t", index_col=0, parse_dates=[0])

# Down-sample randomly for testing
numpy.random.seed(0)
SAMPLE_SIZE = 2_000
selected_IDs = numpy.random.choice(list(data.ID.unique()), size=SAMPLE_SIZE)
selected = [(ID in selected_IDs) for ID in data.ID]
data = data[selected]

# Add relevant factors
data['sex'] = data.ID.map(ukbb.sex)
birth_year = data.ID.map(ukbb.birth_year)
data['age'] = data.index.year - birth_year

data['time'] = data.index
shift_one = numpy.arange(len(data)) - 1
has_previous = ((data.iloc[shift_one].ID.values == data.ID.values) &
                (data.iloc[shift_one].time.values == data.time.values - pandas.to_timedelta("1D")))
def get_previous_value(data, column):
    return numpy.where(has_previous,
                       data.iloc[shift_one][column],
                       float("NaN"))

data['quality'] = data.main_sleep_ratio
data['prev_quality'] = get_previous_value(data, "quality")
data['prev_main_sleep_onset'] = get_previous_value(data, "main_sleep_onset")

for_model = data.dropna(subset=["quality", "main_sleep_onset", "prev_quality", "prev_main_sleep_onset"])
for_model['num_days'] = for_model.ID.map(for_model.groupby("ID").main_sleep_onset.count())
for_model = for_model[for_model.num_days > 5]
model = smf.mixedlm("quality ~ main_sleep_onset + prev_main_sleep_onset",
                    data = for_model,
                    groups = for_model["ID"],
                    re_formula = "~ main_sleep_onset + prev_main_sleep_onset"
                    )
fit = model.fit()
