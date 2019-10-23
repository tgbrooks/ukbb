import pandas
import pylab
import numpy

pandas.plotting.register_matplotlib_converters()

MALE = 1
FEMALE = 0
MALE_COLOR = "b"
FEMALE_COLOR = "r"

# NOTE: set this to False for typical useage
# just randomizing to see what it looks like comparing to actual sex
RANDOMIZE_GENDER = False

# Warning: QC plots are slow
QC = False

ukbb = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)

data = pandas.read_csv("../processed/activity_by_day.txt", sep="\t", index_col=0, parse_dates=[0])
raw_data = data.copy()

data.drop(columns=["count"], inplace=True)

# Regress out age as an activity confounder
birth_year = data.ID.map(ukbb.birth_year)
age = data.index.year - birth_year
age_centered = age - age.mean()
no_nans = numpy.isfinite(data).all(axis=1)
value_data = data.drop(columns="ID")[no_nans]
mean = value_data.mean(axis=0)
age_coeff = numpy.linalg.lstsq(age_centered[no_nans].values.reshape((-1,1)), value_data-mean, rcond=None)
age_adjustment = age_centered.values.reshape((-1,1)) * age_coeff[0]
data.iloc[:,1:] -= age_adjustment # Adjust all columns but ID
data['age'] = age

# Find the number of participants with data for each day
by_day = data.groupby(level=0)
N = by_day.count().ID

sex = ukbb.sex
if RANDOMIZE_GENDER:
    # Perform a randomization so we can compare male/female differences
    # to a random difference
    numpy.random.seed(0)
    sex[:] = numpy.random.choice([0,1], size=sex.shape)

# Gather gender-specific data
male_by_day = data[data.ID.map(sex) == MALE].groupby(level=0)
male_means = male_by_day.mean()
male_std = male_by_day.std()
male_N = male_by_day.count().ID

female_by_day = data[data.ID.map(sex) == FEMALE].groupby(level=0)
female_means = female_by_day.mean()
female_std = female_by_day.std()
female_N = female_by_day.count().ID

def plot_week_difference(variables, diff=7, interval_size=1.96):
    def week_diff(means, std, N):
        week_ago =  means.shift(diff,"D")
        week_ago_std = std.shift(diff,"D")
        week_ago_N = N.shift(diff,"D")
        # SEM of the difference
        # uses that sigma^2 = sigma_A^2 + sigma_B^2 is the STD of A-B if A,B are normal
        SEM = std.divide( numpy.sqrt(N), axis=0)
        week_ago_SEM = week_ago_std.divide(numpy.sqrt(week_ago_N), axis=0)
        diff_SEM = numpy.sqrt(SEM**2 + week_ago_SEM**2)
        return means - week_ago, diff_SEM

    if diff == "mean":
        male_diff = male_means - male_means.mean(axis=0)
        male_SEM = male_std.divide(numpy.sqrt(male_N), axis=0)
        female_diff = female_means - female_means.mean(axis=0)
        female_SEM = female_std.divide(numpy.sqrt(female_N), axis=0)
    else:
        male_diff, male_SEM = week_diff(male_means, male_std, male_N)
        female_diff, female_SEM = week_diff(female_means, female_std, female_N)

    fig = pylab.figure()
    axes = fig.subplots(nrows=len(variables), sharex=True, squeeze=False)
    for variable, ax in zip(variables, axes.flatten()):
        ax.plot(male_diff.index, numpy.zeros(shape=male_diff.index.shape), color="k")

        ax.plot(male_diff.index, male_diff[variable], color=MALE_COLOR, label="M")
        ax.fill_between(male_SEM.index,
                        male_diff[variable]-interval_size*male_SEM[variable],
                        male_diff[variable]+interval_size*male_SEM[variable],
                        color=MALE_COLOR, alpha=0.5)

        ax.plot(female_diff.index, female_diff[variable], color=FEMALE_COLOR, label="F")
        ax.fill_between(female_SEM.index,
                        female_diff[variable]-interval_size*female_SEM[variable],
                        female_diff[variable]+interval_size*female_SEM[variable],
                        color=FEMALE_COLOR, alpha=0.5)

        ax.set_ylabel(variable)
        ax.legend()
    pylab.show()

plot_week_difference(["main_sleep_duration"])



##### Check for device-ID by week bias
# for QC
if QC:
    summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", sep="\t", index_col=0)

    # Create a plot of device ID by dates of useage to detect patterns
    # E.g. if there is a systemic batching of devices in and our ever week, say
    device_id = data.ID.map(summary['file-deviceID'])
    device_ids = list(device_id.unique())
    dates = list(data.index.unique())
    grid = numpy.zeros((len(device_ids), len(dates)))
    for i, (date, row) in enumerate(data.iterrows()):
        date_index = dates.index(date)
        device_id_index = device_ids.index(device_id[i])
        grid[device_id_index, date_index] += 1

    # order device IDs by similarity
    import scipy.cluster.hierarchy as hierarchy
    Z = hierarchy.linkage(grid)
    leaves_order = hierarchy.leaves_list(Z)
    grid = grid[leaves_order]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.imshow(grid)
    pylab.show()
