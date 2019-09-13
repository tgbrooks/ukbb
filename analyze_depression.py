import pandas

# Load the data
ukbb_data = pandas.read_csv("../processed/ukbb_data_table.txt", sep="\t", index_col=0)
acc_summary = pandas.read_csv("../processed/activity_summary_aggregate.txt", sep="\t", index_col=0)
activity_features = pandas.read_csv("../processed/activity_features_aggregate.txt", sep="\t", index_col = 0)

data = ukbb_data.join(acc_summary, how="inner").join(activity_features, how="inner")

# Determine which subset to use
data['quality_actigraphy'] = (data['quality-goodCalibration'] == 1) & ~(data['quality-daylightSavingsCrossover'] == 1) & (data['quality-goodWearTime'] == 1)

data['has_questionnaire'] = ~data.date_of_mental_health_questionnaire.isna()

data['use'] = data.quality_actigraphy & data.has_questionnaire

# Binarized depression value
data['lifetime_depression'] = (data.ever_prolonged_depression == 1) | (data.ever_prolonged_loss_of_interest == 1)

data['lifetime_anxiety'] =  (data.ever_worried_much_more == 1) | (data.longest_period_worried >= 6) | (data.longest_period_worried == -999)
