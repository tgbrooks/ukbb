import datafield_codings

### Data about which fields to take
general_fields = dict(
    actigraphy_file = 90004,
    birth_year = 34,
    sex = 31,
    assessment_center = 54,
)

mental_health_fields = dict(
    date_of_mental_health_questionnaire = 20400,
    ever_prolonged_depression = 20446,
    ever_prolonged_loss_of_interest = 20441,
    ever_worried_much_more = 20425,
    ever_felt_worried_more_than_month = 20421,
    sought_professional_help_mental_distress = 20499,
    ever_mental_distress_prevent_activities = 20500,
    #mental_health_problems_diagnosed = 20544, # List variable, currently we don't handle those
    general_happiness = 20458,
    happiness_with_health = 20459,
    belief_life_meaningful = 20460,
)

# Depends upon ever_worried_much_more and longest_period_worried > 6 months
anxiety_dependent_fields = dict(
    longest_period_worried = 20420, #NOTE: dependent upon 20420 but not 20425
    #activities_to_treat_anxiety = 20550, # List variable, currently we don't handle those
    difficulties_concentrating_worst_anxiety = 20419,
    difficulty_stopping_worrying_worst_anxiety = 20541,
    easily_tired_worst_anxiety = 20429,
    freq_of_difficulty_controlling_worry_worst_anxiety = 20537,
    freq_of_inability_to_stop_worrying_worst_anxiety = 20539,
    trouble_sleeping_worst_anxiety = 20427,
    impact_normal_roles_worst_anxiety = 20418,
    on_edge_worst_anxiety = 20423,
    irritable_worst_anxiety = 20422,
    multiple_worries_worst_anxiety = 20540,
    number_of_worries_worst_anxiety = 20543,
    professional_informed_anxiety = 20428,
    restless_worst_anxiety = 20426,
    stronger_worrying_anxiety = 20542,
    #substances_taken_anxiety = 20549, # List variable, currently we don't handle those
    tense_sore_aching_worst_anxiety = 20417,
    worried_most_days_worst_anxiety = 20538,
)

recent_anxiety = dict(
    recent_irritability = 20505,
    recent_foreboding = 20512,
    recent_anxiety = 20506,
    recent_uncontrollable_worrying = 20509,
    recent_restlessness = 20516,
    recent_trouble_relaxing = 20515,
    recent_worrying_too_much = 20520,
)


# Fields only asked if ever_prolonged_depression or ever_prolonged_loss_of_interest were True
depression_dependent_fields = dict(
    #activities_to_treat_depression = 20547, # List variable, currently we don't handle those
    age_at_first_episode = 20433,
    age_at_last_episode = 20434,
    depression_related_to_childbirth = 20445,
    depression_related_to_event = 20447,
    sleep_change_worst_episode = 20532,
    difficulty_concentrating_worst_episode = 20435,
    duration_worst_episode = 20438,
    tiredness_worst_episode = 20449,
    worthlessness_worst_episode = 20450,
    fraction_of_day_worst_episode = 20436,
    frequency_depressed_days_worst_episode = 20439,
    impact_on_normal_roles_worst_episode = 20440,
    number_depressed_periods = 20442,
    professional_informed_about_depression = 20448,
    #substances_taken_for_depression = 20546, # List variable, currently we don't handle those
    thoughts_of_death_worst_episode = 20437,
    #weight_change_worst_episode = 20536, # Categorical variable, currently we don't handle those
)

# These dependent fields were only asked if sleep_change_worst_episode
sleep_change_type_fields = dict(
    sleeping_too_much_worst_episode = 20534,
    trouble_falling_asleep_worst_episode = 20533,
    waking_too_early_worst_episode = 20535,
)

# Fields specifically about 'recent' behavior changes
# Recall that the questionnaire was given much after the actigraphy
recent_depression = dict(
    recent_changes_in_speed = 20518,
    recent_feelings_of_depression = 20510,
    recent_feeling_inadequacy = 20507,
    recent_feeling_tired_low_energy = 20519,
    recent_lack_of_interest = 20514,
    recent_poor_appetite_overeating = 20511,
    recent_thoughts_of_suicide = 20513,
    recent_trouble_concentrating = 20508,
    recent_sleep_troubles = 20517,
)

mania_fields = dict(
    ever_extreme_irritability = 20502,
    ever_mania = 20501,
)

# Dependent upon either of the mania fields
mania_dependent_fields = dict(
    longest_period_of_mania = 20492,
    #manifestations_of_mania = 20548, # many categories TODO: code this variable
    severity_of_problems_due_to_mania = 20493,
)

# Biological Samples
blood_fields = dict(
    testosterone = 30850, # pmol/L
    oestradiol = 30800, #pmol/L
)

# Female-specific
female_specific_fields = dict(
    age_bilateral_oophorectomy = 3882,
    age_hysterectomy = 2824,
    age_menopause = 3581,
    age_last_HRT = 3546,
    age_start_HRT = 3536,
    age_periods_started = 2714,
    bilateral_oophorectomy = 2834,
    ever_hysterectomy = 3591,
    ever_HRT = 2814,
    had_menopause = 2724,
)

# Employment history fields
employment_fields = dict(
    consecutive_night_shifts_during_mixed_shifts=22644,
    consecutive_night_shifts_during_night_sihfts=22654,
    day_shifts_worked=22630,
    job_involved_shift_work=22620,
    mixture_of_day_and_night_shifts=22640,
    night_shifts_worked=22650,
    number_night_shifts_monthly_during_mixed_shifts=22643,
    number_night_shifts_monthly_during_night_shifts=22653,
    period_spent_working_day_shifts=22631,
    period_spent_working_mixed_shifts,22641,
    period_spent_working_night_shifts=22651,
    rest_days_during_mixed_shift_periods=22645,
    rest_days_during_night_shift_periods=22655,
    length_of_night_shift_during_mixed_shifts=22642,
    length_of_night_shift_during_night_shifts=22652,
    #22604Work hours - lumped category
    work_hours_per_week=22605,
    year_job_ended=22603,
    year_job_start=22602,
)

# Collect all the different fields
field_groups = [general_fields, mental_health_fields, anxiety_dependent_fields, recent_anxiety, depression_dependent_fields, sleep_change_type_fields, recent_depression, mania_fields, mania_dependent_fields]
all_fields = dict()
for group in field_groups:
    all_fields.update(group)

def get_columns(data, field_dict):
    for field_name, field in field_dict.items():
        coding = datafield_codings.fields_to_codings[field]
        if coding.get("type", None) == "array":
            cols = [c for c in data.columns if c.startswith(f"f.{field}.")]
            yield from cols
        else:
            yield f"f.{field}.0.0"
