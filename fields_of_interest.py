### Data about which fields to take
general_fields = dict(
    actigraphy_file = 90004,
    birth_year = 34,
    birth_month = 52,
    sex = 31,
    assessment_center = 54,
)

general_mental_health_fields = dict(
    date_of_mental_health_questionnaire = 20400,
    ever_prolonged_depression = 20446,
    ever_prolonged_loss_of_interest = 20441,
    ever_worried_much_more = 20425,
    ever_felt_worried_more_than_month = 20421,
    sought_professional_help_mental_distress = 20499,
    ever_mental_distress_prevent_activities = 20500,
    mental_health_problems_diagnosed = 20544,
    general_happiness = 20458,
    happiness_with_health = 20459,
    belief_life_meaningful = 20460,
    longest_period_worried = 20420,
)

# Depends upon ever_worried_much_more and longest_period_worried > 6 months
anxiety_dependent_fields = dict(
    activities_to_treat_anxiety = 20550,
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
    substances_taken_anxiety = 20549,
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
    activities_to_treat_depression = 20547,
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
    substances_taken_for_depression = 20546,
    thoughts_of_death_worst_episode = 20437,
    weight_change_worst_episode = 20536,
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
    manifestations_of_mania = 20548, # many categories TODO: code this variable
    severity_of_problems_due_to_mania = 20493,
)

trauma_fields = dict(
    felt_loved_as_child = 20489,
    physically_abused_by_family_as_child = 20488,
    felt_hated_by_family_member = 20487,
    sexually_molested_as_child = 20490,
    someone_to_take_to_doctor_as_child = 20491,
    been_in_confiding_relationship_as_adult = 20522,
    physical_violence_by_partner_as_adult = 20523,
    belittlement_by_partner_as_adult = 20521,
    sexual_interference_by_partner_as_adult = 20524,
    able_to_pay_rent_as_adult = 20525,
    victim_of_sexual_assault = 20531,
    victim_of_violent_crime = 20529,
    been_in_series_accident = 20526,
    witnessed_sudden_violent_death = 20530,
    diagnosed_with_life_threatening_illness = 20528,
    been_in_combat_war_zone = 20527,
    disturbing_thoughts_past_month =20497,
    felt_upset_when_reminded_of_experience_past_month = 20498,
    avoided_activities_because_of_stressful_experience_past_month = 20495,
    felt_distance_from_others_past_month = 20496,
    felt_irritable_past_month = 20494,
)

self_harm_fields = dict(
    every_thought_life_not_worth_living = 20479,
    ever_contemplated_self_harm = 20485,
    contemplated_self_harm_last_year = 20486,
    ever_self_harmed = 20480,
    number_times_self_harmed = 20482,
    self_harmed_past_year = 20481,
    methods_of_self_harm_used = 20553,
    actions_taken_following_self_harm = 20554,
    ever_attempted_suicide = 20483,
    attempted_suicide_past_year = 20484,
)

addiction_fields = dict(
    behavior_misc_addictions = 20552,
    ever_addicted_behavior_misc = 20431,
    ever_addicted_alcohol = 20406,
    ever_addicted_substance_or_behavior = 20401,
    ever_addicted_drugs = 20456,
    ongoing_addiction_drug = 20457,
    ever_addicted_medication = 20503,
    ongoing_addiction_medication = 20504,
    substance_of_addiction = 20551,
    ever_physically_dependent_on_alcohol = 20404,
    ongoing_addiction_alcohol = 20415,
    ongoing_addiction_behavior_misc = 20432,
)

cannabis_fields = dict(
    age_last_took_cannabis = 20455,
    ever_took_cannabis = 20453,
    max_frequency_cannabis = 20454,
)

# Biological Samples
blood_fields = dict(
    testosterone = 30850, # pmol/L
    oestradiol = 30800, #pmol/L
)

# Physical measures
physical_measures = dict(
    diastolic_blood_pressure = 4079,
    diastolic_blood_pressure_manual = 94,
    pulse_rate = 102,
    systolic_blood_pressure = 4080,
    systolic_blood_pressure_anual = 93,
    BMI = 21001,
    Height = 12144,
    hip_circumference = 49,
    waist_circumference = 48,
    weight = 21002,
    IPAQ_activity_group = 22032,
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
    year_job_ended=22603,
    year_job_start=22602,
    #22604Work hours - lumped category
    work_hours_per_week=22605,
    job_involved_shift_work=22620,
    day_shifts_worked=22630,
    night_shifts_worked=22650,
    mixture_of_day_and_night_shifts=22640,
    number_night_shifts_monthly_during_mixed_shifts=22643,
    number_night_shifts_monthly_during_night_shifts=22653,
    period_spent_working_day_shifts=22631,
    period_spent_working_mixed_shifts=22641,
    period_spent_working_night_shifts=22651,
    rest_days_during_mixed_shift_periods=22645,
    rest_days_during_night_shift_periods=22655,
    length_of_night_shift_during_mixed_shifts=22642,
    length_of_night_shift_during_night_shifts=22652,
    consecutive_night_shifts_during_mixed_shifts=22644,
    consecutive_night_shifts_during_night_sihfts=22654,
)

# Covariates
covariates = dict(
        alcohol_frequency = 1558,
        education = 6138,
        ethnicity = 21000,
        overall_health = 2178,
        household_income = 738,
        smoking = 20116,
)

#Medications
medications = dict(
    medication_cholesterol_bp_diabetes = 6177,
    medication_cholesterol_bp_diabetes_or_exog_hormones = 6153,

)
# Collect all the different fields
field_groups = [general_fields, general_mental_health_fields, anxiety_dependent_fields, recent_anxiety, depression_dependent_fields, sleep_change_type_fields, recent_depression, mania_fields, mania_dependent_fields, blood_fields, female_specific_fields, employment_fields, covariates, physical_measures, medications, addiction_fields, cannabis_fields, trauma_fields, self_harm_fields]
all_fields = dict()
for group in field_groups:
    all_fields.update(group)

# Just general characteristics, covariates, etc. not specific to any project
general_groups = [covariates, medications, general_fields,  blood_fields, physical_measures, female_specific_fields]
all_general_fields = dict()
for group in general_groups:
    all_general_fields.update(group)

# Just mental health fields
mental_health_groups = [general_fields, general_mental_health_fields, anxiety_dependent_fields, recent_anxiety, depression_dependent_fields, sleep_change_type_fields, recent_depression, mania_fields, mania_dependent_fields, addiction_fields, cannabis_fields, trauma_fields, self_harm_fields]
mental_health_fields = dict()
for group in mental_health_groups:
    mental_health_fields.update(group)
