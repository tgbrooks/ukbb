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
    ever_thought_life_not_worth_living = 20479,
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

# Mood questions from an online questionnaire
mood_fields = dict(
    mood_down_in_dumps_last_week = 23046,
    mood_downhearted_depressed_last_week = 23072,
    mood_felt_calm_last_week = 23047,
    mood_happy_last_week = 23076,
    mood_nervous_last_week = 23045,
    mood_when_described = 23079,
)

# Mental health questions from the assessment center
mental_health_assessment = dict(
    bipolar_mdd = 20126, #Bipolar and major depression status
    bipolar = 20122, #Bipolar disorder status
    ever_depressed = 4598, #Ever depressed for a whole week
    ever_highly_irritable = 4653, #Ever highly irritable/argumentative for 2 days
    ever_manic = 4642, #Ever manic/hyper for 2 days
    ever_unenthusiastic =4631, #Ever unenthusiastic/disinterested for a whole week
    family_relationship_satisfaction = 4559, #Family relationship satisfaction
    fed_up_feelings = 1960, #Fed-up feelings
    financial_satisfcation = 4581, #Financial situation satisfaction
    freq_depressed_mood_last_2_weeks = 2050, #Frequency of depressed mood in last 2 weeks
    freq_tenseness_last_2_weeks = 2070, #Frequency of tenseness / restlessness in last 2 weeks
    freq_tiredness_last_2_weeks = 2080, #Frequency of tiredness / lethargy in last 2 weeks
    freq_unenthusiasm_last_2_weeks = 2060, #Frequency of unenthusiasm / disinterest in last 2 weeks
    friendship_satisfaction = 4570, #Friendships satisfaction
    guilty_feelings = 2030, #Guilty feelings
    happiness = 4526, #Happiness
    health_satisfication = 4548, #Health satisfaction
    illness_injury_last_2_weeks = 6145, #Illness, injury, bereavement, stress in last 2 years
    #10721Illness, injury, bereavement, stress in last 2 years (pilot)
    irritability = 1940, #Irritability
    lenth_longest_mania = 5663, #Length of longest manic/irritable episode
    loneliness_isolation = 2020, #Loneliness, isolation
    longest_depression = 4609, #Longest period of depression
    longest_unenthusiasm = 5375, #Longest period of unenthusiasm / disinterest
    manic_symptoms = 6156, #Manic/hyper symptoms
    miserableness = 1930, #Miserableness
    mood_swings = 1920, #Mood swings
    nervous_feelings = 1970, #Nervous feelings
    neuroticism_score = 20127, #Neuroticism score
    number_depressed_episodes = 4620, #Number of depression episodes
    number_unenthusiasm_episodes = 5386, #Number of unenthusiastic/disinterested episodes
    probable_recurrent_mdd_moderate = 20124, #Probable recurrent major depression (moderate)
    pprobable_recurrent_mdd_severe = 20125, #Probable recurrent major depression (severe)
    risk_taking = 2040, #Risk taking
    seen_psychiatrist = 2100, #Seen a psychiatrist for nerves, anxiety, tension or depression
    seen_doctor_for_mental_health = 2090, #Seen doctor (GP) for nerves, anxiety, tension or depression
    sensitivity_hurt_feelings = 1950, #Sensitivity / hurt feelings
    severity_of_mania = 5674, #Severity of manic/irritable episodes
    single_episode_mdd = 20123, #Single episode of probable major depression
    nerves = 2010, #Suffer from 'nerves'
    tense = 1990, #Tense / 'highly strung'
    job_satisfcation = 4537, #Work/job satisfaction
    worrier = 1980, #Worrier / anxious feelings
    worry_too_long = 2000, #Worry too long after embarrassment
)

# Biological Samples
blood_fields = dict(
    # Blood biochemistry fields
    alanine_aminotransferase = 30620, #Alanine aminotransferase
    albumin = 30600, #Albumin
    alkaline_phosphatase = 30610, #Alkaline phosphatase
    apolipoprotein_A = 30630, #Apolipoprotein A
    apolipoprotein_B = 30640, #Apolipoprotein B
    aspartate_aminotransferase = 30650, #Aspartate aminotransferase
    c_reactive_protein = 30710, #C-reactive protein
    calcium = 30680, #Calcium
    cholesterol = 30690, #Cholesterol
    creatinine = 30700, #Creatinine
    cystatin_C = 30720, #Cystatin C
    direct_bilirubin = 30660, #Direct bilirubin
    gamma_glutamyltransferase = 30730, #Gamma glutamyltransferase
    glucose = 30740, #Glucose
    glycated_heamoglobin = 30750, #Glycated haemoglobin (HbA1c)
    hdl_cholesterol = 30760, #HDL cholesterol
    igf_1 = 30770, #IGF-1
    ldl_direct = 30780, #LDL direct
    lipoprotein_A = 30790, #Lipoprotein A
    oestradiol = 30800, #Oestradiol pmol/L
    phosphate = 30810, #Phosphate
    rheumatoid_factor = 30820, #Rheumatoid factor
    shbg = 30830, #SHBG
    testosterone = 30850, #Testosterone pmol/L
    total_bilirubin = 30840, #Total bilirubin
    total_protein = 30860, #Total protein
    triglycerides = 30870, #Triglycerides
    urate = 30880, #Urate
    urea = 30670, #Urea
    vitamin_D = 30890, #Vitamin D

    # Blood count fields
    basophill_count = 30160, #Basophill count
    basophill_percent = 30220, #Basophill percentage
    eosinophill_count = 30150, #Eosinophill count
    eosinophill_percent = 30210, #Eosinophill percentage
    haematocrit_percent = 30030, #Haematocrit percentage
    haemoglobin_concentration = 30020, #Haemoglobin concentration
    high_light_scatter_reticulocyte_count = 30300, #High light scatter reticulocyte count
    high_light_scatter_reticulocyte_percent = 30290, #High light scatter reticulocyte percentage
    immature_reticulocyte_fraction = 30280, #Immature reticulocyte fraction
    lymphocyte_count = 30120, #Lymphocyte count
    lymphocyte_percent = 30180, #Lymphocyte percentage
    mean_corpuscular_haemoglobin = 30050, #Mean corpuscular haemoglobin
    mean_corpuscular_haemoglobin_conc = 30060, #Mean corpuscular haemoglobin concentration
    mean_corpuscular_volume = 30040, #Mean corpuscular volume
    mean_platelt_volume = 30100, #Mean platelet (thrombocyte) volume
    mean_reticulocyte_volume = 30260, #Mean reticulocyte volume
    mean_sphered_cell_volume = 30270, #Mean sphered cell volume
    monocyte_count = 30130, #Monocyte count
    monocyte_percent = 30190, #Monocyte percentage
    neutrophill_count = 30140, #Neutrophill count
    neutrophill_percent = 30200, #Neutrophill percentage
    nucleated_red_blood_cell_count = 30170, #Nucleated red blood cell count
    nucleated_red_blood_cell_percent = 30230, #Nucleated red blood cell percentage
    platelet_count = 30080, #Platelet count
    platelet_crit = 30090, #Platelet crit
    platelet_distribution_width = 30110, #Platelet distribution width
    red_blood_cell_count = 30010, #Red blood cell (erythrocyte) count
    red_blood_cell_distribution_width = 30070, #Red blood cell (erythrocyte) distribution width
    reticulocyte_count = 30250, #Reticulocyte count
    reticulocyte_percentage = 30240, #Reticulocyte percentage
    white_blood_cell_count = 30000, #White blood cell (leukocyte) count

    blood_smaple_time_collected = 3166,
    blood_sample_fasting_time = 74,
)

urine = dict(
    urine_creatinine = 30510, #Creatinine (enzymatic) in urine
    urine_creatinine_flag = 30515, #Creatinine (enzymatic) in urine result flag
    urine_microalbumin = 30500, #Microalbumin in urine
    urine_microalbumin_flag = 30505, #Microalbumin in urine result flag
    urine_potassium = 30520, #Potassium in urine
    urine_potassium_flag = 30525, #Potassium in urine result flag
    urine_sodium = 30530, #Sodium in urine
    urine_sodium_flag = 30535, #Sodium in urine result flag
)

arterial_stiffness = dict(
    arterial_stiffness_absence_of_notch = 4204,#Absence of notch position in the pulse waveform
    #arterial_stiffness_pulsewave_id = 4136,#Arterial pulse-wave stiffness device ID
    #arterial_stiffness_sitffness_id = 4206,#Arterial stiffness device ID
    arterial_stiffness_position_of_notch = 4199,#Position of pulse wave notch
    arterial_stiffness_position_of_peak = 4198,#Position of the pulse wave peak
    arterial_stiffness_position_of_shoulder = 4200,#Position of the shoulder on the pulse waveform
    arterial_stiffness_pulse_rate = 4194,#Pulse rate
    arterial_stiffness_arterial_stiffness_index = 21021,#Pulse wave Arterial Stiffness index
    arterial_stiffness_peak_to_peak_time = 4196,#Pulse wave peak to peak time
    arterial_stiffness_pressure_versus_time_curve = 4205,#Pulse wave pressure versus time response curve
    arterial_stiffness_wave_reflection_index = 4195,#Pulse wave reflection index
    arterial_stiffness_pulse_wave_velocity = 4207,#Pulse wave velocity (manual entry)
    #arterial_stiffness_skipping_reason = 20051,#Reason for skipping arterial stiffness
    #arterial_stiffness_method = 4186,#Stiffness method
)

hearing_test = dict(
    hearing_test_speech_recognition_threshold_left = 20019,#Speech-reception-threshold (SRT) estimate (left)
    hearting_test_speech_recognition_threshold_right = 20021,#Speech-reception-threshold (SRT) estimate (right)
)

impedance = dict(
    impedance_arm_fat_mass_left = 23124, #Arm fat mass (left)
    impedance_arm_fat_msas_right = 23120, #Arm fat mass (right)
    impedance_arm_fat_percent_left = 23123, #Arm fat percentage (left)
    impedance_arm_fat_percent_right = 23119, #Arm fat percentage (right)
    impedance_arm_fat_free_mass_left = 23125, #Arm fat-free mass (left)
    impedance_arm_fat_free_mass_right = 23121, #Arm fat-free mass (right)
    impedance_arm_mass_left = 23126, #Arm predicted mass (left)
    impedance_arm_mass_right = 23122, #Arm predicted mass (right)
    impedance_bassl_metabolic_rate = 23105, #Basal metabolic rate
    impedance_body_fat_percent = 23099, #Body fat percentage
    impedance_bmmi = 23104, #Body mass index (BMI)
    impedance_leg_fat mass_left = 23116, #Leg fat mass (left)
    impedance_leg_fat_mass_right = 23112, #Leg fat mass (right)
    impedance_leg_fat_percent_left = 23115, #Leg fat percentage (left)
    impedance_leg_fat_percent_right = 23111, #Leg fat percentage (right)
    impedance_leg_fat_free_mass_left = 23117, #Leg fat-free mass (left)
    impedance_leg_fat_free_mass_right = 23113, #Leg fat-free mass (right)
    impedance_leg_mass_left = 23118, #Leg predicted mass (left)
    impedance_leg_mass_right = 23114, #Leg predicted mass (right)
    impedance_trunk_fat_mass = 23128, #Trunk fat mass
    impedance_trunk_fat_percent = 23127, #Trunk fat percentage
    impedance_trunk_fat_free_masss = 23129, #Trunk fat-free mass
    impedance_trunk_mass = 23130, #Trunk predicted mass
    impedance_weight = 23098, #Weight
    impedance_whole_body_fat_mass = 23100, #Whole body fat mass
    impedance_whole_body_fat_free_mass = 23101, #Whole body fat-free mass
    impedance_whole_body_water_mass = 23102, #Whole body water mass
    #43Impedance device ID
    #23110Impedance of arm (left)
    #23109Impedance of arm (right)
    #6222Impedance of arm, manual entry (left)
    #6221Impedance of arm, manual entry (right)
    #23108Impedance of leg (left)
    #23107Impedance of leg (right)
    #6220Impedance of leg, manual entry (left)
    #6219Impedance of leg, manual entry (right)
    #23106Impedance of whole body
    #6218Impedance of whole body, manual entry
)

# Infectious disease markers
def name(line):
    name = line[:line.index(':')]
    return name.replace(" ", "_").replace("-","_").replace("/", "").replace(".","_")
def number(line):
    return line[line.index(':')+1:]

# Actually these are all used to derive the below summaries, so should generally not be used
infectious_diseases_antigens = { f"inf_dis_{name(line)}": number(line) for line in 
'''
23000:1gG antigen for Herpes Simplex virus-1
23001:2mgG unique antigen for Herpes Simplex virus-2
23049:Antigen assay QC indicator
23048:Antigen assay date
23026:BK VP1 antigen for Human Polyomavirus BKV
23039:CagA antigen for Helicobacter pylori
23043:Catalase antigen for Helicobacter pylori
23018:Core antigen for Hepatitis C Virus
23030:E6 antigen for Human Papillomavirus type-16
23031:E7 antigen for Human Papillomavirus type-16
23006:EA-D antigen for Epstein-Barr Virus
23004:EBNA-1 antigen for Epstein-Barr Virus
23042:GroEL antigen for Helicobacter pylori
23016:HBc antigen for Hepatitis B Virus
23017:HBe antigen for Hepatitis B Virus
23025:HIV-1 env antigen for Human Immunodeficiency Virus
23024:HIV-1 gag antigen for Human Immunodeficiency Virus
23023:HTLV-1 env antigen for Human T-Lymphotropic Virus 1
23022:HTLV-1 gag antigen for Human T-Lymphotropic Virus 1
23010:IE1A antigen for Human Herpesvirus-6
23011:IE1B antigen for Human Herpesvirus-6
23027:JC VP1 antigen for Human Polyomavirus JCV
23015:K8.1 antigen for Kaposi's Sarcoma-Associated Herpesvirus
23029:L1 antigen for Human Papillomavirus type-16
23032:L1 antigen for Human Papillomavirus type-18
23014:LANA antigen for Kaposi's Sarcoma-Associated Herpesvirus
23028:MC VP1 antigen for Merkel Cell Polyomavirus
23019:NS3 antigen for Hepatitis C Virus
23041:OMP antigen for Helicobacter pylori
23037:PorB antigen for Chlamydia trachomatis
23013:U14 antigen for Human Herpesvirus-7
23044:UreA antigen for Helicobacter pylori
23003:VCA p18 antigen for Epstein-Barr Virus
23040:VacA antigen for Helicobacter pylori
23005:ZEBRA antigen for Epstein-Barr Virus
23002:gE / gI antigen for Varicella Zoster Virus
23034:momp A antigen for Chlamydia trachomatis
23033:momp D antigen for Chlamydia trachomatis
23012:p101 k antigen for Human Herpesvirus-6
23020:p22 antigen for Toxoplasma gondii
23038:pGP3 antigen for Chlamydia trachomatis
23009:pp 28 antigen for Human Cytomegalovirus
23008:pp 52 antigen for Human Cytomegalovirus
23007:pp150 Nter antigen for Human Cytomegalovirus
23021:sag1 antigen for Toxoplasma gondii
23035:tarp-D F1 antigen for Chlamydia trachomatis
23036:tarp-D F2 antigen for Chlamydia trachomatis
'''.strip().splitlines()}

# Summary True/False values for each infectious disease
infectious_diseases = { f"inf_dis_{name(line)}": number(line) for line in 
'''
23065:BKV seropositivity for Human Polyomavirus BKV
23070:C. trachomatis Definition I seropositivity for Chlamydia trachomatis
23071:C. trachomatis Definition II seropositivity for Chlamydia trachomatis
23054:CMV seropositivity for Human Cytomegalovirus
23053:EBV seropositivity for Epstein-Barr Virus
23073:H. pylori Definition I seropositivity for Helicobacter pylori
23074:H. pylori Definition II seropositivity for Helicobacter pylori
23060:HBV seropositivity for Hepatitis B Virus
23061:HCV seropositivity for Hepatitis C Virus
23055:HHV-6 overall seropositivity for Human Herpesvirus-6
23056:HHV-6A seropositivity for Human Herpesvirus-6
23057:HHV-6B seropositivity for Human Herpesvirus-6
23058:HHV-7 seropositivity for Human Herpesvirus-7
23064:HIV-1 seropositivity for Human Immunodeficiency Virus
23068:HPV 16 Definition I seropositivity for Human Papillomavirus type-16
23075:HPV 16 Definition II seropositivity for Human Papillomavirus type-16
23069:HPV 18 seropositivity for Human Papillomavirus type-18
23050:HSV-1 seropositivity for Herpes Simplex virus-1
23051:HSV-2 seropositivity for Herpes Simplex virus-2
23063:HTLV-1 seropositivity for Human T-Lymphotropic Virus 1
23066:JCV seropositivity for Human Polyomavirus JCV
23059:KSHV seropositivity for Kaposi's Sarcoma-Associated Herpesvirus
23067:MCV seropositivity for Merkel Cell Polyomavirus
23062:T. gondii seropositivity for Toxoplasma gondii
23052:VZV seropositivity for Varicella Zoster Virus
'''.strip().splitlines()}

# Physical measures
physical_measures = dict(
    diastolic_blood_pressure = 4079,
    diastolic_blood_pressure_manual = 94,
    pulse_rate = 102,
    systolic_blood_pressure = 4080,
    systolic_blood_pressure_manual = 93,
    BMI = 21001,
    height = 12144,
    hip_circumference = 49,
    waist_circumference = 48,
    weight = 21002,
    IPAQ_activity_group = 22032,
    hand_grip_strength_left = 46,
    hand_grip_strength_right = 47,
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
    year_job_started=22602,
    year_job_ended=22603,
    job_code=22601,
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
    medication_for_pain_relief_constipation_heartburn = 6154,
)

# Full medications list - needs to be processed separately since long array
full_medications = dict(
        medication_code =  20003,
)


# Sleep questionnaire
sleep = dict(
    daytime_dozing = 1220,
    getting_up_in_morning = 1170,
    monring_evening_persion = 1180,
    nap_during_day = 1190,
    sleep_duration = 1160,
    sleeplessness = 1200,
    snoring = 1210,
)

# Self-reported conditions
# Needs to be handled specially since it is a long array
# and coding is hanlded as described here:
# http://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=6
self_reported_conditions = dict(
    condition_code = 20002,
)

# Collect all the different fields
field_groups = [general_fields, general_mental_health_fields, anxiety_dependent_fields, recent_anxiety, depression_dependent_fields, sleep_change_type_fields, recent_depression, mania_fields, mania_dependent_fields, blood_fields, female_specific_fields, employment_fields, covariates, physical_measures, medications, addiction_fields, cannabis_fields, trauma_fields, self_harm_fields, infectious_diseases, sleep, full_medications, medications, mental_health_assessment, mood_fields, arterial_stiffness, impedance, urine]
all_fields = dict()
for group in field_groups:
    all_fields.update(group)

# Just general characteristics, covariates, etc. not specific to any project
general_groups = [covariates, medications, general_fields,  blood_fields, physical_measures, female_specific_fields, arterial_stiffness, impedance, urine]
all_general_fields = dict()
for group in general_groups:
    all_general_fields.update(group)

# Just mental health fields
mental_health_groups = [general_fields, general_mental_health_fields, anxiety_dependent_fields, recent_anxiety, depression_dependent_fields, sleep_change_type_fields, recent_depression, mania_fields, mania_dependent_fields, addiction_fields, cannabis_fields, trauma_fields, self_harm_fields, mood_fields, mental_health_assessment]
mental_health_fields = dict()

for group in mental_health_groups:
    mental_health_fields.update(group)
