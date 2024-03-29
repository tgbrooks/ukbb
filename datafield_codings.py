# Contains information about (some) data codings for UKBiobank Fields
# selected for codings that we need

# a coding has the following:
# fields = list of field numbers that indicate which fields are coded by this coding
# to_nan = list of values that should be treated as missing (NaN) if present in this coding's field
# fillvalue = value to use when filling in a field of this coding if it was not asked
#             due to a dependent question being answered in the negative
#             eg: we fill "Did you have a change in appetite (during worst episode of depression)?" with NO
#                 if they said they never had an episode of depression
codings = dict(

    coding9 = dict(
        #0   Female
        #1   Male
        fields = [31],
        to_nan = [],
    ),

    coding10 = dict(
        # Values 110** are different assessment centers
        fields=[54],
        to_nan=[],
        # We don't handle categorical for now, and this is probably unecessary to handle it for this field
        #type="categorical",
    ),

    coding502 = dict(
        #0  No
        #1  Yes
        fields = [ 21066, 21068, 20532, 20435, 20541, 20431, 20406, 20401, 20456, 20503, 20468, 20474, 21024,
                   20421, 20502, 20501, 20463, 20404, 20466, 20471, 20499, 20500, 20477, 20425, 21065, 20449,
                   20450, 21067, 20540, 20428, 20448, 21064, 20542, 20437, 20538],
        to_nan = [-818, -121],
        fillvalue = 0,
    ),

    coding503 = dict(
        #-818 Prefer not to answer
        #0 No
        #1 Yes
        fields = [ 21027, 20484, 20486, 21035, 21038, 20447, 21074, 20483, 21062, 21063, 20446, 20441,
                    20480, 21073, 21071, 20457, 20504, 20415, 20432, 20481, 21076, ],
        to_nan = [-818],
        fillvalue = 0,
    ),

    coding504 = dict(
        #-818    Prefer not to answer
        #1   Not at all
        #2   Several days
        #3   More than half the days
        #4   Nearly every day
         fields = [ 20518, 20505, 20510, 20512, 20507, 20519, 20506, 20509, 20514, 20511, 20516, 20513, 20508, 20515, 20520, 20517, ],
         to_nan = [-818],
         type = "ordinal",
         fillvalue = 1,
    ),

    coding505 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less than half of the day
            #2   About half of the day
            #3   Most of the day
            #4   All day long
            fields = [ 20436, ],
            to_nan = [-121, -818],
            type = "ordinal",
            fillvalue = 0,
    ),

    coding506 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less often
            #2   Almost every day
            #3   Every day
            fields = [20439],
            to_nan = [-818, -121],
            type = "ordinal",
            fillvalue = 0,
    ),

    coding507 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   Stayed about the same or was on a diet
            #1   Gained weight
            #2   Lost weight
            #3   Both gained and lost some weight during the episode
            fields = [20536],
            to_nan = [-818, -121],
            type = "categorical",
            fillvalue = 0,
    ),

    coding508 = dict(
            #0   No
            #1   Yes
            fields = [ 20534, 20533, 20535],
            to_nan = [],
            fillvalue = 0,
    ),

    coding509 = dict(
            #-818    Prefer not to answer
            #1   Less than a month
            #2   Between one and three months
            #3   Over three months, but less than six months
            #4   Over six months, but less than 12 months
            #5   One to two years
            #6   Over two years
            fields = [20438],
            to_nan = [-818],
            type="ordinal",
            fillvalue = 0,
    ),

    coding510 = dict(
            #-818    Prefer not to answer
            #0   Not at all
            #1   A little
            #2   Somewhat
            #3   A lot
            fields = [20418, 20440],
            to_nan = [-818],
            fillvalue = 0,
    ),

    coding511 = dict(
            #-818    Prefer not to answer
            #-999    Too many to count / One episode ran into the next
            # or a number (of episodes)
            fields = [20442],
            to_nan = [-818],
            #TOOD: what to do with -999?
            fillvalue = 0,
    ),

    coding513 = dict(
            # Age in years if answered
            fields = [20433, 20434],
            to_nan = [-818, -121],
            fillvalue = float("NaN"),
    ),

    coding514 = dict(
            #-818    Prefer not to answer
            #-313    Not applicable
            #-121    Do not know
            #0   No
            #1   Yes
            fields = [20445],
            to_nan = [-818, -313, -121],
            fillvalue = 0,
    ),

    coding515 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less than 24 hours
            #2   At least a day, but less than a week
            #3   A week or more
            fields = [20492],
            to_nan = [-818, -121],
            type="ordinal",
            fillvalue = 0,
    ),

    coding516 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   No problems
            #1   Needed treatment or caused problems with work, relationships, finances, the law or other aspects of life.
            fields = [20493],
            to_nan = [-818, -121],
            fillvalue = 0,
    ),

    coding517 = dict(
            #-999    All my life / as long as I can remember
            # or period in months
            fields = [20420],
            to_nan = [],
            fillvalue = 0,
            # TODO: what to do with -999?
    ),

    coding518 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   One thing
            #2   More than one thing
            fields = [20543],
            to_nan = [-818, -121],
            fillvalue = 0, # TODO: is this a reasonable fill-value?
    ),

    coding519 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   Never
            #1   Rarely
            #2   Sometimes
            #3   Often
            fields = [ 20537, 20539],
            to_nan = [-818, -121],
            type="ordinal",
            fillvalue = 0,
    ),

    coding520 = dict(
            #-121    Do not know
            #0   No
            #1   Yes
            fields = [ 20419, 20429, 20427, 20423, 20422, 20426, 20417, ],
            to_nan = [-121],
            fillvalue = 0,
    ),

    coding537 = dict(
            #-818   Prefer not to answer
            #-121    Do not know
            #1   Extremely happy
            #2   Very happy
            #3   Moderately happy
            #4   Moderately unhappy
            #5   Very unhappy
            #6   Extremely unhappy
            fields = [ 20458, 20459],
            to_nan = [-818, -121],
            type = "ordinal",
    ),

    coding538 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Not at all
            #2   A little
            #3   A moderate amount
            #4   Very much
            #5   An extreme amount
            fields = [20460],
            to_nan = [-818, -121],
            type = "ordinal",
    ),

    coding1405 = dict(
            #1  Unprescribed medication (more than once)
            #3   Medication prescribed to you (for at least two weeks)
            #4   Drugs or alcohol (more than once)
            #-818    Prefer not to answer
            fields = [20549, 20546],
            to_nan = [-818],
            type="array",
            fillvalue = "", #TODO: how do we fill in as an empty list?
    ),

    coding1406 = dict(
            #1   Talking therapies, such as psychotherapy, counselling, group therapy or CBT
            #3   Other therapeutic activities such as mindfulness, yoga or art classes
            #-818    Prefer not to answer
            fields = [20550, 20547],
            to_nan = [-818],
            type = "array",
            fillvalue = "", #TODO: how do we fill in as an empty list?
    ),

    coding100349 = dict(
        #1	Yes
        #0	No
        #-1	Do not know
        #-3	Prefer not to answer
        fields=[1767, 3912, 3942, 3571, 1677, 3616, 2335, 2443, 4598, 2345, 2674, 2694, 2365, 2774, 4653, 4642, 2907, 2784, 4631, 2814, 4067, 1797, 1960, 3005, 2463, 2956, 10844, 2030, 3799, 10793, 2257, 3414, 1940, 3773, 5463, 4728, 5452, 5496, 5485, 5474, 2644, 2020, 2188, 1787, 3720, 1930, 1920, 1835, 3404, 1970, 4501, 1777, 10016, 2040, 2100, 2090, 1950, 4717, 2986, 3741, 2010, 5529, 1990, 2316, 1980, 2000],
        to_nan=[-1,-3],
    ),

    coding100579 = dict(
        #1	Yes
        #0	No
        #2	Not sure - had a hysterectomy
        #3	Not sure - other reason
        #-3	Prefer not to answer
        fields=[2724],
        to_nan=[2,3,-3],
    ),

    coding100698 = dict(
        #-313 Ongoing when data entered
        fields=[22603],
        type="array",
    ),

    coding489 = dict(
        #0	Shift pattern was worked for some (but not all) of job
        #1	Shift pattern was worked for whole of job
        #9	This type of shift pattern was not worked during job
        fields=[22650],
        type="array",
    ),

    codingNONE_array = dict(
        fields=[22605, 22620],
        type="array",
    ),
)

field_to_codings = {}
for coding_name, coding in codings.items():
    for field in coding['fields']:
        field_to_codings[field] = coding
