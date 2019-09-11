# Contains information about (some) data codings for UKBiobank Fields
# selected for codings that we need

codings = dict(

    502 = dict(
        #0  No
        #1  Yes
        fields = [ 21066, 21068, 20532, 20435, 20541, 20431, 20406, 20401, 20456, 20503, 20468, 20474, 21024, 
                   20421, 20502, 20501, 20463, 20404, 20466, 20471, 20499, 20500, 20477, 20425, 21065, 20449,
                   20450, 21067, 20540, 20428, 20448, 21064, 20542, 20437, 20538],
        to_nan = [-818, -121],
    ),

    504 = dict(
        #-818    Prefer not to answer
        #1   Not at all
        #2   Several days
        #3   More than half the days
        #4   Nearly every day
         fields = [ 20518, 20505, 20510, 20512, 20507, 20519, 20506, 20509, 20514, 20511, 20516, 20513, 20508, 20515, 20520, 20517, ],
     to_nan = [-818],
    ),

    503 = dict(
        #-818 Prefer not to answer
        #0 No
        #1 Yes
        fields = [ 21027, 20484, 20486, 21035, 21038, 20447, 21074, 20483, 21062, 21063, 20446, 20441,
                    20480, 21073, 21071, 20457, 20504, 20415, 20432, 20481, 21076, ],
        to_nan = [-818]
    ),

    520 = dict(
            #-121    Do not know
            #0   No
            #1   Yes

            fields = [ 20419, 20429, 20427, 20423, 20422, 20426, 20417, ],
            to_nan = [-121]
    ),

    505 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less than half of the day
            #2   About half of the day
            #3   Most of the day
            #4   All day long
            fields = [ 20436, ],
            to_nan = [-121, -818],
    ),
    506 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less often
            #2   Almost every day
            #3   Every day
            fields = [20439],
            to_nan = [-818, -121],
    ),
    506 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   Stayed about the same or was on a diet
            #1   Gained weight
            #2   Lost weight
            #3   Both gained and lost some weight during the episode
            fields = [20536],
            to_nan = [-818, -121]
    ),

    1406 = dict(
            #1   Talking therapies, such as psychotherapy, counselling, group therapy or CBT
            #3   Other therapeutic activities such as mindfulness, yoga or art classes
            #-818    Prefer not to answer
            fields = [20550, 20547],
            to_nan = [-818],
    ),

    513 = dict(
            # Age in years if answered
            fields = [20433, 20434],
            to_nan = [-818, -121],
    ),

    511 = dict(
            #-818    Prefer not to answer
            #-999    Too many to count / One episode ran into the next
            # or a number (of episodes)
            fields = [20442],
            to_nan = [-818],
            #TOOD: what to do with -999?
    ),

    510 = dict(
            #-818    Prefer not to answer
            #0   Not at all
            #1   A little
            #2   Somewhat
            #3   A lot
            fields = [20418, 20440],
            to_nan = [-818],
    ),

    514 = dict(
            #-818    Prefer not to answer
            #-313    Not applicable
            #-121    Do not know
            #0   No
            #1   Yes
            fields = [20445],
            to_nan = [-818, -313, -121],
    ),

    515 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   Less than 24 hours
            #2   At least a day, but less than a week
            #3   A week or more
            fields = [20492],
            to_nan = [-818, -121],
    ),

    516 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   No problems
            #1   Needed treatment or caused problems with work, relationships, finances, the law or other aspects of life.
            fields = [20493],
            to_nan = [-818, -121],
    ),

    517 = dict(
            #-999    All my life / as long as I can remember
            # or period in months
            fields = [20420],
            to_nan = [],
            # TODO: what to do with -999?
    ),

    518 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #1   One thing
            #2   More than one thing
            fields = [20543],
            to_nan = [-818, -121],
    ),

    519 = dict(
            #-818    Prefer not to answer
            #-121    Do not know
            #0   Never
            #1   Rarely
            #2   Sometimes
            #3   Often
            fields = [ 20537, 20539],
            to_nan = [-818, -121],
    ),

)
