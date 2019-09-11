# Contains information about (some) data codings for UKBiobank Fields
# selected for codings that we need

codings = dict(
    502 = dict(
        fields = [
            21066,
            21068,
            20532,
            20435,
            20541,
            20431,
            20406,
            20401,
            20456,
            20503,
            20468,
            20474,
            21024,
            20421,
            20502,
            20501,
            20463,
            20404,
            20466,
            20471,
            20499,
            20500,
            20477,
            20425,
            21065,
            20449,
            20450,
            21067,
            20540,
            20428,
            20448,
            21064,
            20542,
            20437,
            20538,
            ],
        to_nan = [-818, -121],
        # Yes = 1, No = 0
    ),

    504 = dict(
        #-818    Prefer not to answer
        #1   Not at all
        #2   Several days
        #3   More than half the days
        #4   Nearly every day
         fields = [
             20518,
             20505,
             20510,
             20512,
             20507,
             20519,
             20506,
             20509,
             20514,
             20511,
             20516,
             20513,
             20508,
             20515,
             20520,
             20517,
        ],
     to_nan = [-818],
    ),

    503 = dict(
        #-818 Prefer not to answer
        #0 No
        #1 Yes
        fields = [
            21027,
            20484,
            20486,
            21035,
            21038,
            20447,
            21074,
            20483,
            21062,
            21063,
            20446,
            20441,
            20480,
            21073,
            21071,
            20457,
            20504,
            20415,
            20432,
            20481,
            21076,
        ],
        to_nan = [-818]
    ),

    520 = dict(
            #-121    Do not know
            #0   No
            #1   Yes

            fields = [
                20419,
                20429,
                20427,
                20423,
                20422,
                20426,
                20417,
                ],
            to_nan = [-121]
    ),
)

