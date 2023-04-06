import pandas
import pathlib

# Time between actigraphy reading and the first allowed diagnoses
# All subjects whose first diagnosis occurs this much after actigraphy
# will be counted as cases. Anyone with this diagnosis prior to that cutoff
# will be excluded from analysis as neither control nor case
LAG_TIME = 365.25 * pandas.to_timedelta("1D")

def load_longitudinal_diagnoses(selected_ids, actigraphy_start_date, INPUTDIR, OUTDIR, RECOMPUTE=False):
    '''
    Loads the case/exclude/control status for subjects in with ID in 'selected_ids'
    and whose start of actigraphy measurement date is given in the Series actigraphy_start_date

    Returns a case_status DataFrame with the following columns:
    ID: ID of the subject
    PHECODE: phecode here
    status: 'case' if a valid longitudinal case (first evidence of phecode occurs after actigraphy)
            'exclude' if has a prior record of the diagnosis (or excluded phecodes) from before the actigraphy
    first_date: if a case, this gives the earliest known date of the diagnosis

    Contol subjects have no entries whatsoever in the case_status dataframe
    '''

    # Load the PheCode mappings
    # Downloaded from https://phewascatalog.org/phecodes_icd10
    # Has columns:
    # ICD10 | PHECODE | Exl. Phecodes | Excl. Phenotypes
    phecode_info = pandas.read_csv("../phecode_definitions1.2.csv", dtype=dict(phecode=str)).set_index('phecode')
    phecode_info.index = phecode_info.index.str.lstrip('0')
    phecode_info['exclude_start'] = [x.split('-')[0].lstrip('0') if x==x else x for x in phecode_info.phecode_exclude_range]
    phecode_info['exclude_end'] = [x.split('-')[1].lstrip('0') if x==x else x for x in phecode_info.phecode_exclude_range]
    phecode_map = pandas.read_csv("../Phecode_map_v1_2_icd10_beta.csv", dtype=dict(PHECODE=str))
    phecode_map.set_index(phecode_map.ICD10.str.replace(".","", regex=False), inplace=True) # Remove '.' to match UKBB-style ICD10 codes

    #  Determine the set of phecodes to exclude from 'controls' for a given diagnosis
    # So phecode_exclusions maps phecode -> excluded where excluded is a phecode that
    # should not be counted as a case
    phecode_exclusions = []
    def zfill(s, n=4):
        ''' like str.zfill but pads a fixed number past the decimal point '''
        if s == s:
            if '.' in s:
                return '0'*max(n-s.index('.'),0) + s
            else:
                return s.zfill(n)
        else:
            return s
    base_phecodes = phecode_info.index.map(zfill)
    for phecode in phecode_info.index:
        if not isinstance(phecode_info.loc[phecode, 'phecode_exclude_range'], str):
            continue
        exclude_start = zfill(phecode_info.loc[phecode, 'exclude_start'].split('.')[0])
        exclude_end = zfill(phecode_info.loc[phecode, 'exclude_end'])
        in_range = (base_phecodes >= exclude_start) & (base_phecodes <= exclude_end)
        phecode_exclusions.extend([
            {"phecode": phecode, "excluded": p}
                for p in phecode_info.index[in_range]
        ])
    phecode_exclusions = pandas.DataFrame(phecode_exclusions)

    # v1.2 Downloaded from https://phewascatalog.org/phecodes
    phecode_map_icd9 = pandas.read_csv("../phecode_icd9_map_unrolled.csv", dtype=dict(phecode=str))
    phecode_map_icd9.rename(columns={"icd9":"ICD9", "phecode":"PHECODE"}, inplace=True)
    phecode_map_icd9.set_index( phecode_map_icd9['ICD9'].str.replace(".","", regex=False), inplace=True) # Remove dots to match UKBB-style ICD9s
    phecode_map_icd9['PHECODE'] = phecode_map_icd9.PHECODE.str.lstrip('0')

    # Map the phecodes to their parent / grandparent PheCODEs
    # Eg. 585.31 has parent 585.3 and grandparent 585
    def n_digits_after_decimal_point(string, n):
        # give a specified number of digits past decimal point
        # assuming string is of the form "123.456"
        if '.' not in string:
            return string # No digits
        if n == 0: #Don't include '.' at all
            return string[:string.index('.')]
        return string[:string.index('.')+n+1]
    phecode_parents = pandas.concat([
        # Parents
        pandas.DataFrame({
            "phecode": phecode_info.index,
            "parent": phecode_info.index.map(lambda x: n_digits_after_decimal_point(x, 1)),
        }),
        # Grand parents
        pandas.DataFrame({
            "phecode": phecode_info.index,
            "parent": phecode_info.index.map(lambda x: n_digits_after_decimal_point(x, 0)),
        })
    ])
    phecode_parents = phecode_parents[phecode_parents.phecode != phecode_parents.parent].drop_duplicates()
    phecode_parents = phecode_parents[phecode_parents.parent.isin(phecode_info.index)]

    # Extend the phecode map to map any code to all the phecodes 'above' the code in the tree
    phecode_map_extended = pandas.concat([
        phecode_map,
        pandas.merge(
            phecode_map,
            phecode_parents,
            left_on = "PHECODE",
            right_on = "phecode",
        )[['ICD10', 'parent', 'Exl. Phecodes', 'Excl. Phenotypes']].rename(columns={"parent": "PHECODE"})
    ]).drop_duplicates()
    phecode_map_extended.set_index(phecode_map_extended.ICD10.str.replace(".","", regex=False), inplace=True) # Remove '.' to match UKBB-style ICD10 codes
    phecode_map_icd9_extended = pandas.concat([
        phecode_map_icd9,
        pandas.merge(
            phecode_map_icd9,
            phecode_parents,
            left_on = "PHECODE",
            right_on = "phecode",
        )[['ICD9', 'parent']].rename(columns={"parent": "PHECODE"})
    ]).drop_duplicates()
    phecode_map_icd9_extended.set_index( phecode_map_icd9_extended['ICD9'].str.replace(".","", regex=False), inplace=True) # Remove dots to match UKBB-style ICD9s

    ##### Load Patient Data
    icd10_entries_raw = pandas.read_csv(INPUTDIR/"ukbb_icd10_entries.txt", sep="\t", parse_dates=["first_date"])
    # Select our cohort from all the entries
    icd10_entries_raw = icd10_entries_raw[icd10_entries_raw.ID.isin(selected_ids)].copy()
    icd10_entries_raw.rename(columns={"ICD10_code": "ICD10"}, inplace=True)
    icd10_entries = icd10_entries_raw.join(phecode_map_extended.PHECODE, on="ICD10")

    ### and the ICD9 data
    icd9_entries = pandas.read_csv(INPUTDIR/"ukbb_icd9_entries.txt", sep="\t")
    icd9_entries.rename(columns={"ICD9_code": "ICD9"}, inplace=True)
    icd9_entries = icd9_entries.join(phecode_map_icd9_extended.PHECODE, on="ICD9")
    icd9_entries = icd9_entries[icd9_entries.ID.isin(selected_ids)]

    # Self-reported conditions from the interview stage of the UK Biobank
    self_reported = pandas.read_csv(INPUTDIR/"ukbb_self_reported_conditions.txt", sep="\t", dtype={"condition_code":int})
    data_fields = pandas.read_csv("../Data_Dictionary_Showcase.csv", index_col="FieldID")
    codings = pandas.read_csv("../Codings_Showcase.csv", dtype={"Coding": int})
    condition_code_to_meaning = codings[codings.Coding  == data_fields.loc[20002].Coding].drop_duplicates(subset=["Value"], keep=False).set_index("Value")
    self_reported["condition"] = self_reported.condition_code.astype(str).map(condition_code_to_meaning.Meaning)
    self_reported = self_reported[self_reported.ID.isin(selected_ids)]

    # Convert self-reported conditions to phecodes
    # Load manaully mapped self-reports to phecodes
    self_report_phecode_map = pandas.read_csv("../self_report_conditions_meanings.txt", sep="\t", dtype={"PheCODE": str})
    self_report_phecode_map_extended = pandas.concat([
        self_report_phecode_map[['Value', 'PheCODE', 'Meaning']],
        pandas.merge(
            self_report_phecode_map[['Value', 'PheCODE', 'Meaning']],
            phecode_parents,
            left_on="PheCODE",
            right_on="phecode"
        )[['Value', 'parent', 'Meaning']].rename(columns={"parent": "PheCODE"})
    ])
    self_reported = pandas.merge(
        self_reported,
        self_report_phecode_map_extended[['Value', 'PheCODE']],
        left_on="condition_code",
        right_on='Value',
        how="left",
    )

    ## Get case data for those occuring after actigraphy
    # Only use icd10 since other data predates actigraphy
    first_date = pandas.to_datetime(icd10_entries.first_date)
    # People can be excluded for having a diagnosis predating their actigraphy (exact cutoff time)
    # or from having a diagnosis within the lag period (lagged cutoff time)
    # We track these separately, even though they are both excluded from the main study,
    # so that we can report their numbers
    exact_cutoff_time = icd10_entries.ID.map(actigraphy_start_date)
    lagged_cutoff_time = icd10_entries.ID.map(actigraphy_start_date) + LAG_TIME
    icd10_entries_after_actigraphy = icd10_entries[first_date > lagged_cutoff_time].copy()
    icd10_entries_after_actigraphy['case_status'] = 'case'

    # All icd9 / self-reported diagnoses are prior as well as earlier ICD10s
    # as well as ICD10s that occur before the actigraphy
    prior_existing_diagnosis_long = pandas.concat([
        icd10_entries.loc[first_date <= exact_cutoff_time, ['ID', 'PHECODE']],
        icd9_entries[['ID', 'PHECODE']],
        self_reported[['ID', 'PheCODE']].rename(columns={"PheCODE": "PHECODE"}),
    ]).drop_duplicates()
    prior_existing_diagnosis_long['case_status'] = 'prior_case_exact' # Not a case - had the exact PHECODE diagnosis prior to start of actigraphy
    # add in a prior 'diagnosis' for anyone with an excluded phecode
    prior_from_exclusions = pandas.merge(
            prior_existing_diagnosis_long,
            phecode_exclusions,
            left_on=["PHECODE"],
            right_on=["excluded"]
        )[['ID', 'phecode']].rename(columns={"phecode": "PHECODE"})
    prior_from_exclusions['case_status'] = 'prior_case' # Not a case - had one of the excluded PHECODE diagnoses prior to the start of actigraphy
    prior_existing_diagnosis_long = (
        pandas.concat([
            prior_existing_diagnosis_long,
            prior_from_exclusions,
        ])
        .drop_duplicates()
        .dropna()
    )

    # And those who obtain the diagnosis first within the lag period
    within_lag_period_diagnosis_long = icd10_entries.loc[
        (first_date <= lagged_cutoff_time) & (first_date > exact_cutoff_time),
        ['ID', 'PHECODE'],
    ]
    # add in a lag-period 'diagnosis' for anyone with an excluded phecode
    within_lag_period_diagnosis_long = pandas.concat([
        within_lag_period_diagnosis_long,
        pandas.merge(
            within_lag_period_diagnosis_long,
            phecode_exclusions,
            left_on=["PHECODE"],
            right_on=["excluded"]
        )[['ID', 'phecode']].rename(columns={"phecode": "PHECODE"})
    ]).drop_duplicates().dropna()
    within_lag_period_diagnosis_long['case_status'] = 'within_lag_period_case' # Not a  case

    # Classify each person as prior_case, within_lag_period_case, or case
    # Order the three possibilities and take the first occurence within each individual
    case_status_basic = pandas.concat([
        icd10_entries_after_actigraphy,
        prior_existing_diagnosis_long,
        within_lag_period_diagnosis_long,
    ])
    case_status_basic['case_status'] = case_status_basic['case_status'].astype('category').cat.set_categories(['prior_case_exact', 'prior_case', 'within_lag_period_case', 'case']).cat.as_ordered()
    # We convert the three case status categories to integers since it's strangely MUCH faster than using them as categorical values for this
    coding = {"prior_case_exact": 0, "prior_case": 1, "within_lag_period_case": 2, "case": 3}
    reverse_coding = {b:a for a,b in coding.items()}
    case_status_basic['case_status_coded'] = case_status_basic['case_status'].map(coding).astype(int)
    case_status_basic = (
        case_status_basic
            .groupby(["ID", "PHECODE"])
            .case_status_coded
            .min() # prioritize prior_case status, then within_lag_period_case, then case
            .astype('category')
            .cat.rename_categories(reverse_coding)
    )

    # Create the result dataframe
    phecode_first_date = icd10_entries_after_actigraphy.groupby(['ID', 'PHECODE']).first_date.min()
    case_status = pandas.DataFrame({
        "case_status": case_status_basic,
        "first_date": phecode_first_date,
    }).reset_index()
    case_status.loc[case_status.case_status.isin(['prior_case', 'prior_case_exact']), 'first_date'] = float("NaN")
    case_status['first_date'] = pandas.to_datetime(case_status.first_date)

    # summarize the phecode contents
    details_file = pathlib.Path(OUTDIR / "phecode_details.txt")
    if RECOMPUTE or not details_file.exists():
        phecode_groups = list(phecode_info.index)
        phecode_group_details = {}
        icd10_entries_by_id = icd10_entries_raw.set_index("ID")
        cases_only = case_status.query("case_status == 'case'")
        for group in phecode_groups:
            ICD10_codes = sorted(phecode_map_extended.loc[phecode_map_extended.PHECODE == group].ICD10)
            ICD10_codes_stripped = [code.replace('.', '') for code in ICD10_codes]
            case_subjects = cases_only[(cases_only.PHECODE == group)].ID
            icd10_for_cases = icd10_entries_by_id.loc[case_subjects]
            icd10_counts = icd10_for_cases[icd10_for_cases.ICD10.isin(ICD10_codes_stripped)].ICD10.value_counts()
            case_counts = [icd10_counts.get(icd10,default=0) for icd10 in ICD10_codes_stripped]
            phecode_group_details[group] = {
                "Meaning": phecode_info.phenotype.loc[group],
                "Category": phecode_info.category.loc[group],
                "phecodes": ';'.join([group] + list(phecode_parents[phecode_parents.parent == group].phecode)),
                "ICD10_codes": ';'.join(f'{icd10} ({count})' for icd10, count in zip(ICD10_codes, case_counts)),
                "ICD9_codes": ';'.join(sorted(phecode_map_icd9_extended.loc[phecode_map_icd9_extended.PHECODE == group].ICD9)),
                "self_reported_condition_codes": ';'.join(sorted(self_report_phecode_map_extended.loc[self_report_phecode_map_extended.PheCODE == group,'Meaning'])),
                "controls_excluded_phecode": ";".join(sorted(phecode_exclusions.excluded[phecode_exclusions.phecode == group])),
            }
        phecode_details = pandas.DataFrame(phecode_group_details)
        phecode_details.to_csv(details_file, sep="\t", index=False)
    else:
        phecode_details = pandas.read_csv(details_file, sep="\t")

    return case_status, phecode_info, phecode_details
