#!/usr/bin/env python
'''
This script is used for compliance with the UKBB's occaisonal requests
for removal of individuals from the study.
'''

import argparse

parser = argparse.ArgumentParser(
        description="""Sometimes individuals withdraw from the UK Biobank studies. This script removes their records from our data to comply with their withdrawal.""")
parser.add_argument("id_file", help="text file with EIDs of the participants to be removed, one on each line")
parser.add_argument("temp_dir", help="directory to move things to do before deleting them")
parser.add_argument("--yes",
                    help="CAUTION: performs all operations without yes/no prompting each",
                    default=False,
                    action="store_true")
args = parser.parse_args()

import pathlib
import collections

ids_removing = open(args.id_file).readlines()
ids_removing = [id.strip() for id in ids_removing]
print(f"Removing: {','.join(ids_removing)}")

temp_dir = pathlib.Path(args.temp_dir)
del_dir = temp_dir / "delete"
del_dir.mkdir(exist_ok=True)
overwrite_dir = temp_dir / "overwrite"
overwrite_dir.mkdir(exist_ok=True)
original_dir = temp_dir / "originals"
original_dir.mkdir(exist_ok=True)


import pandas
hdf_files_to_clean = pathlib.Path("../processed/").glob("*.h5")
for file in hdf_files_to_clean:
    data = pandas.read_hdf(str(file))
    remove = data.index.isin(ids_removing)
    new_data = data[~remove]

    print(f"\tWas {data.shape} before")
    print(f"\tNow {new_data.shape} before")

    dir = file.parent.name
    long_name = f"{dir}.{file.name}"
    new_file = overwrite_dir / long_name
    new_data.to_hdf(str(new_file), key="data", sep="\t", mode="w", format="table")

    yn = input(f"Confirm re-writing of {file} (y/n)")
    if yn in ["y", "Y"] or args.yes:
        backup = original_dir / long_name
        print(f"Moving {file} to {backup}")
        file.rename(backup)
        new_file.rename(file)

tab_separated_files_to_clean = [*pathlib.Path("../data/").glob("*.tab"),
                                 *pathlib.Path("../processed/").glob("*.txt")]

for file in tab_separated_files_to_clean:
    print(f"Processing {file}")

    with file.open() as f:
        data = f.readlines()

    # We want to handle some files that have two indexes with ID being the second one
    # so check the header
    # NOTE: this is a hacky solution, so check the outputs!
    ID_index = 0
    header = data[0].split('\t')
    if header[1] == "ID":
        ID_index = 1


    def remove_line(line):
        id_entry = line.split("\t")[ID_index]
        if id_entry in ids_removing:
            return True
        else:
            return False

    new_data = [line for line in data if not remove_line(line)]
    print(f"\tWas {len(data)} lines")
    print(f"\tNow {len(new_data)} lines")
    dir = file.parent.name
    long_name = f"{dir}.{file.name}"
    new_file = overwrite_dir / long_name
    new_file.write_text(''.join(new_data))

    yn = input(f"Confirm re-writing of {file} (y/n)")
    if yn in ["y", "Y"] or args.yes:
        backup = original_dir / long_name
        print(f"Moving {file} to {backup}")
        file.rename(backup)
        new_file.rename(file)

dirs_to_scrub = ["../processed/acc_analysis",
                 "../data/raw_actigraphy",
                 "../processed/activity_features"]
for dir in dirs_to_scrub:
    print(f"Scrubbing {dir} directory")
    for id in ids_removing:
        files = list(pathlib.Path(dir).glob(f"{id}*"))
        if not files:
            continue

        file_names = '\n\t'.join(str(f) for f in files)
        print(f"Removing: {file_names}")
        yn = input(f"Confirm moving to temp dir? (y/n)")
        if yn in ["y", "Y"] or args.yes:
            for f in files:
                target = del_dir / f.name
                print(f"Moving {f} to {target}")
                f.rename(target)

print("Done processing")
print(f"ALERT: Now review the files in {str(temp_dir)}")
print("If removal of IDs was successful, delete the contents of delete/ and original/")
print("And overwrite/ should be empty")
print("Removal of the IDs is NOT COMPLETE until these files have been deleted")
while True:
    print("Confirm that you know these files must still be deleted by typing YES")
    x = input('>')
    if x == "YES":
        break
