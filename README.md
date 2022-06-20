# UK Biobank Actigraphy Anlaysis - Biorhythm Atlas

A study associating actigraphy (wrist worn accelerometry and associated sensors) data to disease phenotypes in the UK Biobank.

This repository contains a Snakemake pipeline to process and aggregate the actigraphy data in the UK Biobank.
It then contains scripts (longitudinal_analysis.py) to generate figures and results tables for our [Temperature Biorhythm Atlas](http://bioinf.itmat.upenn.edu/biorhythm_atlas/) phenome-wide scan of predictive power of temperature rhythms for disease diagnoses.

Temperature rhythms were assessed from the included temperature sensor embedded in the actigraphy device (an Axivity AX3).
These temperature rhythms capture the expected phenotype of wrist temperature in the extremities (higher at night, with an approximately 6 degrees C difference from peak to trough).
A cosinor fit was performed to determine the amplitude (difference from peak to mid-level of the cosine fit) of the amplitude. 

Phenotypes were associated via the [PheCODE](https://phewascatalog.org/phecodes) map, which maps ICD9 and ICD10 codes to phenotypes.
In-patient hospital records were used to determine diagnoses that occured after the actigraphy recording.
A Cox proportional hazards model was then used to determine the associations of the temperature amplitude with these diagnoses.
Individuals with low temperature amplitude were more likely to develop diseases across a wide range of conditions.
