# NOTE: may want to run 'snakemake -n' first to see what is going to be run
LOGFILE=snakemake.log
PROJECT=UKBB_FITZGERALD
MAXJOBS=5
# Can never download more than 10 at a time from UKBB, we'll use less than that to be safe
MAXDOWNLOADS=5

source venv/bin/activate
module load java/openjdk-1.8.0
# NOTE: removed --resources download=$MAXDOWNLOADS for now since having problem submitting jobs
bsub -P $PROJECT -oo $LOGFILE snakemake --cluster-config cluster.json --cluster "bsub -P $PROJECT -M 10000 -oo {cluster.output} -eo {cluster.error} -J {cluster.name}" -j $MAXJOBS --keep-going
