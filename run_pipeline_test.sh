# NOTE: may want to run 'snakemake -n' first to see what is going to be run
LOGFILE=snakemake.log
ERRFILE=snakemake.err
PROJECT=UKBB_FITZGERALD
MAXJOBS=80

# Can never download more than 10 at a time from UKBB, we'll use less than that to be safe
MAXDOWNLOADS=9

source venv/bin/activate
module load java/openjdk-1.8.0
bsub -P $PROJECT -oo $LOGFILE -eo $ERRFILE snakemake --cluster-config cluster.json --cluster "./tsub.py --timeout 60 '-P $PROJECT -M 10000 -oo {cluster.output} -eo {cluster.error} -J {cluster.name}'" --cluster-status "./status.py" -j $MAXJOBS --resources download=$MAXDOWNLOADS --keep-going
