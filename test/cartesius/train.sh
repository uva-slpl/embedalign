# This script assumes global variables have already been set, namely

#X=
#Y=
#CORPUS=
#DATADIR=
#MODEL=
#ARCHITECTURE=
#COPY_OUTPUT_DIR=
#SCRIPTSDIR=
#TRAINER=
#VENV=


# Sanity checks
PASS=1
for stem in training dev test; do 
    # TODO: sanity check SCRIPTSDIR and DGM4NLP
    if [[ ! -e ${DATADIR}/${stem}.${X} ]]; then 
        echo "File not found: ${DATADIR}/${stem}.${X}"
        PASS=0
    fi
    if [[ ! -e ${DATADIR}/${stem}.${Y} ]]; then 
        echo "File not found: ${DATADIR}/${stem}.${Y}"
        PASS=0
    fi
done
if [[ $PASS == 0 ]]; then
    exit
fi

# Get a working directory on scratch space
mkdir -p ${TMPDIR}/0
mkdir -p ${TMPDIR}/1
## Schedule archive job
sbatch --dependency=afterany:${SLURM_JOB_ID} ${SCRIPTSDIR}/archive.sh ${SLURM_JOB_ID} $(readlink -f ${TMPDIR}) ${MODEL}

# Load modules and python environment
module load python/3.5.2
module load gcc/5.2.0
module load cuda/8.0.44
module load cudnn/8.0-v6.0
# Source python venv
source ${VENV}/bin/activate

# Run training: on cartesius each machine has 2 GPUs, thus we run two jobs

python3 ${TRAINER} \
    ${DATADIR} \
    ${X} \
    ${Y} \
    ${ARCHITECTURE} \
    ${TMPDIR}/0 \
    0 \
    > ${TMPDIR}/0/out 2> ${TMPDIR}/0/log &

python3 ${TRAINER} \
    ${DATADIR} \
    ${X} \
    ${Y} \
    ${ARCHITECTURE} \
    ${TMPDIR}/1 \
    1 \
    > ${TMPDIR}/1/out 2> ${TMPDIR}/1/log &

PID=$!
echo "DGM4NLP PID=${PID}"
echo "Model: ${MODEL}"
echo "Architecture: ${ARCHITECTURE}"
echo "You can check the logfile using:"
echo "log0: `readlink -f ${TMPDIR}/0/log`"
echo "log1: `readlink -f ${TMPDIR}/1/log`"
echo "You will eventually find the results in: ${COPY_OUTPUT_DIR}"

wait

# copy all output from scratch to our home dir
# if we used a dependent job we will also have a copy in the archive
mkdir -p ${COPY_OUTPUT_DIR}
rsync -vat ${TMPDIR} ${COPY_OUTPUT_DIR} &
# copy the exact version of the code that produced these models
cp -R ${DGM4NLP} ${COPY_OUTPUT_DIR}/

wait
