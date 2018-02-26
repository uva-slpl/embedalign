# This script assume the following variables have been set
# Details of the training: this is the stuff you can change
#DECODER=
#X=
#Y=
#CORPUS=
#DATADIR=
#MODEL=
#TESTSET=
#GPU=
#BATCHSIZE=
#VENV=
#CRITERION=
#MODELS[0]=
#MODELS[1]=
# etc.


# Load modules and python environment
module load python/3.5.2
module load gcc/5.2.0
module load cuda/8.0.44
module load cudnn/8.0-v6.0
# Source python venv
source ${VENV}/bin/activate


# Decode and compute AER
for model in ${MODELS[@]}; do
    ckpt=${model}/model.${CRITERION}.ckpt
    tokenizer=${model}/tokenizer.pickle
    output="${model}/`basename ${TESTSET}`.${CRITERION}.results"
    python -m embedalign.decoder \
        ${ckpt} \
        ${tokenizer} \
        --testset ${TESTSET}.${X} ${TESTSET}.${Y} ${TESTSET}.naacl \
        --gpu ${GPU} \
        --batch-size ${BATCHSIZE} > ${output} 
    echo "$output"
done


wait
