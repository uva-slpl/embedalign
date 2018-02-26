#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-6:00:00
#SBATCH -p gpu
#SBATCH -J embedalign
#SBATCH --array=1-5

TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=handsards
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
# estimate (30 epochs): < 1h30
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,anneal"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam-50epochs,anneal"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,anneal-wait"
# estimate (30 epochs): < 3h
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,anneal"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam-50epochs,anneal"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,anneal-wait"
#####
COPY_OUTPUT_DIR="${HOME}/experiments/${MODEL}/${CORPUS}.${X}-${Y}/${ARCHITECTURE}/${TIMESTAMP}"
DGM4NLP=${HOME}/git/dgm4nlp/
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius
TRAINER=${SCRIPTSDIR}/${MODEL}/${MODEL}.py
VENV=${HOME}/envs/dgm4nlp

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/train.sh

wait

