#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 5-00:00:00
#SBATCH -p gpu
#SBATCH -J giga


TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=giga
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
RUNNER=embedalign-giga
ARCHITECTURE="gen:256,inf:rnn-z,opt:adam,giga"
#ARCHITECTURE="gen:256,inf:rnn-s_rnn-z,opt:adam,giga"
COPY_OUTPUT_DIR="${HOME}/experiments/${MODEL}/${CORPUS}.${X}-${Y}/${ARCHITECTURE}/${TIMESTAMP}"
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius
TRAINER=${SCRIPTSDIR}/${MODEL}/${RUNNER}.py
VENV=${HOME}/envs/dgm4nlp

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/train.sh

wait
