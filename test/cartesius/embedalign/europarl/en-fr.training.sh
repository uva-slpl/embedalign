#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-10:00:00
#SBATCH -p gpu
#SBATCH -J embedalign
#SBATCH --array=1-2

TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
X=en
YN=fr
Y=fr.dummy
CORPUS=europarl
DATADIR=${HOME}/data/${CORPUS}/${X}-${YN}
MODEL=embedalign
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,europarl"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,anneal,europarl"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,europarl"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,anneal,europarl"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,europarl,dummy"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,europarl,dummy"
#ARCHITECTURE="gen:128,inf:bow-z,opt:adam,anneal,europarl,dummy"
ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,anneal,europarl,dummy"
#ARCHITECTURE="gen:128,inf:rnn-z,opt:adam,wait,anneal,europarl"
#ARCHITECTURE="gen:128,inf:rnn-s_rnn-z,opt:adam,e:10,anneal,europarl"
#ARCHITECTURE="gen:128,inf:rnn-s_rnn-z,opt:adam,e:10,wait,anneal,europarl"
COPY_OUTPUT_DIR="${HOME}/experiments/${MODEL}/${CORPUS}.${X}-${YN}/${ARCHITECTURE}/${TIMESTAMP}"
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius
TRAINER=${SCRIPTSDIR}/${MODEL}/${MODEL}.py
VENV=${HOME}/envs/dgm4nlp

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/train.sh

wait
