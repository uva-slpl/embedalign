#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-0:10:00
#SBATCH -p gpu
#SBATCH -J venv

# Load modules and python environment
module load python/3.5.2

# Delete old environment if exists
rm -fr ${HOME}/envs/dgm4nlp

# Create a new one
mkdir -p ${HOME}/envs
python3 -m venv ${HOME}/envs/dgm4nlp

# Source the environment
source ${HOME}/envs/dgm4nlp/bin/activate

echo "Virtual environment: ${HOME}/envs/dgm4nlp"
echo "Using python from venv: `which python`"

module load gcc/5.2.0
module load cuda/8.0.44
module load cudnn/8.0-v6.0

# Install all dependencies

pip install --upgrade pip
pip install yolk3k
pip install tensorflow-gpu
pip install tabulate
pip install numpy
pip install scipy
pip install dill

# Install dgm4nlp (in develop mode)
cd ${HOME}/git/dgm4nlp
python3 setup.py develop
cd ${HOME}/git/embedalign
python3 setup.py develop

# Test installation
cd 
python3 -c "import dgm4nlp;"
python3 -c "import embedalign;"

# List packages in environment
echo "Packages installed:"
yolk -l

