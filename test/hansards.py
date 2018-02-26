"""
:Authors: - Wilker Aziz
"""
import dgm4nlp.device as dev
config = dev.tf_config('0', allow_growth=True)

import os
import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime
from dgm4nlp.annealing import AnnealingSchedule
from embedalign.trainer import test_embedalign
from embedalign.infnet import get_infnet

import tensorflow as tf

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 1. Get a unique working directory and save script for reproducibility
base_dir = 'debug/hansards'
os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
logging.info('Workspace: %s', output_dir)
shutil.copy(os.path.abspath(__file__), output_dir)


test_embedalign(
    training_x='hansards/shuffled.e',
    training_y='hansards/shuffled.f',
    val_x='hansards/trial.en-fr.en',
    val_y='hansards/trial.en-fr.fr',
    val_naacl='hansards/trial.en-fr.naacl',
    test_x='hansards/test.en-fr.en',
    test_y='hansards/test.en-fr.fr',
    test_naacl='hansards/test.en-fr.naacl',
    output_dir=output_dir,
    # preprocessing
    nb_words=[30000, 30000],
    longest_sequence=[50, 50],
    # architecture
    dx=128,
    dh=128,
    q=get_infnet('rnn-z', dz=100),
    nb_softmax_samples=[1000, 1000],
    softmax_approximation=['botev-batch', 'botev-batch'],
    softmax_q_inv_temperature=[0., 0.],
    # optimiser
    nb_epochs=20,
    batch_size=100,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    grad_clipping=[-1., 1.],
    annealing_z=AnnealingSchedule(initial=0., final=1., step=0.01, nb_updates=100),
    config=config
)
