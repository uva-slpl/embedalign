"""
:Authors: - Wilker Aziz
"""
import tensorflow as tf
from dgm4nlp.annealing import AnnealingSchedule
from embedalign.infnet import get_infnet


def get_hparams(option):
    if option == 'gen:128,inf:bow-z,opt:adam':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-z-vardrop.9,opt:adam':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False, dropout=True, dropout_params=[1.0, 1.0, 0.9, True]),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-z-vardrop.8,opt:adam':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False, dropout=True, dropout_params=[1.0, 1.0, 0.8, True]),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-z-vardrop.7,opt:adam':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False, dropout=True, dropout_params=[1.0, 1.0, 0.7, True]),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:bow-z,opt:adam-50epochs':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam-50epochs':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:bow-z,opt:adam,anneal':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.01, nb_updates=500)  # Handsards epoch contains about 2k batches, this schedule requires 50k batches to reach 1.0, this means about 23 epochs
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam,anneal':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.01, nb_updates=500)
        )
    elif option == 'gen:128,inf:bow-z,opt:adam,anneal-wait':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=1689, step=0.01, nb_updates=500)  # Handsards epoch contains about 2.2k batches, this schedule requires 50k+2.2k batches to reach 1.0, this means about 24 epochs
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam,anneal-wait':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=1689, step=0.01, nb_updates=500)
        )
    elif option == 'gen:128,inf:bow-z,opt:adam-50epochs,anneal-wait':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=1689, step=0.01, nb_updates=500)  # Handsards epoch contains about 2.2k batches, this schedule requires 50k+2.2k batches to reach 1.0, this means about 24 epochs
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam-50epochs,anneal-wait':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=1689, step=0.01, nb_updates=500)
        )
    elif option == 'gen:128,inf:bow-z,opt:adam-50epochs,anneal':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.01, nb_updates=500)  # Handsards epoch contains about 2k batches, this schedule requires 50k batches to reach 1.0, this means about 23 epochs
        )
    elif option == 'gen:128,inf:rnn-z,opt:adam-50epochs,anneal':
        # preprocessing
        hparams = dict(
            nb_words=[30000, 30000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=50,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.01, nb_updates=500)
        )
    elif option == 'gen:256,inf:bow-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:bow-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )

    elif option == 'gen:128,inf:bow-z,opt:adam,europarl,dummy':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:64,inf:bow-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:256,inf:bow-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:128,inf:bow-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:128,inf:bow-z,opt:adam,anneal,europarl,dummy':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=100)
        )
    elif option == 'gen:64,inf:bow-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:256,inf:bow-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )

    elif option == 'gen:128,inf:bow-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )
    elif option == 'gen:64,inf:bow-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('bow-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )

    elif option == 'gen:256,inf:rnn-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )

    elif option == 'gen:128,inf:rnn-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )

    elif option == 'gen:128,inf:rnn-z,opt:adam,europarl,dummy':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:64,inf:rnn-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )

    elif option == 'gen:256,inf:rnn-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:128,inf:rnn-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:128,inf:rnn-z,opt:adam,anneal,europarl,dummy':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=100)
        )
    elif option == 'gen:64,inf:rnn-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:256,inf:rnn-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )

    elif option == 'gen:128,inf:rnn-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )
    elif option == 'gen:64,inf:rnn-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=30,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )

    elif option == 'gen:256,inf:rnn-z,opt:adam,smallL2,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 50000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=20,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:256,inf:rnn-z,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:64,inf:rnn-z,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('rnn-z', dz=100, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:256,inf:rnn-z:256,smapprox,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=256, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[100, 100],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:256,inf:rnn-z:200,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-z', dz=200, hierarchical=False),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:256,inf:rnn-s_rnn-z,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=3,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[5.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:64,inf:rnn-s_rnn-z,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=64,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=3,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[5.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )


    elif option == 'gen:256,inf:rnn-s:100_rnn-z:200,opt:adam,giga':
        # preprocessing
        hparams = dict(
            nb_words=[300000, 300000],
            longest_sequence=[50, 50],
            # architecture
            dx=256,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=200, ds=100, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=5,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:128,inf:rnn-s_rnn-z,opt:adam,e:10,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[5.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )
    elif option == 'gen:128,inf:rnn-s_rnn-z,opt:adam,e:10,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[-1., 1.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=200)
        )

    elif option == 'gen:128,inf:rnn-s_rnn-z,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-s:128_rnn-z:256,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=256, ds=128, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )
    elif option == 'gen:128,inf:rnn-s:128_rnn-z:256,opt:adam,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=256, ds=128, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule()
        )
    elif option == 'gen:128,inf:rnn-s_rnn-z,opt:adam,smallL2,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 50000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:128,inf:rnn-s:128_rnn-z:256,opt:adam,smallL2,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 50000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, rnn-z', dz=256, ds=128, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )
    elif option == 'gen:128,inf:rnn-s_bow-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('rnn-s, bow-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=150)
        )

    elif option == 'gen:128,inf:bow-s_rnn-z,opt:adam,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., step=0.001, nb_updates=500)
        )

    elif option == 'gen:128,inf:bow-s_rnn-z,opt:adam,wait,anneal,europarl':
        # preprocessing
        hparams = dict(
            nb_words=[120000, 120000],
            longest_sequence=[50, 50],
            # architecture
            dx=128,
            dh=128,
            q=get_infnet('bow-s, rnn-z', dz=100, ds=64, hierarchical=True),
            mc_kl=True,
            nb_softmax_samples=[1000, 1000],
            softmax_approximation=['botev-batch', 'botev-batch'],
            softmax_q_inv_temperature=[0., 0.],
            # optimiser
            nb_epochs=10,
            batch_size=100,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            grad_clipping=[10.],
            annealing_z=AnnealingSchedule(initial=0., final=1., wait=16500, step=0.01, nb_updates=500)
        )
     
    return hparams
