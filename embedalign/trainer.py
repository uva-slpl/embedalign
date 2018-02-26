import logging
import os
import sys

from collections import namedtuple
import dill
import numpy as np
import tensorflow as tf

from dgm4nlp.annealing import AnnealingSchedule
from dgm4nlp.nlputils import AERSufficientStatistics
from dgm4nlp.nlputils import Multitext
from dgm4nlp.nlputils import class_distribution
from dgm4nlp.tf.bilingual import prepare_test
from dgm4nlp.tf.bilingual import prepare_training
from dgm4nlp.tf.bilingual import prepare_validation
from dgm4nlp.tf.opt import minimise
from dgm4nlp.tf.ssoftmax import np_get_support
from dgm4nlp.tf.checkpoint import ModelTracker

from embedalign.model import EmbedAlignModel
from embedalign.model import VariationalApproximationSpecs
from embedalign.viterbi import get_viterbi


class EmbedAlignTrainer:
    """
    Takes care of training a model with SGD.
    """

    def __init__(self, model: EmbedAlignModel,
                 training, val, val_wa, test, test_wa,
                 output_dir,
                 # architecture
                 nb_softmax_samples=[0, 0],
                 softmax_approximation=['botev-batch', 'botev-batch'],
                 softmax_q_inv_temperature=[0., 0.],
                 # optimisation
                 num_epochs=5,
                 batch_size=16,
                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
                 grad_clipping=None,
                 # annealing step
                 annealing_s=AnnealingSchedule(),
                 annealing_z=AnnealingSchedule(),
                 # tensorflow
                 session=None):
        """

        :param model: EmbedAlignModel
        :param training: training Multitext
        :param val: validation Multitext
        :param val_wa: validation manual alignments
        :param test: test Multitext
        :param test_wa: test manual alignments
        :param output_dir: where to save files
        :param nb_softmax_samples: enable sampled softmax approximations for P(X|z) and P(Y|x_a,z_a)
        :param softmax_approximation: choose type of sampled softmax approximations for P(X|z) and P(Y|x_a,z_a)
        :param softmax_q_inv_temperature: choose temperature for proxy in 'botev-batch' approximation
        :param num_epochs: maximum number of epochs
        :param batch_size: maximum samples in a batch
        :param optimizer: a tf optimiser
        :param grad_clipping: configure gradient clipping with a pair [floor, ceil]
        :param annealing_s: configure annealing for KL term associated with prior P(S)
        :param annealing_z: configure annealing for KL term associated with prior P(Z|s)
        :param session: a tf session
        """

        self.model = model
        self.session = session
        self.tracker = ModelTracker(session, model, output_dir)

        # Configure softmax approximations
        self.nb_softmax_samples = nb_softmax_samples
        self.softmax_approximation = softmax_approximation
        self.softmax_q_inv_temperature = softmax_q_inv_temperature
        #  for X
        if softmax_q_inv_temperature[0] > 0:
            logging.info('Computing x-side class frequencies (inverse temperature %f)', softmax_q_inv_temperature[0])
            self.freq_x = class_distribution(self.training, stream=0, batch_size=1000, normalise=False)
            self.freq_x **= softmax_q_inv_temperature[0]
        else:
            self.freq_x = None
        #  for Y
        if softmax_q_inv_temperature[1] > 0:
            logging.info('Computing y-side class frequencies (inverse temperature %f)', softmax_q_inv_temperature[1])
            self.freq_y = class_distribution(self.training, stream=1, batch_size=1000, normalise=False)
            self.freq_y **= softmax_q_inv_temperature[1]
        else:
            self.freq_y = None

        # Configure experiment
        os.makedirs(output_dir, exist_ok=True)
        self.training = training
        self.val = val
        self.val_wa = val_wa
        self.output_dir = output_dir
        self.test = test
        self.test_wa = test_wa

        # Configure loss
        self.annealing_s = annealing_s
        self.annealing_z = annealing_z

        # Configure optimiser
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.grad_clipping = grad_clipping
        self._build_optimizer()

    def _build_optimizer(self):
        """Buid the optimizer."""
        self.optimizer_state = minimise(
            loss=self.model.loss,
            optimizer=self.optimizer,
            grad_clipping=self.grad_clipping
        )

    def get_generator(self, corpus: Multitext, training_phase=None):
        if training_phase:
            generator = corpus.batch_iterator(
                batch_size=self.batch_size,
                endless=False,
                shorter_batch='trim',
                dynamic_sequence_length=True
            )
        else:
            generator = corpus.batch_iterator(
                batch_size=1,
                endless=False,
                shorter_batch='trim',
                dynamic_sequence_length=True
            )
        for batch in generator:
            (x, mx), (y, my) = batch
            # input to the TF graph
            is_training = corpus is self.training
            feed_dict = {
                self.model.x: x,
                self.model.y: y,
                self.model.training_phase: is_training,
                self.model.alpha_s: self.annealing_s.alpha(),
                self.model.alpha_z: self.annealing_z.alpha()
            }

            # additional inputs for approximations

            # X side
            vx = self.training.vocab_size(0)
            if 0 < self.nb_softmax_samples[0] < vx and self.softmax_approximation[0] == 'botev-batch':  # in case we are employing a sampled softmax
                support_x, importance_x, nb_probable_x = np_get_support(  # we sample the support using numpy
                    nb_classes=vx,
                    nb_samples=self.nb_softmax_samples[0],
                    labels=x,
                    is_training=is_training,
                    freq=self.freq_x
                )
                feed_dict[self.model.support_x] = support_x
                feed_dict[self.model.importance_x] = importance_x
                #logging.info('X: probable=%s negative=%s', nb_probable_x, support_x.shape[0] - nb_probable_x)

            # Y side
            vy = self.training.vocab_size(1)
            if 0 < self.nb_softmax_samples[1] < vy and self.softmax_approximation[1] == 'botev-batch':  # in case we are employing a sampled softmax
                support_y, importance_y, nb_probable_y = np_get_support(  # we sample the support using numpy
                    nb_classes=vy,
                    nb_samples=self.nb_softmax_samples[1],
                    labels=y,
                    is_training=is_training,
                    freq=self.freq_y
                )
                feed_dict[self.model.support_y] = support_y
                feed_dict[self.model.importance_y] = importance_y
                #logging.info('Y: probable=%s negative=%s', nb_probable_y, support_y.shape[0] - nb_probable_y)

            yield feed_dict

    def checkpoint(self, step, training_loss):

        save_path = self.model.save(self.session, path="{}/model.latest.ckpt".format(self.output_dir))
        logging.info("Model saved in file: %s", save_path)

        self.tracker.register(  # track best training loss
            step=step,
            name='training.objective',
            value=training_loss,
            asc=False
        )

        # evaluate on development set
        if not self.val:
            return


        val_aer, val_acc_x, val_acc_y, val_loss, val_ce_x, val_ce_y, val_kl = self.evaluate(self.val,
                                                                                            iter(self.val_wa))

        logging.info("Steps {} val_loss {:.4f} val_acc_x {:1.4f} val_acc_y {:1.4f} val_aer {:1.2f} ".format(
            step,
            val_loss,
            val_acc_x,
            val_acc_y,
            val_aer))

        # Save best models according to different validation metrics


        self.tracker.register(
            step=step,
            name='validation.objective',
            value=val_loss,
            asc=False
        )

        self.tracker.register(
            step=step,
            name='validation.aer',
            value=val_aer,
            asc=False
        )

        #self.tracker.register(
        #    step=step,
        #    name='validation.acc_x',
        #    value=val_acc_x,
        #    asc=True
        #)

        #self.tracker.register(
        #    step=step,
        #    name='validation.acc_y',
        #    value=val_acc_y,
        #    asc=True
        #)

    def train(self, ckpt_interval=1000):
        """Trains a model."""

        steps, epoch_steps = 0, 1
        loss = 0.0

        for epoch_id in range(1, self.num_epochs + 1):

            loss = 0.0
            ce_x = 0.0
            ce_y = 0.0
            kl = 0.0
            kl_s = 0.0
            kl_z = 0.0
            acc_x_correct = 0
            acc_x_total = 0
            acc_y_correct = 0
            acc_y_total = 0

            for epoch_steps, feed_dict in enumerate(self.get_generator(self.training, training_phase=True), 1):

                # things we want TF to return to us from the computation
                fetches = {
                    "optimizer_state": self.optimizer_state,
                    "loss": self.model.loss,
                    "ce_x": self.model.ce_x,
                    "ce_y": self.model.ce_y,
                    "kl_s": self.model.kl_s,
                    "kl_z": self.model.kl_z,
                    "kl": self.model.kl,
                    "acc_x_correct": self.model.accuracy_x_correct,
                    "acc_x_total": self.model.accuracy_x_total,
                    "acc_y_correct": self.model.accuracy_y_correct,
                    "acc_y_total": self.model.accuracy_y_total,
                    "pa_x": self.model.pa_x,
                    "px_z": self.model.px_z,
                    "py_xza": self.model.py_xza
                }

                res = self.session.run(fetches, feed_dict=feed_dict)

                loss += res["loss"]
                # breakdown of loss
                ce_x += res['ce_x']
                ce_y += res['ce_y']
                kl_s += res["kl_s"]
                kl_z += res["kl_z"]
                kl += res["kl"]
                # accuracy of next word prediction
                acc_x_correct += res["acc_x_correct"]
                acc_x_total += res["acc_x_total"]
                acc_y_correct += res["acc_y_correct"]
                acc_y_total += res["acc_y_total"]
                steps += 1

                # Log some running averages
                if epoch_steps % 100 == 0:
                    logging.info("Epoch {:3d} Steps {:5d} Iteration {:5d} loss {:.4f} ce_x {:.4f} ce_y {:.4f} kl {:.4f} kl_s {:.4f} kl_z {:.4f} alpha_s {:.4f} alpha_z {:.4f} accuracy_x {:1.4f} accuracy_y {:1.4f}".format(
                        epoch_id, steps, epoch_steps, loss / epoch_steps, ce_x / epoch_steps, ce_y / epoch_steps, kl / epoch_steps, kl_s / epoch_steps, kl_z / epoch_steps,
                        self.annealing_s.alpha(),
                        self.annealing_z.alpha(),
                        acc_x_correct / acc_x_total,
                        acc_y_correct / acc_y_total))

                # Possibly update annealing schedules after each parameter update
                self.annealing_s.update()
                self.annealing_z.update()

                if steps % ckpt_interval == 0:
                    #check point at every epoch
                    self.checkpoint(steps, loss / float(epoch_steps))

            # Log end of epoch
            logging.info("End of Epoch {:3d} Steps {:5d} loss {:.4f} ce_x {:.4f} ce_y {:.4f} kl {:.4f} kl_s {:.4f} kl_z {:.4f} alpha_s {:.4f} alpha_z {:.4f} accuracy_x {:1.4f} accuracy_y {:1.4f}".format(
                epoch_id, steps, loss / epoch_steps, ce_x / epoch_steps, ce_y / epoch_steps, kl / epoch_steps, kl_s / epoch_steps, kl_z / epoch_steps,
                self.annealing_s.alpha(),
                self.annealing_z.alpha(),
                acc_x_correct / acc_x_total,
                acc_y_correct / acc_y_total))

        # Final checkpoint
        self.checkpoint(steps, loss / float(epoch_steps))
        if self.val:  # compile a report
            with open('%s/report-best.txt' % self.output_dir, 'w') as fo:
                for metric, metric_data in self.tracker:
                    print('metric=%s value=%s step=%d path=%s' %
                          (metric,
                           metric_data.value,
                           metric_data.step,
                           metric_data.path), file=fo)

    def evaluate(self, corpus: Multitext, wa_iterator=None):
        """Evaluate the model on a data set."""
        metric = AERSufficientStatistics()
        accuracy_x_correct = 0
        accuracy_x_total = 0
        accuracy_y_correct = 0
        accuracy_y_total = 0
        loss = 0.0
        ce_x = 0.0
        ce_y = 0.0
        kl = 0.0

        for batch_id, feed_dict in enumerate(self.get_generator(corpus, training_phase=False), 1):

            # things we want TF to return to us from the computation
            fetches = {
                "optimizer_state": self.optimizer_state,
                "loss": self.model.loss,
                "ce_x": self.model.ce_x,
                "ce_y": self.model.ce_y,
                "kl": self.model.kl,
                "acc_x_correct": self.model.accuracy_x_correct,
                "acc_x_total": self.model.accuracy_x_total,
                "acc_y_correct": self.model.accuracy_y_correct,
                "acc_y_total": self.model.accuracy_y_total,
                "pa_x": self.model.pa_x,
                "px_z": self.model.px_z,
                "py_xza": self.model.py_xza
            }

            res = self.session.run(fetches, feed_dict=feed_dict)

            # Track some quantities
            accuracy_x_correct += res['acc_x_correct']
            accuracy_x_total += res['acc_x_total']
            accuracy_y_correct += res['acc_y_correct']
            accuracy_y_total += res['acc_y_total']
            loss += res['loss']
            ce_x += res['ce_x']
            ce_y += res['ce_y']
            kl += res['kl']

            # Get the input as it will be necessary for Viterbi
            x = feed_dict[self.model.x]
            y = feed_dict[self.model.y]
            align, prob = get_viterbi(x, y, res['pa_x'], res['py_xza'])

            if wa_iterator is not None:
                # Compute sufficient statistics for AER
                y_len = np.sum(np.sign(y), axis=1, dtype="int64")
                for alignment, N, (sure, probable) in zip(align, y_len, wa_iterator):
                    # the evaluation ignores NULL links, so we discard them
                    # j is 1-based in the naacl format
                    pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
                    metric.update(sure=sure, probable=probable, predicted=pred)

        accuracy_x = accuracy_x_correct / float(accuracy_x_total)
        accuracy_y = accuracy_y_correct / float(accuracy_y_total)

        return metric.aer(), accuracy_x, accuracy_y, loss / batch_id, ce_x / batch_id, ce_y / batch_id, kl / batch_id


def test_embedalign(  # input paths
        training_x, training_y,
        val_x, val_y, val_naacl,
        test_x, test_y, test_naacl,
        # output paths
        output_dir,
        ckpt_interval=1000,
        # data preprocessing
        nb_words=[1000, 1000],
        shortest_sequence=[1, 1],
        longest_sequence=[30, 30],
        bos_str=['-NULL-', None],
        eos_str=[None, None],
        lowercase=True,
        # architecture
        nb_softmax_samples=[0, 0],
        softmax_approximation=['botev-batch', 'botev-batch'],
        softmax_q_inv_temperature=[0., 0.],
        dx=128,
        dh=128,
        q=VariationalApproximationSpecs(dz=128),
        mc_kl=False,
        attention_z=False,
        improved_features_for_py=False,
        # optimiser
        nb_epochs=50,
        batch_size=100,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
        grad_clipping=None,
        annealing_s=AnnealingSchedule(),
        annealing_z=AnnealingSchedule(),
        config=None
):
    tks, training = prepare_training(training_x,
                                     training_y,
                                     nb_words=nb_words,
                                     shortest_sequence=shortest_sequence,
                                     longest_sequence=longest_sequence,
                                     bos_str=bos_str,
                                     eos_str=eos_str,
                                     lowercase=lowercase)
    logging.info("{}/tokenizer.pickle".format(output_dir))
    dill.dump(tks, open("{}/tokenizer.pickle".format(output_dir), 'wb'))

    if val_x:
        val, val_wa = prepare_validation(tks, val_x, val_y, val_naacl,
                                         shortest_sequence=shortest_sequence,
                                         longest_sequence=longest_sequence)

        test, test_wa = (None, None)#prepare_test(tks, test_x, test_y, test_naacl)
    else:
        val, val_wa = (None, None)
        test, test_wa = (None, None)

    tf.reset_default_graph()

    with tf.Session(config=config) as session:
        model = EmbedAlignModel(vx=training.vocab_size(0),
                                vy=training.vocab_size(1),
                                dx=dx, dh=dh,
                                q=q, mc_kl=mc_kl,
                                nb_softmax_samples=nb_softmax_samples,
                                softmax_approximation=softmax_approximation,
                                attention_z=attention_z,
                                improved_features_for_py=improved_features_for_py,
                                session=session)

        trainer = EmbedAlignTrainer(model,
                                    training,
                                    val, val_wa,
                                    test, test_wa,
                                    output_dir,
                                    # architecture
                                    nb_softmax_samples=nb_softmax_samples,
                                    softmax_approximation=softmax_approximation,
                                    softmax_q_inv_temperature=softmax_q_inv_temperature,
                                    # optimizer
                                    num_epochs=nb_epochs,
                                    batch_size=batch_size,
                                    optimizer=optimizer,
                                    grad_clipping=grad_clipping,
                                    annealing_s=annealing_s,
                                    annealing_z=annealing_z,
                                    session=session)

        # now first TF needs to initialize all the variables
        logging.info('Initializing variables...')
        session.run(tf.global_variables_initializer())

        logging.info('Training started')
        trainer.train(ckpt_interval=ckpt_interval)
        #TODO save model!
