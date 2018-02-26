import tensorflow as tf
import numpy as np
import dill
import os
from tabulate import tabulate
from dgm4nlp.nlputils import Multitext
from dgm4nlp.nlputils import AERSufficientStatistics
from dgm4nlp.tf.ssoftmax import np_get_support
from dgm4nlp.tf.nibm1.model import get_viterbi
from dgm4nlp.tf.bilingual import prepare_test
import argparse


class EmbedAlignDecoder:
    """
    Decode using a trained model.

    """

    def __init__(self, ckpt_path: str, vy: int, vx: int,
                 softmax_approximation='botev-batch', config=None):
        """

        :param ckpt_path: this is the path to your checkpoint files (except for suffixes such as .meta)
        :param config:
        """
        self.vy = vy
        self.vx = vx
        self.softmax_approximation = softmax_approximation
        self.meta_graph = '%s.meta' % ckpt_path
        self.ckpt_path = ckpt_path
        self.session = tf.Session(config=config)
        # load architecture computational graph
        self.new_saver = tf.train.import_meta_graph(self.meta_graph)
        # restore latest checkpoint weights
        self.new_saver.restore(self.session, self.ckpt_path)
        # create a graph
        self.graph = tf.get_default_graph()

    def get_generator(self, corpus: Multitext, batch_size):
        generator = corpus.batch_iterator(
            batch_size=batch_size,
            endless=False,
            shorter_batch='trim',
            dynamic_sequence_length=True
        )
        for batch in generator:
            (x, mx), (y, my) = batch
            # input to the TF graph
            feed_dict = {
                self.graph.get_tensor_by_name('X:0'): x,
                self.graph.get_tensor_by_name('Y:0'): y,
                self.graph.get_tensor_by_name('training_phase:0'): False,
                self.graph.get_tensor_by_name('Placeholder:0'): 1.0,
                self.graph.get_tensor_by_name('Placeholder_1:0'): 1.0,
            }
            if self.softmax_approximation == 'botev-batch':  # in case we are employing a sampled softmax
                support_x, importance_x, nb_probable_x = np_get_support(  # we sample the support using numpy
                    nb_classes=self.vx,
                    nb_samples=1,   # does not really matter because is_training is always False here
                    labels=x,
                    is_training=False,
                    freq=None  # TODO: do we need to give support to this here?
                )
                support_y, importance_y, nb_probable_y = np_get_support(  # we sample the support using numpy
                    nb_classes=self.vy,
                    nb_samples=1,  # does not really matter because is_training is always False here
                    labels=y,
                    is_training=False,
                    freq=None  # TODO: do we need to give support to this here?
                )
                feed_dict[self.graph.get_tensor_by_name('support_x:0')] = support_x
                feed_dict[self.graph.get_tensor_by_name('importance_x:0')] = importance_x
                feed_dict[self.graph.get_tensor_by_name('support_y:0')] = support_y
                feed_dict[self.graph.get_tensor_by_name('importance_y:0')] = importance_y
            yield feed_dict

    def evaluate(self, corpus: Multitext, wa_iterator, batch_size=100):
        """Evaluate the model on a data set."""
        metric = AERSufficientStatistics()
        acc_x_correct = 0
        acc_x_total = 0
        acc_y_correct = 0
        acc_y_total = 0
        loss = 0.
        n_samples = 0
        ce_x = 0.
        ce_y = 0.
        kl_s = 0.
        kl_z = 0.
        n_words_y = 0
        n_words_x = 0

        for batch_id, feed_dict in enumerate(self.get_generator(corpus, batch_size=batch_size), 1):

            # things we want TF to return to us from the computation
            fetches = {
                'pa_x': self.graph.get_tensor_by_name('pa_x:0'),
                'py_xza': self.graph.get_tensor_by_name('py_xza:0'),
                'acc_x_correct': self.graph.get_tensor_by_name('acc_x_correct:0'),
                'acc_x_total': self.graph.get_tensor_by_name('acc_x_total:0'),
                'acc_y_correct': self.graph.get_tensor_by_name('acc_y_correct:0'),
                'acc_y_total': self.graph.get_tensor_by_name('acc_y_total:0'),
                'loss': self.graph.get_tensor_by_name('loss:0'),
                'ce_x': self.graph.get_tensor_by_name('ce_x:0'),
                'ce_y': self.graph.get_tensor_by_name('ce_y:0'),
                'kl_s': self.graph.get_tensor_by_name('kl_s:0'),
                'kl_z': self.graph.get_tensor_by_name('kl_z:0'),
            }

            res = self.session.run(fetches, feed_dict=feed_dict)

            # Get the input as it will be necessary for Viterbi
            x = feed_dict[self.graph.get_tensor_by_name('X:0')]
            y = feed_dict[self.graph.get_tensor_by_name('Y:0')]
            align, prob = get_viterbi(x, y, res['pa_x'], res['py_xza'])
            x_len = np.sum(np.sign(x), axis=1, dtype="int64")
            y_len = np.sum(np.sign(y), axis=1, dtype="int64")
            acc_x_correct += res['acc_x_correct']
            acc_x_total += res['acc_x_total']
            acc_y_correct += res['acc_y_correct']
            acc_y_total += res['acc_y_total']
            samples_in_batch = len(y_len)
            loss += res['loss'] * samples_in_batch  # undo mean
            ce_x += res['ce_x'] * samples_in_batch  # undo mean
            ce_y += res['ce_y'] * samples_in_batch  # undo mean
            kl_z += res['kl_z'] * samples_in_batch
            kl_s += res['kl_s'] * samples_in_batch

            n_samples += samples_in_batch
            n_words_x += np.sum(x_len)
            n_words_y += np.sum(y_len)

            # Compute sufficient statistics for AER
            for alignment, N, (sure, probable) in zip(align, y_len, wa_iterator):
                # the evaluation ignores NULL links, so we discard them
                # j is 1-based in the naacl format
                pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
                metric.update(sure=sure, probable=probable, predicted=pred)

        accuracy_x = acc_x_correct / float(acc_x_total)
        accuracy_y = acc_y_correct / float(acc_y_total)

        return metric.aer(), accuracy_x, accuracy_y, loss / n_samples, np.power(2, (ce_x + kl_z + kl_s) / n_words_x), np.power(2, (ce_y + kl_z + kl_s) / n_words_y)


def main(ckpt_path, tokenizer_path, test_x, test_y, test_naacl, gpu='0', batch_size=100):
    """
    :param ckpt_path: path to checkpoint saved by trainer (without suffix .meta)
    :param tokenizer_path: path to tokenizer saved by trainer
    :param test_x: a path or a list of paths (for multiple test sets)
    :param test_y: a path or a list of paths (for multiple test sets)
    :param test_naacl: a path or a list of paths (for multiple test sets)
    :param gpu: which GPU to use (as a string)
    :return:
    """
    import dgm4nlp.device as dev
    config = dev.tf_config(gpu, allow_growth=True)
    tks = dill.load(open(tokenizer_path, 'rb'))
    if type(test_x) is str:
        test_x = [test_x]
        test_y = [test_y]
        test_naacl = [test_naacl]
    decoder = EmbedAlignDecoder(ckpt_path=ckpt_path, config=config, vx=tks[0].vocab_size(), vy=tks[1].vocab_size())
    header = ['dataset', 'loss', 'perp_x', 'perp_y', 'acc_x', 'acc_y', 'AER']
    rows = []
    for x, y, a in zip(test_x, test_y, test_naacl):
        test, test_wa = prepare_test(tks, x, y, a)
        aer, acc_x, acc_y, loss, perp_x, perp_y = decoder.evaluate(test, iter(test_wa), batch_size=batch_size)
        rows.append([os.path.basename(a), loss, perp_x, perp_y, acc_x, acc_y, aer])
    print(tabulate(rows, header))


def get_argparse():
    parser = argparse.ArgumentParser(prog='decoder')
    parser.description = 'EmbedAlign decoding by Viterbi'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('ckpt',
                        type=str,
                        help='Checkpoint file (without suffix)')
    parser.add_argument('tokenizer',
                        type=str,
                        help='Tokenizer saved by trainer')
    parser.add_argument('--testset', '-t', default=[],
                        nargs=3, action='append',
                        help='One or more test sets each specified as a triple (x y naacl)')
    parser.add_argument('--gpu', default='0',
                        type=str,
                        help='GPU to run the model [-1 for CPU]')
    parser.add_argument('--batch-size', default=100,
                        type=int,
                        help='Batch size')
    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(
        ckpt_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        test_x=[x for x, y, a in args.testset],
        test_y=[y for x, y, a in args.testset],
        test_naacl=[a for x, y, a in args.testset],
        gpu=args.gpu
    )
