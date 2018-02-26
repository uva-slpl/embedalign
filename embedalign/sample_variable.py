import tensorflow as tf
import logging
from collections import defaultdict
import numpy as np
from dgm4nlp.tf.ssoftmax import np_get_support
import dill


class Sampler:

    def __init__(self, graph_file, ckpt_path, softmax_approximation='botev-batch', config=None):
        self.meta_graph = graph_file
        self.ckpt_path = ckpt_path
        #self.tks = dill.load(open(tokenizer_path, 'rb'))
        #self.vx = self.tks[0].vocab_size()
        self.softmax_approximation = softmax_approximation
        self.sess = tf.Session(config=config)
        # load architecture
        self.new_saver = tf.train.import_meta_graph(self.meta_graph)
        # restore last check point weights
        self.new_saver.restore(self.sess, self.ckpt_path)
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("X:0")
        self.training_phase = self.graph.get_tensor_by_name("training_phase:0")

    def _s_type(self, x_batch, nb_classes, nb_samples, is_training):
        support_x, importance_x, nb_probable_x = np_get_support(  # we sample the support using numpy
            nb_classes=nb_classes,
            nb_samples=nb_samples,  # does not really matter because is_training is always False here
            labels=x_batch,
            is_training=is_training,
            freq=None  # TODO: do we need to give support to this here?
        )

        feed_dict = {
            self.x: x_batch,
            self.training_phase: is_training
        }
        if self.softmax_approximation == 'botev-batch':
            feed_dict[self.graph.get_tensor_by_name('support_x:0')] = support_x
            feed_dict[self.graph.get_tensor_by_name('importance_x:0')] = importance_x
        return feed_dict

    def sample_xz(self, x_batch, vx, sample_type='discrete', n_samples=1):
        try:
            px_z = self.graph.get_tensor_by_name("px_z:0")
            if sample_type == 'discrete':
                feed_dict = self._s_type(x_batch, vx, 1, False)
            elif sample_type == 'stochastic':
                feed_dict = self._s_type(x_batch, vx, vx, True)

            x = [self.sess.run(px_z, feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensor P(X|z) not in graph!')
        return x




if __name__ == '__main__':

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True # Allows dynamic mem allocation from minimum
    config.gpu_options.visible_device_list = '1'
    ckpt_path = '/mnt/data/mrios/dgm4nlp/euro.en-fr.tf/17-10-19.15h23m53s.gstbk1sh/model.best.ckpt'
    graph_file = '/mnt/data/mrios/dgm4nlp/euro.en-fr.tf/17-10-19.15h23m53s.gstbk1sh/model.best.ckpt.meta'
    tokenizer_file = '/mnt/data/mrios/dgm4nlp/euro.en-fr.tf/17-10-19.15h23m53s.gstbk1sh/tokenizer.pickle'
    tks = dill.load(open(tokenizer_file, 'rb'))
    vx = tks[0].vocab_size()
    p = Sampler(
        graph_file=graph_file,
        ckpt_path=ckpt_path,
        config=config
    )


    x = np.random.randint(0, 10000, size=(1, 5))
    x_samp = p.sample_xz(x_batch=x, vx=vx, sample_type='stochastic', n_samples=2)
    print(x_samp[0].shape)
    print(x_samp)

