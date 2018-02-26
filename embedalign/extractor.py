import tensorflow as tf
import logging
from collections import defaultdict
import numpy as np
from dgm4nlp.tf.ssoftmax import np_get_support

class EmbeddingExtractor:

    def __init__(self, graph_file, ckpt_path, config=None):
        self.meta_graph = graph_file
        self.ckpt_path = ckpt_path
        self.sess = tf.Session(config=config)
        self.softmax_approximation = 'botev-batch' #default
        # load architecture computational graph
        self.new_saver = tf.train.import_meta_graph(self.meta_graph)
        # restore checkpoint
        self.new_saver.restore(self.sess, self.ckpt_path) #tf.train.latest_checkpoint(
        self.graph = tf.get_default_graph()
        # retrieve input variable
        self.x = self.graph.get_tensor_by_name("X:0")
        # retrieve training switch variable (True:trianing, False:Test)
        self.training_phase = self.graph.get_tensor_by_name("training_phase:0")
        #self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")



    def get_z_embedding_batch(self, x_batch, n_samples=1):
        # Retrieve embeddings from latent variable Z
        # we can sempale several n_samples, default 1
        try:
            z_mean = self.graph.get_tensor_by_name("Z:0")
            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
                #self.keep_prob: 1.

            }
            z_rep_values = [self.sess.run(z_mean, feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensor Z not in graph!')
        return z_rep_values


    def get_mean_std_batch(self, x_batch, n_samples=1):
        # Retrieve embeddings from latent variable Z
        # we can sempale several n_samples, default 1
        try:
            mean = self.graph.get_tensor_by_name("z-mean:0")
            log_var = self.graph.get_tensor_by_name('z-var:0')
            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
                #self.keep_prob: 1.

            }
            rep_values = [self.sess.run([mean, log_var], feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensors mean and std not in graph!')
        return rep_values

    def _s_type(self, x_batch, nb_classes, nb_samples, is_training):
        support_y, importance_y, nb_probable_y = np_get_support(  # we sample the support using numpy
            nb_classes=nb_classes,
            nb_samples=nb_samples,  # does not really matter because is_training is always False here
            labels=None,
            is_training=is_training,
            freq=None  # TODO: do we need to give support to this here?
        )
        y = self.graph.get_tensor_by_name("Y:0")
        B = len(x_batch)
        #M = len(x_batch[0])
        y_empt_batch = np.zeros([B, 1])
        feed_dict = {
            self.x: x_batch,
            y: y_empt_batch, #nothing really??
            self.training_phase: is_training
        }
        if self.softmax_approximation == 'botev-batch':
            feed_dict[self.graph.get_tensor_by_name('support_y:0')] = support_y
            feed_dict[self.graph.get_tensor_by_name('importance_y:0')] = importance_y
        return feed_dict


    def get_py_xza(self, x_batch, vy, sample_type='stochastic', n_samples=1):
        try:

            py_xza = self.graph.get_tensor_by_name("py_xza:0")

            if sample_type == 'discrete':
                feed_dict = self._s_type(x_batch, vy, 1, False) #try true
            elif sample_type == 'stochastic':
                feed_dict = self._s_type(x_batch, vy, vy, True)

            rep_values = [self.sess.run(py_xza, feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensor py_xza not in graph!')
        return rep_values

    def get_mustd_py_xza(self, x_batch, vy, sample_type='discrete', n_samples=1):
        try:
            mean = self.graph.get_tensor_by_name("z-mean:0")
            log_var = self.graph.get_tensor_by_name('z-var:0')
            py_xza = self.graph.get_tensor_by_name("py_xza:0")

            if sample_type == 'discrete':
                feed_dict = self._s_type(x_batch, vy, 1, False)
            elif sample_type == 'stochastic':
                feed_dict = self._s_type(x_batch, vy, vy, True)

            rep_values = [self.sess.run([mean, log_var, py_xza], feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensor py_xza not in graph!')
        return rep_values

    def get_s_embedding_batch(self, x_batch, n_samples=1):
        # Retrieve embeddings from latent variable S
        # we can sample several n_samples, default 1
        try:
            s_mean = self.graph.get_tensor_by_name("S/Merge:0")
            feed_dict = {
                self.x: x_batch,
                self.training_phase: False,
            }
            s_rep_values = [self.sess.run(s_mean, feed_dict=feed_dict) for _ in range(n_samples)]
        except:
            raise ValueError('tensor S not in graph!')
        return s_rep_values



#if __name__ == '__main__':
    # example of usage we need the computationa garph and parameters  check point (ckpt) path
    #ckpt_path = '/mnt/data/mrios/dgm4nlp/naacl.test.tf/17-08-08.10h12m01s.6smqlaqt/'
    #graph_file = '/mnt/data/mrios/dgm4nlp/naacl.test.tf/17-08-08.10h12m01s.6smqlaqt/model.ckpt.meta'
    #emdedd_align = EmbeddingExtractor(
    #    graph_file=graph_file,
    #    ckpt_path=ckpt_path,
    #    token_map=None
    #)


    #x = np.random.randint(0, 10000, size=(50, 30))
    #z = emdedd_align.get_z_embedding_batch(x_batch=x)
    #s = emdedd_align.get_s_embedding_batch(x_batch=x)
    #print(z[0].shape, s[0].shape)

