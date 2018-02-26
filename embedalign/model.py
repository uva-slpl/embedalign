import tensorflow as tf
from dgm4nlp.tf.encoder import ffnn
from dgm4nlp.tf.kl import kl_from_q_to_standard_normal
from dgm4nlp.tf.kl import kl_diagonal_gaussians
from dgm4nlp.tf.logit import logit_layer_for_text
from dgm4nlp.tf.logit import logit_layer_for_bitext
from dgm4nlp.tf.encoder import SequenceEncoder
from dgm4nlp.tf.encoder import FeedForwardEncoder
import logging


class VariationalApproximationSpecs:
    """
    Configure an inference network.
    """

    def __init__(self, dz: int, ds=None, hierarchical=False):
        """

        :param dz: dimensionality of latent word embedding
        :param ds: dimensionality of latent sentence embedding
        :param mean_field: whether S and Z_1^m are independent
            - if True, q(s, z_1^m) = q(s) \prod_i q(z_i)
            - otherwise q(s, z_1^m) = q(s) \prod_i q(z_i|s)
        """
        self.dz = dz
        self.ds = ds
        self.hierarchical = hierarchical if ds is not None else False

        # Default configuration for z
        self.z_view = 0
        self.z_mean_hidden_layers = []
        self.z_var_hidden_layers = []

        # Default configuration for s
        self.s_view = 0
        self.s_mean_hidden_layers = []
        self.s_var_hidden_layers = []

        self._encoders = [None]

    def add_encoder(self, encoder: SequenceEncoder):
        self._encoders.append(encoder)
        return len(self._encoders) - 1

    def get_encoder(self, view) -> SequenceEncoder:
        return self._encoders[view]

    def config_z(self, view: int, mean_hidden_layers=[], var_hidden_layers=[]):
        if not (0 <= view < len(self._encoders)):
            raise ValueError('You do not have enough views')
        self.z_view = view
        self.z_mean_hidden_layers = mean_hidden_layers
        self.z_var_hidden_layers = var_hidden_layers

    def config_s(self, view: int, mean_hidden_layers=[], var_hidden_layers=[]):
        if self.ds is None:
            raise ValueError('You are not modelling sentence embeddings')
        if not (0 <= view < len(self._encoders)):
            raise ValueError('You do not have enough views')
        self.s_view = view
        self.s_mean_hidden_layers = mean_hidden_layers
        self.s_var_hidden_layers = var_hidden_layers


def _predict(layers, inputs, name):
    """
    Predict by applying a multilayer transformation to inputs.

    :param layers: list of nonlinear layers specified by pairs (output_units, activation_fn)
    :param inputs: input activations [B * T, di]
    :param name:
    :return: [B * T, do]
    """
    with tf.variable_scope(name):
        # [B * M, do]
        predictions = ffnn(
            inputs,  # [B * M, di]
            layers
        )
    return predictions


class EmbedAlignModel:
    """
    Model specification for embed-align based onIBM1
    """

    def __init__(self,
                 # architecture
                 vx, vy,
                 dx=32,
                 dh=64,
                 nb_softmax_samples=[0, 0],
                 softmax_approximation=['botev-batch', 'botev-batch'],
                 q=VariationalApproximationSpecs(dz=64),
                 attention_z=False,
                 improved_features_for_py=False,
                 mc_kl=False,
                 # tf
                 session=None):
        """

        :param vx: size of vocabulary of x-language
        :param vy: size of vocabulary of y-language
        :param dx: x-embedding dimensionality
        :param dh: hidden units (for MLPs and BiRNNs)
        :param dz: dimensionality of latent encoding (and also of class embeddings)
        :param nb_softmax_samples: number or negative samples for P(X|z) and P(Y|x_a,z_a) (use zero for exact softmax)
        :param softmax_approximation: specify type of sampled logit implementation for P(X|z) and P(Y|x_a, z_a)
            - 'botev' 
            - 'jean' (not available for P(Y|x_a, z_a))
            - 'botev-batch'
        :param birnn: bidirectional encodding in the approximate posterior
        :param birnn_merge_strategy: strategy for merging RNN states: sum/concat/mlp
        :param shared_birnn_encoders: whether we use the same BiRNN for predicting mean and variance
            - if True, then mu_layers and sigma_layers can always be used to specialise the shared bidirectional
            encodings before predicting mean and variance
            - if False, you can probably set mu_layers=[] and sigma_layers=[]
        :param mu_layers: specify non-linear hidden layers for predicting mean from x's encodings
            - we will always add to it a layer with num_outputs=dz and activation_fn=None
        :param sigma_layers: specify non-linear hidden layers for predicting log variance from x's encodings
            - we will always add to it a layer with num_outputs=dz and activation_fn=None
        :param session:
        """

        self.dx = dx
        self.dh = dh
        self.vx = vx
        self.vy = vy
        self.q = q
        self.mc_kl = mc_kl

        self.nb_softmax_samples = nb_softmax_samples
        self.softmax_approximation = softmax_approximation

        self._attention_z = attention_z
        self._improved_features_for_py = improved_features_for_py

        self._create_placeholders()
        self._create_weights()
        self._build_model()

        self.saver = tf.train.Saver()
        self.session = session

    def _create_placeholders(self):
        """These are placeholders to feed data to TensorFlow."""

        self.x = tf.placeholder(tf.int64, shape=[None, None], name='X')  # (B, M)
        self.y = tf.placeholder(tf.int64, shape=[None, None], name='Y')  # (B, N)

        self.training_phase = tf.placeholder(tf.bool, name='training_phase')  # []
        self.alpha_s = tf.placeholder(tf.float32)  # [] scalar that weights the KL contribution for prior on S
        self.alpha_z = tf.placeholder(tf.float32)  # [] scalar that weights the KL contribution for prior on Z

        # here we create placeholders for a batch-wise shared sampled support for X
        if self.softmax_approximation[0] == 'botev-batch':
            self.support_x = tf.placeholder(tf.int64, shape=[None], name='support_x')  # [S]
            self.importance_x = tf.placeholder(tf.float32, shape=[None], name='importance_x')  # [S]
        else:
            self.support_x = None
            self.importance_x = None

        # here we create placeholders for a batch-wise shared sampled support for Y|x,a
        if self.softmax_approximation[1] == 'botev-batch':
            self.support_y = tf.placeholder(tf.int64, shape=[None], name='support_y')  # [S]
            self.importance_y = tf.placeholder(tf.float32, shape=[None], name='importance_y')  # [S]
        else:
            self.support_y = None
            self.importance_y = None

    def _create_weights(self):
        """Create weights for the model."""
        pass

    def save(self, session, path="model.ckpt", step=None):
        """Saves the model."""
        return self.saver.save(session, path, global_step=step)

    def _build_model(self):
        """Builds the computational graph for our model."""

        # Embedding for source words
        # (Vx, dx)
        # add glorot init of params but there is not too much diff with uniform
        x_embeddings = tf.get_variable(
            name="x_embeddings", initializer=tf.contrib.layers.xavier_initializer(), #tf.random_uniform_initializer(),
            shape=[self.vx, self.dx])

        # (B, M, dx)
        x_embedded = tf.nn.embedding_lookup(x_embeddings,  # (Vx, dx)
                                            self.x)        # (B, M)

        # these quantities are only known when the batch is provided
        batch_size = tf.shape(self.x)[0]  # B
        longest_x = tf.shape(self.x)[1]   # M
        longest_y = tf.shape(self.y)[1]   # N

        # (B, M)
        x_mask = tf.cast(tf.sign(self.x), tf.float32)
        # (B, N)
        y_mask = tf.cast(tf.sign(self.y), tf.float32)
        # (B,)
        x_len = tf.reduce_sum(tf.sign(self.x), axis=1)

        # 2a. Here I define the alignment component P(A|M=m) = U(1/m)
        # (B, 1)
        lengths = tf.expand_dims(x_len, -1)
        # (B, M)
        pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))
        # (B, 1, M)
        pa_x = tf.expand_dims(pa_x, 1)
        # (B, N, M)
        pa_x = tf.tile(pa_x, [1, longest_y, 1], name='pa_x')

        # Compute encodings that will be available to variational approximations

        # Encode x into [B, M, d?] tensors
        if self.q.z_view == 0:
            h_for_z = x_embedded
        else:
            h_for_z = self.q.get_encoder(self.q.z_view)(inputs=x_embedded, lengths=x_len)
        if self.q.ds is not None:
            if self.q.s_view == 0:
                h_for_s = x_embedded
            elif self.q.s_view == self.q.z_view:
                h_for_s = h_for_z
            else:
                h_for_s = self.q.get_encoder(self.q.s_view)(inputs=x_embedded, lengths=x_len)
        else:
            h_for_s = None  # we are not predicting a distribution over sentence embeddings

        if self.q.ds is not None:  # Predict parameters of q(S)
            logging.info('Using latent sentence representation s')

            # i) Mask invalid positions
            # [B, M, 1]
            weighted_mask = tf.expand_dims(
                # get a float mask [B, M] and normalise it by length
                tf.sequence_mask(x_len, maxlen=longest_x, dtype=tf.float32) / tf.expand_dims(tf.cast(x_len, tf.float32), 1),
                2
            )
            h_for_s *= weighted_mask
            # [B, 1, dh]
            h_for_s = tf.reduce_sum(h_for_s, axis=1, keep_dims=True)

            # ii) Predict mean
            # [B, 1, ds]
            s_mean = FeedForwardEncoder(
                num_units=self.q.ds,
                hidden_layers=self.q.s_mean_hidden_layers,
                name='FF-S-Mean'
            )(h_for_s)
            # [B, ds]
            s_mean = tf.reshape(s_mean, [batch_size, self.q.ds], name='s-mean')

            # iii) Predict log-variance
            # [B, 1, ds]
            s_log_var = FeedForwardEncoder(
                num_units=self.q.ds,
                hidden_layers=self.q.s_var_hidden_layers,
                name='FF-S-LogVar'
            )(h_for_s)
            # [B, ds]
            s_log_var = tf.reshape(s_log_var, [batch_size, self.q.ds], name='s-var')

            # iv) Sample s
            # [B, ds]
            s_epsilon = tf.random_normal(tf.shape(s_mean), name='s-epsilon')
            # [B, ds]
            s = s_mean + tf.exp(s_log_var / 2.) * s_epsilon

            # Here we use the predicted mean for decoding at prediction (validation/test) time
            #  but stick with sampled encodings for training
            # [B, ds]
            s = tf.cond(self.training_phase, true_fn=lambda: s, false_fn=lambda: s_mean, name='S')

            # Replicate the sample per x-token
            # [B, 1, ds]
            s = tf.expand_dims(s, 1)
            # [B, M, ds]
            s = tf.tile(s, [1, longest_x, 1])

            if self.q.hierarchical:  # enrich features for Z prediction with s sample
                logging.info('Using hierarchical approximation q(s, z_1^m) = q(s) \prod_i q(z_i|s)')
                # [B, M, dh + ds]
                h_for_z = tf.concat([h_for_z, s], axis=-1)

        logging.info('Using latent word representation z')
        # Predict parameters of q(Z_1^m)
        # [B, M, dz]
        z_mean = FeedForwardEncoder(
            num_units=self.q.dz,
            hidden_layers=self.q.z_mean_hidden_layers,
            name='FF-Z-Mean'
        )(h_for_z)
        # [B * M, dz]
        z_mean = tf.reshape(z_mean, [-1, self.q.dz], name='z-mean')
        # [B, M, dz]
        z_log_var = FeedForwardEncoder(
            num_units=self.q.dz,
            hidden_layers=self.q.z_var_hidden_layers,
            name='FF-Z-LogVar'
        )(h_for_z)
        # [B * M, dz]
        z_log_var = tf.reshape(z_log_var, [-1, self.q.dz], name='z-var')

        # Get a sample by using the transformation
        # z = \mu(x) + \epsilon \sigma(x)
        #   where \epsilon ~ N(0, I)
        #   and \sigma(x) = \exp( 0.5 * log_var)
        # [B * M, dz]
        epsilon = tf.random_normal(tf.shape(z_log_var), name='z-epsilon')
        # [B * M, dz]
        z = z_mean + tf.exp(z_log_var / 2.) * epsilon

        # Here we use the predicted mean for decoding at prediction (validation/test) time
        #  but stick with sampled encodings for training
        # [B * M, dz]
        z = tf.cond(self.training_phase, true_fn=lambda: z, false_fn=lambda: z_mean)

        # [B, M, dz]
        z = tf.reshape(z, [batch_size, longest_x, self.q.dz], name='Z')

        # Decide on the latent encoding for predicting x tokens
        if not self._attention_z:
            latent_dim_px = self.q.dz
            # [B, M, dl=dz]
            h_for_px = z
        else:  # here we use a self attention mechanism
            logging.info('Self-attention mechanism over z_1^m')
            # [B, M, dz]
            z_keys = FeedForwardEncoder(
                num_units=self.q.dz,
                hidden_layers=[],
                name='FF-Z-Key'
            )(z)
            z_values = z
            # [B, M, M]
            scores = tf.matmul(
                z_keys,  # [B, M, dz]
                z_keys,  # [B, M, dz]
                transpose_b=True
            )
            # mask invalid logits
            scores = tf.where(
                # make the boolean mask [B, M, M]
                condition=tf.tile(
                    # make the boolean mask [B, 1, M]
                    tf.expand_dims(
                        # get a boolean mask [B, M]
                        tf.sequence_mask(x_len, maxlen=longest_x),
                        1
                    ),
                    [1, longest_x, 1]
                ),
                x=scores,
                y=tf.ones(shape=[batch_size, longest_x, longest_x]) * float('-inf')
            )
            # mask diagonal
            scores += tf.diag(tf.fill([tf.shape(scores)[-1]], float('-inf')))
            # Normalise attention
            # [B, M, M]
            attention = tf.nn.softmax(scores)
            # [B, M, dz]
            c = tf.matmul(
                attention,  # [B, M, M]
                z_values    # [B, M, dz]
            )
            latent_dim_px = self.q.dz * 2
            # [B, M, dl=2*dz]
            h_for_px = tf.concat([c, z], axis=-1)

        if self.q.ds is not None:
            logging.info('X conditions on z and s: P(X|z,s)')
            latent_dim_px += self.q.ds
            # [B, M, dl]
            h_for_px = tf.concat([s, h_for_px], axis=-1)

        # Now we implement the generative components P_\theta(X_1^m|Z_1^m=z_1^m) and P_\theta(Y_1^n|X=x_1^m, Z=z_1^m)

        # X decoder (given s and z_1^m)
        with tf.variable_scope('logit-x'):
            # [B * M, Vx|S], [B * M]
            logits_x, targets_x = logit_layer_for_text(
                nb_classes=self.vx,
                inputs=h_for_px,  # [B, M, dl]
                labels=self.x,  # [B, M]
                dim=latent_dim_px,
                nb_softmax_samples=self.nb_softmax_samples[0],
                is_training=self.training_phase,
                approximation=self.softmax_approximation[0],
                support=self.support_x,
                importance=self.importance_x,
                name='P(X|z)' if self.q.ds is None else 'P(X|z,s)'
            )

        # Apply a softmax to obtain distributions that can be used for sampling outside this class
        # (B, M, Vx)
        px_z = tf.reshape(tf.nn.softmax(logits_x), [batch_size, longest_x, -1])

        if self._improved_features_for_py:
            logging.info('Y conditions on everything available: P(Y_j|z_1^m, s, a_j)')
            latent_dim_py = latent_dim_px
            h_for_py = h_for_px
        else:
            logging.info('Y conditions on z_a alone: P(Y_j|z_aj)')
            latent_dim_py = self.q.dz
            h_for_py = z


        # Y decoder (given z_1^m)
        with tf.variable_scope('logit-y'):
            # [B * M, Vy|S], [B * N]
            logits_y, targets_y = logit_layer_for_bitext(
                nb_classes=self.vy,
                inputs=h_for_py,  # [B, M, dl|dz]
                outputs=self.y,  # [B, N]
                dim=latent_dim_py,  # dl|dz
                nb_softmax_samples=self.nb_softmax_samples[1],
                is_training=self.training_phase,
                approximation=self.softmax_approximation[1],
                support=self.support_y,
                importance=self.importance_y,
                name='P(Y|z_a)'
            )

        # Apply a softmax to obtain distributions that can be used for marginalisation
        # [B, M, Vy|S]
        py_xza = tf.reshape(tf.nn.softmax(logits_y), [batch_size, longest_x, -1])

        # 2.c Marginalise alignments

        # P(y|x_1^m,z_1^m) = \sum_a P(A=a|M=m) P(Y=y|X=x_a,Z=z_a)
        # [B, N, Vy]
        py_zx = tf.matmul(
            pa_x,  # [B, N, M]
            py_xza,   # [B, M, Vy]
        )

        # 3. Compute loss

        # 3a. Negative log-likelihood P(X=x|Z=z)
        # [B * M]
        ce_x = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_x,  # [B * M]
            logits=logits_x  # [B * M, Vx|S]
        )

        # Mask invalid positions
        # [B, M]
        ce_x = tf.reshape(ce_x, [batch_size, longest_x])
        # Sum along time dimension
        # [B]
        ce_x = tf.reduce_sum(ce_x * x_mask, axis=1)
        # and average along the sample dimension
        # []
        ce_x = tf.reduce_mean(ce_x, axis=0)

        # 3b. Compute negative log-likelihood \sum_j P(Y_j=y_j|Z_1^m=z_1^m,X_1^m=x_1^m)
        # [B * N]
        ce_y = tf.nn.sparse_softmax_cross_entropy_with_logits(
            # [B * N]
            labels=targets_y,
            # [B * N, Vy|S]
            logits=tf.log(  # tf expects logits of the marginal
                tf.reshape(py_zx, [batch_size * longest_y, -1])  # we collapse sample (B) and time (N) dimensions
            )
        )
        # [B, N]
        ce_y = tf.reshape(ce_y, [batch_size, longest_y])
        # Sum along time dimension
        # [B]
        ce_y = tf.reduce_sum(ce_y * y_mask, axis=1)
        # Average along sample dimension
        # []
        ce_y = tf.reduce_mean(ce_y)

        # 3c. Compute KL terms

        if self.q.ds is None:  # no sentence embedding
            logging.info('Gaussian prior P(Z) = N(0,I)')
            # then we have a standard prior for P(Z)
            # and compute KL(q(Z|x) || N(0,I))
            # [B * M]
            kl_z = kl_from_q_to_standard_normal(z_mean, z_log_var)
            kl_s = tf.zeros(shape=[])

        else:  # with sentence embedding
            logging.info('Gaussian prior P(S) = N(0,I)')
            # We have a standard prior for P(S)
            #  and compute KL(q(S)||N(0,I))
            # [B]
            kl_s = kl_from_q_to_standard_normal(s_mean, s_log_var)
            # []
            kl_s = tf.reduce_mean(kl_s)

            # We have a simple prior for P(Z|s) = N(Ws+b, kappa I)
            logging.info('Gaussian prior P(Z|s) = N(linear(s), sigma^2 I)')

            # TODO: consider options for prior P(Z|s)
            #  1. mean: zero, *linear_fn(s), nonlinear_fn(s)
            #  2. var: one, *kappa, linear_fn(s), nonlinear_fn(s)
            # [B * M, dz]
            prior_z_mean = ffnn(
                inputs=tf.reshape(s, [batch_size * longest_x, self.q.ds]),  # [B * M, ds]
                layers=[[self.q.dz, None]]  # TODO: prior_z_mean_layers
            )
            # [B * M, dz]
            prior_z_log_var = ffnn(
                # TODO: prior var is function of s or function of ones?
                inputs=tf.reshape(s, [batch_size * longest_x, self.q.ds]),  # [B * M, ds]
                layers=[[self.q.dz, None]]  # TODO: prior_z_var_layers
            )

            # Here we condition on the MC sample (s) to estimate KL(q(Z|s)||P(Z|s))
            # then we compute \sum_{i=1}^m KL(q(Z_i|s)||P(Z_i|s))
            # [B * M]
            kl_z = kl_diagonal_gaussians(z_mean, z_log_var, prior_z_mean, prior_z_log_var)

            if False:  # TODO: for q(S)q(Z_1^m) with a linear prior, perhaps we can condition on q(S)'s mean

                # [B, dz]
                prior_z_mean = ffnn(
                    inputs=s_mean,  # [B, ds]
                    layers=[[self.q.dz, None]]  # a single linear layer
                )

                # [B, dz]
                prior_z_log_var = ffnn(
                    inputs=tf.ones_like(s_mean),  # [B, ds]
                    layers=[[self.q.dz, None]]  # a single linear layer
                )

                # [B, 1, dz]
                prior_z_mean = tf.expand_dims(prior_z_mean, 1)
                # [B, M, dz]
                prior_z_mean = tf.tile(prior_z_mean, [1, longest_x, 1])
                # [B * M, dz]
                prior_z_mean = tf.reshape(prior_z_mean, [batch_size * longest_x, -1])

                # [B, 1, dz]
                prior_z_log_var = tf.expand_dims(prior_z_log_var, 1)
                # [B, M, dz]
                prior_z_log_var = tf.tile(prior_z_log_var, [1, longest_x, 1])
                # [B * M, dz]
                prior_z_log_var = tf.reshape(prior_z_log_var, [batch_size * longest_x, -1])

                # TODO: double check that this KL is correct
                # then we compute \sum_{i=1}^m KL(q(Z_i|s)||P(Z_i|s))
                # [B * M]
                kl_z = kl_diagonal_gaussians(z_mean, z_log_var, prior_z_mean, prior_z_log_var)

        # Sum along time dimension (masking invalid steps)
        # [B]
        kl_z = tf.reduce_sum(
            # [B, M]
            tf.reshape(kl_z, [batch_size, -1]) * x_mask,
            axis=-1
        )
        # Average along sample dimension
        # []
        kl_z = tf.reduce_mean(kl_z)

        # 3d. Aggregate everything:
        #  Our loss is the negative ELBO
        #   ELBO = expected log likelihood - KL
        #   LOSS = -ELBO = - expected log likelihood + KL
        #        = CE + KL
        #        because CE = - expected log likelihood
        loss = ce_y + ce_x + self.alpha_z * kl_z + self.alpha_s * kl_s

        # 4. Calculate accuracy of predictions

        # 4a. with respect to x_1^m
        targets_x = tf.reshape(targets_x, [batch_size, longest_x])
        predictions_x = tf.argmax(px_z, axis=2)
        acc_x = tf.equal(predictions_x, targets_x)
        acc_x = tf.cast(acc_x, tf.float32) * x_mask
        # []
        acc_x_correct = tf.reduce_sum(acc_x)
        acc_x_total = tf.reduce_sum(x_mask)
        acc_x = acc_x_correct / acc_x_total

        # 4b. with respect to y_1^n
        # [B, N]
        targets_y = tf.reshape(targets_y, [batch_size, longest_y])
        predictions_y = tf.argmax(py_zx, axis=2)
        acc_y = tf.equal(predictions_y, targets_y)
        acc_y = tf.cast(acc_y, tf.float32) * y_mask
        # []
        acc_y_correct = tf.reduce_sum(acc_y)
        acc_y_total = tf.reduce_sum(y_mask)
        acc_y = acc_y_correct / acc_y_total

        # These quantities are useful for optimisation, decoding, logging.
        # [B, M]
        self.pa_x = tf.identity(pa_x, name='pa_x')
        # [B, M, Vx]
        self.px_z = tf.identity(px_z, name='px_z')
        # [B, M, Vy]
        self.py_xza = tf.identity(py_xza, name='py_xza')
        # []
        self.loss = tf.identity(loss, name='loss')
        self.ce_x = tf.identity(ce_x, name='ce_x')
        self.ce_y = tf.identity(ce_y, name='ce_y')
        self.kl_z = tf.identity(kl_z, name='kl_z')
        self.kl_s = tf.identity(kl_s, name='kl_s')
        self.kl = tf.identity(kl_s + kl_z, name='kl')
        self.predictions_x = tf.identity(predictions_x, name='predictions_x')
        self.accuracy_x = tf.identity(acc_x, name='acc_x')
        self.accuracy_x_correct = tf.identity(tf.cast(acc_x_correct, tf.int64), name='acc_x_correct')
        self.accuracy_x_total = tf.identity(tf.cast(acc_x_total, tf.int64), name='acc_x_total')
        self.predictions_y = tf.identity(predictions_y, 'predictions_y')
        self.accuracy_y = tf.identity(acc_y, name='acc_y')
        self.accuracy_y_correct = tf.identity(tf.cast(acc_y_correct, tf.int64), name='acc_y_correct')
        self.accuracy_y_total = tf.identity(tf.cast(acc_y_total, tf.int64), name='acc_y_total')


