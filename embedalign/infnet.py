"""
:Authors: - Wilker Aziz
"""
from embedalign.model import VariationalApproximationSpecs
import tensorflow as tf
from dgm4nlp.tf.encoder import BidirectionalEncoder, MultiBidirectionalEncoder


def get_infnet(config_str, dz=100, z_dh=None, hierarchical=False, ds=50, s_dh=None, s_mulirnn_layers=2,
               dropout=False,
               dropout_params=[1.0, 1.0, 1.0, False]):
    """

    :param config_str: which type of encoder  (see code below)
    :param dz: dim for z
    :param z_dh: defaults to dz
    :param hierarchical: if True the model will have a sentence embedding s
    :param ds: dim for s
    :param s_dh: defaults to ds
    :param dropout: an empty list or [input, output, state, variational flag]
    :return:
    """
    if z_dh is None:
        z_dh = dz
    if s_dh is None:
        s_dh = ds
    if not hierarchical:
        if config_str == 'bow-z':
            q = VariationalApproximationSpecs(dz=dz)
            q.config_z(view=0, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'rnn-z':
            q = VariationalApproximationSpecs(dz=dz)
            #  add BiRNN encoders
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=z_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_z(view=1, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        else:
            raise ValueError('Unknown config: %s' % config_str)
    else:
        if config_str == 'bow-s, bow-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            q.config_s(view=0, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=0, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'bow-s, rnn-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            # RNN for Z
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=z_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_s(view=0, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=1, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'rnn-s, bow-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            # RNN for S
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=s_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_s(view=1, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=0, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'rnn-s, rnn-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            # shared RNN
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=max(z_dh, s_dh),
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_s(view=1, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=1, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'rnn1-s, rnn2-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            # shared RNN
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=s_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=z_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_s(view=1, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=2, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        elif config_str == 'multirnn-s, rnn-z':
            q = VariationalApproximationSpecs(dz=dz, ds=ds, hierarchical=True)
            # shared RNN
            q.add_encoder(MultiBidirectionalEncoder(  # TODO: dropout?
                num_units=s_dh, num_layers=s_mulirnn_layers, cell_type='lstm', merge_strategy='sum'))
            q.add_encoder(
                BidirectionalEncoder(
                    num_units=z_dh,
                    cell_type='lstm',
                    merge_strategy='sum',
                    dropout=dropout,
                    input_keep_prob=dropout_params[0],
                    output_keep_prob=dropout_params[1],
                    state_keep_prob=dropout_params[2],
                    variational_recurrent=dropout_params[3]
                )
            )
            #  associate a prediction with an encoder (possibly specialising it)
            q.config_s(view=1, mean_hidden_layers=[[s_dh, None]], var_hidden_layers=[[s_dh, tf.nn.relu]])
            q.config_z(view=2, mean_hidden_layers=[[z_dh, None]], var_hidden_layers=[[z_dh, tf.nn.relu]])
        else:
            raise ValueError('Unknown config: %s' % config_str)
    return q
