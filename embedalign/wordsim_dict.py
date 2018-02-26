# -*- coding: utf-8 -*-
import logging
import pickle
from collections import defaultdict, Counter
import os
import argparse
import tempfile
from datetime import datetime
import dill
import numpy as np
from functools import partial
from dgm4nlp.nlputils import Multitext
from dgm4nlp.recipes import smart_wopen

from embedalign.extractor import EmbeddingExtractor


def _prepare_test(tks, x_path, y_path, longest=[50, 50], name='test') -> [Multitext, tuple]:
    """
    Memory-map test data.

    :param tks:
    :param x_path:
    :param y_path:
    :param longest:
    :param name:
    :return:
    """

    logging.info('Memory mapping test data')
    test = Multitext([x_path, y_path],
                     tokenizers=tks,
                     shortest=None,
                     longest=longest,
                     trim=[True, True],
                     mask_dtype='float32',
                     name=name)
    logging.info(' test-samples=%d', test.nb_samples())


    return test

def extract_vec_dict(ckpt_path,
                     graph_file,
                     tokenizer_path,
                     test_x,
                     output_dir,
                     output_type='word2vec',
                     batch_size=100,
                     dz=100,
                     config=None):
    """
    
    :param ckpt_path: path to stored parameters directory
    :param graph_file: path to stored computational graph
    :param tokenizer_path: path to stored tokenizer
    :param test_x: file with sentences to be stored in dictionary with vectors from Z latent variable
    :param output_dir: path to directory to store dictionary
    :param batch_size: batch size to process input
    :param avg_vec: Average of difference vectors of each word default:False
    
    :return: 
    """
    # restore computational graph and parameters
    embed_align = EmbeddingExtractor(
                    graph_file=graph_file,
                    ckpt_path=ckpt_path,
                    config=config)

    vec_dict = defaultdict(partial(np.ndarray, dz))
    #TODO np zeros???


    # load tokenizer from training
    tks = dill.load(open(tokenizer_path, 'rb'))
    # tokenize input file
    test = _prepare_test(tks, test_x, test_x)
    generator = test.batch_iterator(
        batch_size=batch_size,
        endless=False,
        shorter_batch='trim',
        dynamic_sequence_length=True
    )
    #batch_count = 0
    vec_counter = Counter()
    for batch in generator:
        (x, x_mask), (_, _) = batch
        # process input with 1 sample from Z latent variable
        z_batch = embed_align.get_z_embedding_batch(x_batch=x, n_samples=1)

        for i, seq in enumerate(x_mask):
            for j, mask_id in enumerate(seq):
                if mask_id == 0.:
                    break
                token = tks[0].to_str(x[i][j])
                # z[0] first dimension is the number of samples of #
                # sum vectors each token
                vec_dict[token] += z_batch[0][i][j]
                vec_counter.update(token)
    # average vec by number of times each word has been seen
    for token, nb_token in vec_counter.items():
        if token in vec_dict:
            vec_dict[token] = vec_dict[token] / float(nb_token)
            #batch_count += float(nb_token)

    if output_type == 'word2vec':
        # average of vectors over each word
        with smart_wopen("{}/word_dict.vec".format(output_dir)) as vec_file:
            for key, vec in vec_dict.items():
                print('%s %s'%(key, ' '.join([str(i) for i in vec.tolist()])), file=vec_file)
            logging.info("{}/word_dict.vec".format(output_dir))
    else:
        pickle.dump(vec_dict, open("{}/vec_dict.pickle".format(output_dir), 'wb'))
        logging.info("{}/vec_dict.pickle".format(output_dir))

def main(ckpt_path, tokenizer_path, test_path, output_dir, batch_size=100, dz=100, gpu='0'):
    """

    :param ckpt_path:
    :param tokenizer_path:
    :param test_path:
    :param output_dir:
    :param batch_size:
    :param dz:
    :param gpu:
    :return:
    """
    import dgm4nlp.device as dev
    config = dev.tf_config(gpu, allow_growth=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(output_dir, exist_ok=True)  # make sure base_dir exists
    #output_dir = tempfile.mkdtemp(dir=output_dir)
    logging.info('Workspace: %s', output_dir)
    #shutil.copy(os.path.abspath(__file__), output_dir)
    extract_vec_dict(
        ckpt_path=ckpt_path,
        graph_file='%s.meta'%(ckpt_path),
        tokenizer_path=tokenizer_path,
        test_x=test_path,  # '/mnt/data/mrios/dgm4nlp/test_data/1k.e', #
        batch_size=batch_size,
        output_dir=output_dir,
        output_type='word2vec',
        dz=dz,
        config=config
    )
    return

def get_argparse():
    parser = argparse.ArgumentParser(prog='w2v')
    parser.description = 'EmbedAlign vector extractor'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('ckpt',
                        type=str,
                        help='Checkpoint file (without suffix)')
    parser.add_argument('tokenizer',
                        type=str,
                        help='Tokenizer saved by trainer')
    parser.add_argument('--test_path', '-t',
                        type=str,
                        help='test sentences for vector extraction')
    parser.add_argument('--dim_z', '-d', default=100,
                        type=int,
                        help='dimensionality of Z vector default:100')
    parser.add_argument('--output_dir', '-o',
                        type=str,
                        help='output dir for ranked list')
    parser.add_argument('--batch_size', '-b', default=100,
                        type=int,
                        help='batch-size default: 100')
    parser.add_argument('--gpu', default='0',
                        type=str,
                        help='GPU to run the model [-1 for CPU]')

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(
        ckpt_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        test_path=args.test_path,
        dz=args.dim_z,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        gpu=args.gpu
    )
