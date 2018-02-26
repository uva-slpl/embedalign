# -*- coding: utf-8 -*-
import logging
import dill
import numpy as np
import scipy.spatial as sp
from embedalign.extractor import EmbeddingExtractor
from embedalign.sample_variable import Sampler
import argparse
import tensorflow as tf
import os
import shutil
import tempfile
from datetime import datetime
from dgm4nlp.recipes import smart_ropen
import scipy.stats as st
import pickle




def sent_embedding(ckpt_path,
             graph_file,
             tokenizer_path,
             text_path,
             criterion,
             max_len=50,
             config=None):

    # restore computational graph and parameters
    #logging.info(ckpt_path)
    #logging.info(graph_file)
    embed_align = EmbeddingExtractor(
        graph_file=graph_file,
        ckpt_path=ckpt_path,
        config=config
    )

    # load tokenizer from training
    tks = dill.load(open(tokenizer_path, 'rb'))
    # tokenize input file

    #files = [f for f in os.listdir(text_path) if os.path.isfile(os.path.join(text_path, f)) ]
    for file in text_path:
        sent_vec = []
        for sent in open(file, 'r'):
            sent = [sent.strip()] #TODO look for which collumn has a sentence
            sent_tok = tks[0].to_sequences(sent)
            #if len(sent_tok[0]) > max_len:
            #    sent_vec.append(np.zeros(len(sent_tok[0])))
            #    continue
            # [1, M]
            try:
                z_sent = embed_align.get_z_embedding_batch(x_batch=sent_tok)[0]
                z_sent = z_sent[0, 1:, :]
                sent_vec.append(z_sent)
            except:
                sent_vec.append(np.zeros(len(sent_tok[0])))
            #print(sent)
        output_file = "{}.{}.vec.pickle".format(file, criterion)
        vec_file = open(output_file, 'wb')
        pickle.dump(sent_vec, vec_file)
        print('sentvec saved to:', output_file)
        print('num sent:', len(sent_vec))




    logging.info(output_file)
    return output_file




def main(ckpt_path, tokenizer_path, text_path, criterion, max_len, gpu='0'):
    """

    :param ckpt_path:
    :param tokenizer_path:
    :param lexsub_path:
    :param cand_path:
    :param output_dir:
    :param metric:
    :param gpu:
    :return:
    """
    import dgm4nlp.device as dev
    config = dev.tf_config(gpu, allow_growth=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    sent_embedding(
        ckpt_path=ckpt_path,
        graph_file='%s.meta'%(ckpt_path),
        tokenizer_path=tokenizer_path,
        text_path=text_path,
        criterion=criterion,
        max_len=max_len,
        config=config,
    )
    return

def get_argparse():
    parser = argparse.ArgumentParser(prog='SentEval')
    parser.description = 'EmbedAlign for SentEVAL'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('ckpt',
                        type=str,
                        help='Checkpoint file (without suffix)')
    parser.add_argument('tokenizer',
                        type=str,
                        help='Tokenizer saved by trainer')
    parser.add_argument('--text_path', '-t',
                        action='append',
                        help='list of files to parse')
    parser.add_argument('--criterion', '-c',
                        type=str,
                        help='name of model')
    parser.add_argument('--max_len', '-m',
                        type=int,
                        default=50,
                        help='max len of sentences')
    parser.add_argument('--gpu', default='0',
                        type=str,
                        help='GPU to run the model [-1 for CPU]')

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(
        ckpt_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        text_path=args.text_path,
        criterion=args.criterion,
        max_len=args.max_len,
        gpu=args.gpu
    )
