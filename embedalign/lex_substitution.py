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


def bhat_distance_categorical(cat_1, cat_2):
    return np.sum(np.sqrt(cat_1 * cat_2), -1)


def bhat_distance_normal(mean1, var1, mean2, var2):
    return 0.25 * np.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * (((mean1 - mean2) ** 2) / (var1 + var2))


def lex_subs(ckpt_path,
             graph_file,
             tokenizer_path,
             lexsub_path,
             cand_path,
             output_dir,
             heuristic='add',
             top=100,
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

    (keys, sids, wids, test_sents) = _load_test(lexsub_path)
    candidates = _load_cand(cand_path)
    output_file = "{}/lst.out".format(output_dir)
    rank_file  = open(output_file, 'w')
    for i, (key, sid, wid) in enumerate(zip(keys, sids, wids)):
        if key in candidates:
            cand_ctx = list()
            cand_words = list()
            # copy original sentence
            len_orig = len(test_sents[sid])
            cand_ctx.append(' '.join(test_sents[sid].copy()))
            # copy original word
            cand_words.append(test_sents[sid][wid])
            for j, cand in enumerate(candidates[key], start=1):
                if ' ' in cand:
                    # avoid MWE
                    continue
                # create sentences with subtitution in context
                tmp_orig = test_sents[sid].copy()
                tmp_orig[wid] = cand
                cand_ctx.append(' '.join(tmp_orig))
                cand_words.append(cand)

            batch_size = len(cand_ctx) # B
            batch = tks[0].to_sequences(cand_ctx)
            batch = [i[1:] for i in batch] # quit null word in X from tokenizer
            # [Samples, B, M, dz]
            # [B, M, dz]
            z_batch = embed_align.get_z_embedding_batch(x_batch=batch, n_samples=1)[0]
            #tok_space_wid = wid + 1
            t = z_batch[0, wid]

            ctx = np.concatenate((z_batch[0, :wid], z_batch[0, wid+1:len_orig]), axis=0)
            sub_dict = dict()
            for k, _ in enumerate(range(batch_size-1), start=1):
                #print(cand_ctx[k])
                s = z_batch[k, wid]
                ctx_size, _ = ctx.shape  # M or \C\
                #ctx_size -= 1 # quit NULL word from tokenizer
                if heuristic == 'add':
                    # TODO check for on ctx!!!
                    # TODO use ctx of t or s default t?
                    score = (cos(s, t) + np.sum([cos(s, c) for c in ctx])) / float(ctx_size + 1)
                elif heuristic == 'mult':
                    score = np.float_power((pcos(s,t) * np.prod([pcos(s, c) for c in ctx])), 1./float(ctx_size + 1))
                else:
                    score = cos(s,t)
                sub_dict[cand_words[k]] = score
                #print(cand_words[0], cand_words[k], sub_dict[cand_words[k]])
            cand_rank = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
            cand_rank = ['%s %0.4f' % (i, j) for i, j in cand_rank]
            print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank[:top])), file=rank_file)
    logging.info(output_file)
    return output_file

def _kl_np(params_i: list, params_j: list):
        location_i, scale_i = params_i  # [mean, std]
        location_j, scale_j = params_j  # [mean, std]
        var_i = scale_i ** 2
        var_j = scale_j ** 2
        term1 = 1 / (2 * var_j) * ((location_i - location_j) ** 2 + var_i - var_j)
        term2 = np.log(scale_j) - np.log(scale_i)
        return term1 + term2


def _cat_kl_np(p1, p2):
    return np.sum(np.transpose(p1) * (np.log(p1) - np.log(p2)))

#def _get_py_z(z_batch, vy):
    # TODO!
    #py_z = np.zeros([1,1,1])
    #return py_z

def lex_subs_KL(ckpt_path,
             graph_file,
             tokenizer_path,
             lexsub_path,
             cand_path,
             output_dir,
             top=100,
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

    (keys, sids, wids, test_sents) = _load_test(lexsub_path)
    candidates = _load_cand(cand_path)
    output_file = "{}/lst.out".format(output_dir)
    rank_file  = open(output_file, 'w')
    #sess_normal = tf.Session(config=config)
    #normal = Normal()
    for i, (key, sid, wid) in enumerate(zip(keys, sids, wids)):
        if key in candidates:
            cand_ctx = list()
            cand_words = list()
            # copy original sentence
            len_orig = len(test_sents[sid])
            cand_ctx.append(' '.join(test_sents[sid].copy()))
            # copy original word
            cand_words.append(test_sents[sid][wid])
            for j, cand in enumerate(candidates[key], start=1):
                if ' ' in cand:
                    # avoid MWE
                    continue
                # create sentences with subtitution in context
                tmp_orig = test_sents[sid].copy()
                tmp_orig[wid] = cand
                cand_ctx.append(' '.join(tmp_orig))
                cand_words.append(cand)

            batch = tks[0].to_sequences(cand_ctx)
            #batch = [i[1:] for i in batch] # quit null word in X from tokenizer
            B = len(batch)  # B
            M = len(batch[0])
            # [Samples, B, M, dz]
            # [B, M, dz]
            embed_batch_mean, embed_batch_logvar = embed_align.get_mean_std_batch(x_batch=batch, n_samples=1)[0]
            #tok_space_wid = wid + 1
            (_, dz) = embed_batch_mean.shape
            embed_batch_mean = np.reshape(embed_batch_mean, (B, M, dz))
            embed_batch_logvar = np.reshape(embed_batch_logvar, (B, M, dz))
            shifted_wid = wid + 1
            t_mean, t_logvar = embed_batch_mean[0][shifted_wid], embed_batch_logvar[0][shifted_wid]
            t_std = np.exp(t_logvar / 2.)
            #ctx = np.concatenate((z_batch[0, :wid], z_batch[0, wid+1:len_orig]), axis=0)
            sub_dict = dict()
            for k, _ in enumerate(range(B - 1), start=1):
                #print(cand_ctx[k])
                s_mean, s_logvar = embed_batch_mean[k][shifted_wid], embed_batch_logvar[k][shifted_wid]
                s_std = np.exp(s_logvar / 2.)
                #ctx_size -= 1 # quit NULL word from tokenizer
                kl = _kl_np([t_mean, t_std], [s_mean, s_std])#sess_normal.run(normal.kl([t_mean, t_std], [s_mean, s_std]))
                #print(kl.shape)
                #print(kl)
                sub_dict[cand_words[k]] = 1. - np.sum(kl)
                #print(cand_words[0], cand_words[k], sub_dict[cand_words[k]])
            cand_rank = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
            cand_rank = ['%s %0.4f' % (i, j) for i, j in cand_rank]
            print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank[:top])), file=rank_file)

    logging.info(output_file)
    return output_file

def lex_subs_SamplecompKL(ckpt_path,
             graph_file,
             tokenizer_path,
             lexsub_path,
             cand_path,
             output_dir,
             top=100,
             alpha=1.,
             n_samples=20,
             config=None, eps=0):

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
    vy = tks[1].vocab_size()
    # tokenize input file

    (keys, sids, wids, test_sents) = _load_test(lexsub_path)
    candidates = _load_cand(cand_path)
    output_file = "{}/lst.out".format(output_dir)
    rank_file  = open(output_file, 'w')
    #sess_normal = tf.Session(config=config)
    #normal = Normal()
    for i, (key, sid, wid) in enumerate(zip(keys, sids, wids)):
        if key not in candidates:
            continue
        cand_ctx = list()
        cand_words = list()
        # copy original sentence
        len_orig = len(test_sents[sid])
        cand_ctx.append(' '.join(test_sents[sid].copy()))
        # copy original word
        cand_words.append(test_sents[sid][wid])
        for j, cand in enumerate(candidates[key], start=1):
            if ' ' in cand:  # TODO: if MWE, we can introduce a simple heuristic, namely, the sum of z's
                # avoid MWE
                continue
            # create sentences with subtitution in context
            tmp_orig = test_sents[sid].copy()
            tmp_orig[wid] = cand
            cand_ctx.append(' '.join(tmp_orig))
            cand_words.append(cand)

        batch = tks[0].to_sequences(cand_ctx)
        # TODO: this may affect LSTM encodings
        #batch = [i[1:] for i in batch] # quit null word in X from tokenizer
        B = len(batch)  # B
        M = len(batch[0])
        # [Samples, B, M, dz]
        # array of n_samples of [B, M, dz]
        embed_batch_mean, embed_batch_logvar = embed_align.get_mean_std_batch(x_batch=batch, n_samples=1)[0]
        #z_batch = embed_align.get_z_embedding_batch(x_batch=batch, n_samples=1)[0]
        # [B, M, Vy]
        # TODO it needs Y to compute Cat(softmax(z))
        #embed_batch_mean, embed_batch_logvar, py_xza_batch = embed_align.get_mustd_py_xza(
        #    x_batch=batch,
        #    vy=vy,
        #    n_samples=1)[0]

        #TODO for n samples

        #tok_space_wid = wid + 1
        (_, dz) = embed_batch_mean.shape
        embed_batch_mean = np.reshape(embed_batch_mean, (B, M, dz))
        embed_batch_logvar = np.reshape(embed_batch_logvar, (B, M, dz))
        #t_mean, t_logvar = embed_batch_mean[0][wid], embed_batch_logvar[0][wid]

        # [B, Vy]
        posterior_params = np.zeros([B, vy])
        for s in range(n_samples):
            # [B, M, Vy]
            py_xza_batch = np.array(embed_align.get_py_xza(
                x_batch=batch,
                vy=vy,
                n_samples=1))[0]
            posterior_params += py_xza_batch[:, wid + 1, :]  # wid is shiffted because tf adds a BOS symbol
        posterior_params /= n_samples
        posterior_params += eps

        #py_xza_batch = np.array(embed_align.get_py_xza(
        #    x_batch=batch,
        #    vy=vy,
        #    n_samples=n_samples))[:, :, wid, :]
        #posterior_params = np.mean(py_xza_batch, axis=0) + eps
        target_params = posterior_params[0]
        kls = np.zeros(B)
        for c in range(B):
            cand_params = posterior_params[c]
            # KL[theta_t || theta_c ]
            kl = st.entropy(target_params, cand_params)
            kls[c] = kl
        # we exclude the target word itself from the ranking
        ranking = sorted(zip(cand_words[1:], kls[1:]), key=lambda pair: pair[1])
        cand_rank = ['%s %.4f' % (word, kl) for word, kl in ranking]
        print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank[:top])), file=rank_file)
        print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank[:top])))

    logging.info(output_file)
    return output_file


def lex_subs_compKL(ckpt_path,
             graph_file,
             tokenizer_path,
             lexsub_path,
             cand_path,
             output_dir,
             top=100,
             alpha=1.,
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
    vy = tks[1].vocab_size()
    # tokenize input file

    (keys, sids, wids, test_sents) = _load_test(lexsub_path)
    candidates = _load_cand(cand_path)
    output_file = "{}/lst.out".format(output_dir)
    rank_file  = open(output_file, 'w')
    #sess_normal = tf.Session(config=config)
    #normal = Normal()
    for i, (key, sid, wid) in enumerate(zip(keys, sids, wids)):
        if key in candidates:
            cand_ctx = list()
            cand_words = list()
            # copy original sentence
            len_orig = len(test_sents[sid])
            cand_ctx.append(' '.join(test_sents[sid].copy()))
            # copy original word
            cand_words.append(test_sents[sid][wid])
            for j, cand in enumerate(candidates[key], start=1):
                if ' ' in cand:
                    # avoid MWE
                    continue
                # create sentences with subtitution in context
                tmp_orig = test_sents[sid].copy()
                tmp_orig[wid] = cand
                cand_ctx.append(' '.join(tmp_orig))
                cand_words.append(cand)

            batch = tks[0].to_sequences(cand_ctx)
            batch = [i[1:] for i in batch] # quit null word in X from tokenizer
            B = len(batch)  # B
            M = len(batch[0])
            # [Samples, B, M, dz]
            # [B, M, dz]
            #embed_batch_mean, embed_batch_logvar = embed_align.get_mean_std_batch(x_batch=batch, n_samples=1)[0]
            #z_batch = embed_align.get_z_embedding_batch(x_batch=batch, n_samples=1)[0]
            # [B, M, Vy]
            # TODO it needs Y to compute Cat(softmax(z))
            embed_batch_mean, embed_batch_logvar, py_xza_batch = embed_align.get_mustd_py_xza(
                x_batch=batch,
                vy=vy,
                n_samples=1)[0]

            #tok_space_wid = wid + 1
            (_, dz) = embed_batch_mean.shape
            embed_batch_mean = np.reshape(embed_batch_mean, (B, M, dz))
            embed_batch_logvar = np.reshape(embed_batch_logvar, (B, M, dz))
            t_mean, t_logvar = embed_batch_mean[0][wid], embed_batch_logvar[0][wid]
             #Vy
            t_std = np.exp(t_logvar / 2.)
            t_p1 = py_xza_batch[0][wid]
            #ctx = np.concatenate((z_batch[0, :wid], z_batch[0, wid+1:len_orig]), axis=0)
            sub_dict = dict()
            for k, _ in enumerate(range(B - 1), start=1):
                #print(cand_ctx[k])
                s_mean, s_logvar = embed_batch_mean[k][wid], embed_batch_logvar[k][wid]

                s_std = np.exp(s_logvar / 2.)
                #ctx_size -= 1 # quit NULL word from tokenizer
                # [dz]
                kl_z = _kl_np([t_mean, t_std], [s_mean, s_std])#sess_normal.run(normal.kl([t_mean, t_std], [s_mean, s_std]))
                # [Vy]
                s_p2 = py_xza_batch[k][wid]
                kl_cat = _cat_kl_np(t_p1, s_p2)

                kl = (alpha * np.sum(kl_z)) + ((1. - alpha) * np.sum(kl_cat)) # todo add coef aplha
                #kl = np.sum(kl_z) + np.sum(kl_cat)
                #print(kl.shape)
                sub_dict[cand_words[k]] = 1. - kl
                #print(cand_words[0], cand_words[k], sub_dict[cand_words[k]])
            cand_rank = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
            cand_rank = ['%s %0.4f' % (i, j) for i, j in cand_rank]
            print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank[:top])), file=rank_file)

    logging.info(output_file)
    return output_file


def lex_subs_ranker(ckpt_path,
                    graph_file,
                    tokenizer_path,
                    lexsub_path,
                    cand_path,
                    output_dir,
                    sample_type='stochastic',
                    ranking='cand-list', #'full-vocab'
                    n_samples=10,
                    top=10,
                    config=None):

    # restore computational graph and parameters
    #logging.info(ckpt_path)
    #logging.info(graph_file)
    p = Sampler(
        graph_file=graph_file,
        ckpt_path=ckpt_path,
        config=config
    )

    # load tokenizer from training
    tks = dill.load(open(tokenizer_path, 'rb'))
    vx = tks[0].vocab_size()
    unk = tks[0]._vocab['-UNK-']
    # tokenize input file

    (keys, sids, wids, test_sents) = _load_test(lexsub_path)
    candidates = _load_cand(cand_path)
    output_file = "{}/lst.out".format(output_dir)
    rank_file  = open(output_file, 'w')
    for i, (key, sid, wid) in enumerate(zip(keys, sids, wids)):
        if key in candidates:
            orig_ctx = list()

            # copy original sentence
            len_orig = len(test_sents[sid])
            orig_ctx.append(' '.join(test_sents[sid]))
            # copy original word
            cand_words = candidates[key]
            cand_ids = [tks[0]._vocab.get(w, unk) for w in cand_words] # TODO!!! is it correct to call _vocab??

            x = tks[0].to_sequences(orig_ctx)
            x = [x[0][1:]] #quit NULL from 0 position

            # [Samples, B, M, dz]
            # [B, M, dz]
            if sample_type == 'stochastic':
                p_xz = p.sample_xz(x_batch=x, vx=vx, sample_type='stochastic', n_samples=n_samples)
                p_xz = np.mean(p_xz, axis=0)
            else:
                p_xz = p.sample_xz(x_batch=x, vx=vx, sample_type='discrete')[0]

            #tok_space_wid = wid + 1
            sub_dict = dict()
            if ranking == 'cand-list': # filters candidates from vocab in softmax
                for idx in cand_ids:
                    try:
                        sub_dict[idx] = p_xz[0, wid, idx] #TODO if index exists???
                    except:
                        continue
            elif ranking == 'full-vocab':
                keys = [i for i in range(vx)]
                sub_dict = dict(zip(keys, p_xz[0, wid]))

            cand_rank = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
            cand_rank = ['%s %0.4f' % (tks[0].to_str(i), j) for i, j in cand_rank[:top]]
            print('RANKED\t%s %s\t%s' % (key, sid, '\t'.join(cand_rank)), file=rank_file)
    logging.info(output_file)
    return

def cos(a, b):
    return 1. - sp.distance.cosine(a, b)

def pcos(a, b):
    return (cos(a,b) + 1.) / 2.


def _load_test(test_file):
    keys = []
    sids = []
    wids = []
    test_sent = dict()

    with smart_ropen(test_file) as testf:
        for line in testf:
            (key, sid, wid, sent) = line.strip().split('\t')
            keys.append(key)
            sids.append(int(sid))
            wids.append(int(wid))
            test_sent[int(sid)] = sent.split(' ')
    return keys, sids, wids, test_sent


def _load_cand(cand_file):
    candidates = dict()
    with smart_ropen(cand_file) as cf:
        for line in cf:
            (key, cand) = line.strip().split('::')
            #(word, _) = key.split('.')
            cols = cand.split(';')
            candidates[key] = cols
    return candidates

def main(ckpt_path, tokenizer_path, lexsub_path, cand_path, output_dir, metric='kl', gpu='0'):
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
    os.makedirs(output_dir, exist_ok=True)  # make sure base_dir exists
    #output_dir = tempfile.mkdtemp(dir=output_dir)
    logging.info('Workspace: %s', output_dir)
    #shutil.copy(os.path.abspath(__file__), output_dir)
    if metric == 'kl':
        _ = lex_subs_KL(
            ckpt_path=ckpt_path,
            graph_file='%s.meta'%(ckpt_path),
            tokenizer_path=tokenizer_path,
            lexsub_path=lexsub_path,
            cand_path=cand_path,
            output_dir=output_dir,
            config=config,
        )
    elif metric == 'cos':
        _ = lex_subs(
            ckpt_path=ckpt_path,
            graph_file='%s.meta'%(ckpt_path),
            tokenizer_path=tokenizer_path,
            lexsub_path=lexsub_path,
            cand_path=cand_path,
            output_dir=output_dir,
            heuristic='cos',  # add, mult, cos
            config=config,
        )
    return

def get_argparse():
    parser = argparse.ArgumentParser(prog='lexsub')
    parser.description = 'EmbedAlign lexical susbtitution task'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('ckpt',
                        type=str,
                        help='Checkpoint file (without suffix)')
    parser.add_argument('tokenizer',
                        type=str,
                        help='Tokenizer saved by trainer')
    parser.add_argument('--lexsub_path', '-l',
                        type=str,
                        help='lex sub path for test set')
    parser.add_argument('--cand_path', '-c',
                        type=str,
                        help='cand path for test set')
    parser.add_argument('--output_dir', '-d',
                        type=str,
                        help='output dir for ranked list')
    parser.add_argument('--metric', '-m', default='kl',
                        type=str,
                        help='metric to rank, default: kl')
    parser.add_argument('--gpu', default='0',
                        type=str,
                        help='GPU to run the model [-1 for CPU]')

    return parser


if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(
        ckpt_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        lexsub_path=args.lexsub_path,
        cand_path=args.cand_path,
        output_dir=args.output_dir,
        metric=args.metric,
        gpu=args.gpu
    )
