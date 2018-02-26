"""
:Authors: - Wilker Aziz
"""
import sys

if len(sys.argv) != 7:
    raise ValueError('Usage: %s data-dir x-lang y-lang architecture working-dir gpu' % sys.argv[0])

data_dir, x_lang, y_lang, architecture, output_dir, gpu = sys.argv[1:]


def make_file_name(stem, suffix, basedir=''):
    return '%s/%s.%s' % (basedir, stem, suffix) if basedir else '%s.%s' % (stem, suffix)

import dgm4nlp.device as dev
config = dev.tf_config(gpu, allow_growth=True)

import os
import logging
from dgm4nlp.tf.embedalign.trainer import test_embedalign
from dgm4nlp.tf.embedalign.architectures import get_hparams

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

training_x = make_file_name('training', x_lang, data_dir)
training_y = make_file_name('training', y_lang, data_dir)
#val_x = make_file_name('dev', x_lang, data_dir)
#val_y = make_file_name('dev', y_lang, data_dir)
#val_naacl = make_file_name('dev', 'naacl', data_dir)
#test_x = make_file_name('test', x_lang, data_dir)
#test_y = make_file_name('test', y_lang, data_dir)
#test_naacl = make_file_name('test', 'naacl', data_dir)

print('training_x=%s' % training_x)
print('training_y=%s' % training_y)
#print('val_x=%s' % val_x)
#print('val_y=%s' % val_y)
#print('val_naacl=%s' % val_naacl)
#print('test_x=%s' % test_x)
#print('test_y=%s' % test_y)
#print('test_naacl=%s' % test_naacl)
print('architecture=%s' % architecture)
print('output=%s' % output_dir)

os.makedirs(output_dir, exist_ok=True)  # make sure base_dir exists


test_embedalign(
    training_x=training_x,
    training_y=training_y,
    val_x=None,
    val_y=None,
    val_naacl=None,
    test_x=None,
    test_y=None,
    test_naacl=None,
    output_dir=output_dir,
    config=config,
    **get_hparams(architecture)
)
