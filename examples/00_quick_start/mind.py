#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Guocheng Qian. All rights reserved.</i>

import __init__
import sys
import os
import tensorflow as tf
import logging
import argparse
import time
import uuid

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources 
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.models.nrmma import NRMMAModel

from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

from tools.utils import configure_logger, print_args


parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For ModelNet Classification')
# ----------------- Base
parser.add_argument('--model', type=str, default='nrms', metavar='N',
                    choices=['nrms', 'naml', 'npa', 'nrmma'])
parser.add_argument('--MIND_type', type=str, default='small')
parser.add_argument('--data_dir', type=str, default='/data/recsys/mind')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

epochs = args.epochs
seed = args.seed
batch_size = args.batch_size

# Options: demo, small, large
MIND_type = args.MIND_type.lower()
data_dir = args.data_dir
data_path = os.path.join(data_dir, MIND_type)

model_type = args.model
timestamp = time.strftime('%Y%m%d-%H%M%S')
exp_name = '-'.join((model_type, timestamp, str(uuid.uuid4())))
exp_path = os.path.join(data_path, model_type, exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)


configure_logger(exp_path, exp_name)
print_args(args)

logging.info("System version: {}".format(sys.version))
logging.info("Tensorflow version: {}".format(tf.__version__))
# ## Download and load data
logging.info(f"dataset will be put in: {data_path} \n")

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = os.path.join(data_path, "utils", f'{model_type}.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/',
                               os.path.join(data_path, 'utils'), mind_utils)


# ## Create hyper-parameters
logging.info(f"create hyper-parameters by loading the file {yaml_file}")
hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          vertDict_file=vertDict_file,
                          subvertDict_file=subvertDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
logging.info(hparams)


# ## Train the NRMS model


if model_type == 'nrms':
    iterator = MINDIterator
    model = NRMSModel(hparams, iterator, seed=seed)
elif model_type == 'naml':
    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator, seed=seed)
elif model_type == 'npa':
    iterator = MINDIterator
    model = NPAModel(hparams, iterator, seed=seed)
elif model_type == 'nrmma':
    iterator = MINDAllIterator
    model = NRMMAModel(hparams, iterator, seed=seed)

else:
    raise NotImplementedError(f"{exp_name} is not implemented")

# In[8]:
model_path = os.path.join(exp_path, model_type)
model_name = model_type + '_ckpt'
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,
          model_path=model_path, model_name=model_name)
res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
logging.info(res_syn)


# ## Reference
# \[1\] Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)<br>
# \[2\] Wu, Fangzhao, et al. "MIND: A Large-scale Dataset for News Recommendation" Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://msnews.github.io/competition.html <br>
# \[3\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/
