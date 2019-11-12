import getpass
import os
import numpy as np
from datetime import datetime
import torch
from transformers import *
from text_utils import TextDownload
from model_utils import AllModels
import argparse
import time
import glob

import logging

all_filenames = glob.glob('log/*')

logging.basicConfig(filename='log/logging-{}.log'.format(len(all_filenames)), level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


if getpass.getuser() == 'Mitch':
    head = 'C:/Users/Mitch/PycharmProjects'
else:
    head = '/home/kinne174/private/PythonProjects/'



def BERT_embeddings(parameter_dict):
    # returns a numpy array with the embeddings as determined by BERT

    logging.info('Starting to gather context.')
    # get list of contexts
    all_context = []
    for p in ['dev', 'test', 'train']:
        TD = TextDownload(dataset_name='ARC', partition=p, difficulty='')
        all_context.extend(TD.load())
        if getpass.getuser() == 'Mitch':
            all_context = all_context[:50]
            break
    logging.info('Finished gathering context.')

    # load a model if one has been saved
    # or if in parameter dict a specific pre-trained embedder is wanted
    if parameter_dict['pretrained_embedder'] == 'MultipleChoice':
        model_class = BertForMultipleChoice
    else:
        model_class = BertModel

    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    logging.info('Adding padding.')
    concatenated_QA_context = TD.padding(all_context, tokenizer, tokenizer.cls_token, tokenizer.sep_token, 512)
    logging.info('Finished adding padding.')

    assert isinstance(concatenated_QA_context, list)

    input_ids = [tokenizer.encode(s, add_special_tokens=False) for s in concatenated_QA_context]
    assert all([len(input_ids[0]) == len(ii) for ii in input_ids[1:]]), [len(input_ids[0]) == len(ii) for ii in input_ids[1:]].index(False)
    input_ids = torch.tensor(input_ids)

    logging.info('Starting creating sentence embeddings.')
    with torch.no_grad():
        all_hidden_states = model(input_ids)
        pooled_hidden_state = all_hidden_states[1]
        last_hidden_states = all_hidden_states[0]


    torch_array = pooled_hidden_state

    if parameter_dict['embedding_average_pooling'] == 'average':
        torch_array = torch.mean(last_hidden_states, dim=1)
    elif parameter_dict['embedding_average_pooling'] == 'pooling':
        abs_array = torch.abs(last_hidden_states)
        max_indices = torch.argmax(abs_array, dim=1)
        torch_array = torch.zeros(max_indices.size())
        for ii in range(torch_array.size()[0]):
            for jj in range(torch_array.size()[1]):
                torch_array[ii, jj] = last_hidden_states[ii, max_indices[ii, jj], jj]

    logging.info('Finished creating sentence embeddings.')

    out_array = torch_array.numpy()

    correct_labels = np.array([np.int(c.correct) for c in all_context], dtype=np.int)

    assert correct_labels.shape[0] == len(all_context)

    return out_array, correct_labels


def baseline(parameter_dict):
    '''
    main function to give a baseline for a specified combination of parameters such as sentence embedding, model, evaluation
    method etc.
    '''

    sentence_embedder = parameter_dict['sentence_embedder']

    if sentence_embedder == 'BERT':
        embeddings_df, labels = BERT_embeddings(parameter_dict)

    which_model = parameter_dict['which_model']
    AM = AllModels(embeddings_df, labels, which_model)

    logging.info('Starting to fit models.')
    AM.fit()
    logging.info('Finished fitting models.')

    logging.info('Results: {}'.format(AM.results()))

    # TODO implement this for more models, figure out how to save information and datasets
    # TODO see if pytorch/huggingface has a way to speed things up with a GPU or multiple nodes


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # # arguments
    # parser.add_argument('--dataset', type=str, default='ARC')
    # parser.add_argument('--num_folds', type=int, default=3)
    #
    # args = parser.parse_args()
    # print(args)
    # globals().update(args.__dict__)

    parameter_dict = {'sentence_embedder': 'BERT', 'pretrained_embedder': '', 'embedding_average_pooling': '',
                      'which_model': 'all'}

    logging.info('parameters: {}'.format(parameter_dict))

    baseline(parameter_dict)



# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# assert tokenizer.sep_token is not None
# assert tokenizer.pad_token is not None
# assert tokenizer.cls_token is not None
# s = tokenizer.bos_token + 'hello how are you?' + tokenizer.sep_token + tokenizer.pad_token + tokenizer.cls_token
# input_ids = torch.tensor(tokenizer.encode(s))  # Batch size 1
# outputs = model(input_ids, labels=input_ids)
# loss, logits = outputs[:2]
#
# MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,       'gpt2'),
#           (CTRLModel,       CTRLTokenizer,       'ctrl'),
#           (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
#           (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
#           (RobertaModel,    RobertaTokenizer,    'roberta-base')]
#
# # first part is to load in the question answer and contexts and create a dataset with each line an embedding and 0/1
# # depending on if the pairing is correct or not
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
#
#     # Encode text
#     encode_in = tokenizer.encode("Here is some text to encode. [SEP] There is a lot to say.", add_special_tokens=True) # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
#     decoded = tokenizer.decode(encode_in)
#     input_ids = torch.tensor([encode_in])
#     with torch.no_grad():
#         all_hidden_states = model(input_ids)
#         last_hidden_states = all_hidden_states[0]  # Models outputs are now tuples
