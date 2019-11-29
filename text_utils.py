import json_lines
from collections import namedtuple
import os
import getpass
import torch
import tqdm
from transformers import (BertTokenizer, BertModel, BertForMultipleChoice, TransfoXLTokenizer, TransfoXLModel, XLNetModel,
                          XLNetTokenizer, RobertaTokenizer, RobertaModel, DistilBertModel, DistilBertTokenizer)
from torchsummary import summary


if getpass.getuser() == 'Mitch':
    head = 'C:/Users/Mitch/PycharmProjects'
else:
    head = '/home/kinne174/private/PythonProjects/'

def load(partition):
    filename = os.path.join(head, 'ARC/ARC-with-context/{}.jsonl'.format(partition))

    all_context = []

    with open(filename, 'r') as file:
        jsonl_reader = json_lines.reader(file)
        for line in jsonl_reader:
            all_context.extend(collect(line))

    return all_context

def collect(info):
    CONTEXT_DOCUMENT = namedtuple('CONTEXT_DOCUMENT', 'id question_text choice_text choice_label context correct')

    current_context = []

    id = info['id']
    question_text = info['question']['stem']

    for c in info['question']['choices']:
        choice_label = c['label']
        choice_text = c['text']
        context = c['para']
        correct = choice_label == info['answerKey']

        current_context.append(CONTEXT_DOCUMENT(id=id, question_text=question_text, choice_text=choice_text,
                                                choice_label=choice_label, context=context, correct=correct))
    return current_context

def padding(all_context, tokenizer, cls_token, sep_token, max_allowed_len):

    cls_token = '' if cls_token is None else cls_token
    sep_token = '' if sep_token is None else sep_token

    all_combined = [cls_token + ' ' + c.question_text + ' ' + c.choice_text + ' ' + sep_token + ' ' + c.context
                    for c in all_context]
    all_tokenized_sentences = [tokenizer.encode(s) for s in all_combined]

    pad_id = 0

    all_len = [len(t) for t in all_tokenized_sentences]

    max_len = max(all_len)
    max_len = max_len if max_len < max_allowed_len else max_allowed_len

    out_ids = []
    for i, t in enumerate(all_tokenized_sentences):
        current_len = all_len[i]
        num_padding_tokens = max_len - current_len

        if num_padding_tokens < 0:
            new_id = t[0:max_len]
        else:
            new_id = t + [pad_id]*num_padding_tokens
        assert len(new_id) == max_len

        out_ids.append(new_id)

    return out_ids

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


if __name__ == '__main__':

    import logging
    import glob

    all_filenames = glob.glob(os.path.join(head, 'hf_transformers/log/*'))

    logging.basicConfig(filename=os.path.join(head, 'hf_transformers/log/logging-{}.log'.format(len(all_filenames))),
                        level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info('CUDA available: {}'.format(torch.cuda.is_available()))
    # assert getpass.getuser() == 'Mitch' or torch.cuda.is_available()

    CLASSES = {#'TransfoXLModel': (TransfoXLTokenizer, TransfoXLModel, 'transfo-xl-wt103'),
               'RobertaModel': (RobertaTokenizer, RobertaModel, 'roberta-base'),
               'RobertaLargeMNLI': (RobertaTokenizer, RobertaModel, 'roberta-large-mnli'),
               'XLNetModel': (XLNetTokenizer, XLNetModel, 'xlnet-base-cased'),
               'DistilBertModel': (DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased'),
               'BertModel': (BertTokenizer, BertModel, 'bert-base-uncased'),
               'BertModelSquad': (BertTokenizer, BertModel, 'bert-large-uncased-whole-word-masking-finetuned-squad'),
               'DistilBertModelSquad': (DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased-distilled-squad')}

    # tokenizer_class_dict = {'BertTokenizer': BertTokenizer, 'TransfoXLTokenizer': TransfoXLTokenizer,
    #                         'RobertaTokenizer': RobertaTokenizer, 'XLNetTokenizer': XLNetTokenizer,
    #                         'DistilBertTokenizer': DistilBertTokenizer}
    # model_class_dict = {'BertModel': BertModel, 'BertForMultipleChoice': BertForMultipleChoice,
    #                     'TransfoXLModel': TransfoXLModel, 'XLNetModel': XLNetModel, 'RobertaModel': RobertaModel,
    #                     'DistilBertModel': DistilBertModel}
    # pretrained_weights_dict = {'BertTokenizer': 'bert-base-uncased',
    #                            'TransfoXLTokenizer': 'transfo-xl-wt103',
    #                            'XLNetTokenizer': 'xlnet-base-cased',
    #                            'RobertaTokenizer': 'roberta-base',
    #                            'DistilBertModel': 'distilbert-base-uncased'}

    partitions = ['dev', 'train', 'test']

    device = get_device()

    for model_str, (tokenizer_class, model_class, pretrained_weights) in CLASSES.items():

    # for tokenizer_str, tokenizer_class in tokenizer_class_dict.items():
    #     pretrained_weights = pretrained_weights_dict[tokenizer_str]
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # for model_str, model_class in model_class_dict.items():
        model = model_class.from_pretrained(pretrained_weights)
        model.to(device)

        for p in partitions:

            try:

                # if os.path.exists(os.path.join(head, 'hf_transformers/data/{}_features_cls_{}.pt'.format(model_str, p))) or \
                #         os.path.exists(os.path.join(head, 'hf_transformers/data/{}_features_mean_{}.pt'.format(model_str, p))) or \
                #         os.path.exists(os.path.join(head, 'hf_transformers/data/{}_features_pool_{}.pt'.format(model_str, p))):
                #     continue

                logging.info('Model: {}, partition: {}'.format(model_str, p))

                logging.info('Loading tokens')
                tokens = load(p)
                logging.info('Finished loading tokens')

                logging.info('Padding')
                concatenated_QA_context = padding(tokens, tokenizer, tokenizer.cls_token, tokenizer.sep_token, 512)
                logging.info('Finished padding')

                logging.info('Encoding')
                input_ids = [tokenizer.encode(s, add_special_tokens=False) for s in concatenated_QA_context]
                logging.info('Finished encoding')
                assert all([len(input_ids[0]) == len(ii) for ii in input_ids[1:]])

                input_ids = torch.tensor(input_ids)

                batch_size = 250
                input_size = (batch_size, *input_ids.size()[1:])

                logging.info(summary(model, input_size))

                torch.save(input_ids, os.path.join(head, 'hf_transformers/data/{}_tokens_{}.pt'.format(model_str, p)))
                # torch.load('')

                logging.info('Starting embeddings')
                num_iterations = (input_ids.size()[0] // batch_size) + 1
                batch_iterator = tqdm.trange(num_iterations, desc='Batch Number')
                for i in batch_iterator:
                    with torch.no_grad():
                        model.eval()
                        if i == num_iterations - 1:
                            temp_ids = input_ids[batch_size * i:, :].to(device)
                        else:
                            temp_ids = input_ids[batch_size * i:batch_size * (i + 1), :].to(device)

                        logging.info('Memory allocated on gpu before loading is {} GB'.format(torch.cuda.memory_allocated(device) *1e-9))
                        logging.info('Memory cached on gpu before lading is {} GB'.format(torch.cuda.memory_cached(device) *1e-9))
                        all_hidden_states = model(temp_ids)
                        logging.info('Memory allocated on gpu after loading is {} GB'.format(
                            torch.cuda.memory_allocated(device) * 1e-9))
                        logging.info('Memory cached on gpu after lading is {} GB'.format(
                            torch.cuda.memory_cached(device) * 1e-9))
                        last_hidden_states = all_hidden_states[0]
                        if model_str in ['BertModel', 'RobertaModel']:
                            pooled_hidden_state = all_hidden_states[1]

                        temp_last_hidden_states_mean = torch.mean(last_hidden_states, dim=1)

                        abs_array = torch.abs(last_hidden_states)
                        max_indices = torch.argmax(abs_array, dim=1)
                        temp_last_hidden_states_pool = torch.zeros(max_indices.size())
                        for ii in range(temp_last_hidden_states_pool.size()[0]):
                            for jj in range(temp_last_hidden_states_pool.size()[1]):
                                temp_last_hidden_states_pool[ii, jj] = last_hidden_states[ii, max_indices[ii, jj], jj]

                        if i is 0:
                            if model_str in ['BertModel', 'RobertaModel']:
                                out_pooled = pooled_hidden_state
                            out_last_pool = temp_last_hidden_states_pool
                            out_last_mean = temp_last_hidden_states_mean
                        else:
                            if model_str in ['BertModel', 'RobertaModel']:
                                out_pooled = torch.cat((out_pooled, pooled_hidden_state), dim=0)
                            out_last_pool = torch.cat((out_last_pool, temp_last_hidden_states_pool), dim=0)
                            out_last_mean = torch.cat((out_last_mean, temp_last_hidden_states_mean), dim=0)
                    logging.info('Finished iteration {} of {}'.format(i, num_iterations))

                logging.info('Finished embeddings')

                if model_str in ['BertModel', 'RobertaModel']:
                    torch.save(out_pooled, os.path.join(head, 'hf_transformers/data/{}_features_cls_{}.pt'.format(model_str, p)))
                torch.save(out_last_mean, os.path.join(head, 'hf_transformers/data/{}_features_mean_{}.pt'.format(model_str, p)))
                torch.save(out_last_pool, os.path.join(head, 'hf_transformers/data/{}_features_pool_{}.pt'.format(model_str, p)))
            except RuntimeError:
                continue






