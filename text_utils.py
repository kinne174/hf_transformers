import json_lines
import numpy as np
from collections import namedtuple
import os
import getpass
import time

if getpass.getuser() == 'Mitch':
    head = 'C:/Users/Mitch/PycharmProjects'
else:
    head = '/home/kinne174/private/PythonProjects/'


class TextDownload:
    def __init__(self, dataset_name, partition, difficulty):
        assert isinstance(dataset_name, str)
        assert isinstance(partition, str)
        assert isinstance(difficulty, str)

        assert partition in ['dev', 'test', 'train']
        # assert difficulty in ['Challenge', 'Easy']

        self.dataset_name = dataset_name
        self.partition = partition
        self.difficulty = difficulty  # unused

    def load(self):
        if self.dataset_name == 'ARC':
            filename = os.path.join(head, 'ARC/ARC-with-context/{}.jsonl'.format(self.partition))
        else:
            raise Exception('Not implemented yet')

        all_context = []

        with open(filename, 'r') as file:
            jsonl_reader = json_lines.reader(file)
            for line in jsonl_reader:
                all_context.extend(self.collect(line))

        return all_context

    @staticmethod
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

    @staticmethod
    def padding(all_context, tokenizer, cls_token, sep_token, max_allowed_len):

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

