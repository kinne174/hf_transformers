import os
import numpy as np
import spacy
import getpass
from scipy.sparse import csr_matrix, save_npz
import json

if getpass.getuser() == 'Mitch':
    # directory on my computer
    head = 'C:/Users/Mitch/PycharmProjects'
    nlp_model = 'en_core_web_sm'
else:
    # directory on compute
    head = '/home/kinne174/private/PythonProjects'
    nlp_model = 'en_core_web_md'

nlp = spacy.load(nlp_model, disable=['parser' 'ner', 'textcat'])

def doc_to_spans(list_of_texts, join_string=' ||| '):
    # convert list of text to lemmas using work around
    # https://towardsdatascience.com/a-couple-tricks-for-using-spacy-at-scale-54affd8326cf
    num_iterations = int(np.ceil(len(list_of_texts) / 1000))
    new_docs = []
    for ii in range(num_iterations):
        temp_list_of_texts = list_of_texts[ii * 1000:(ii + 1) * 1000]
        stripped_string = join_string.strip()
        all_docs = nlp(join_string.join(temp_list_of_texts))
        split_inds = [i for i, token in enumerate(all_docs) if token.text == stripped_string] + [len(all_docs)]
        new_docs.extend([all_docs[(i + 1 if i > 0 else i):j] for i, j in zip([0] + split_inds[:-1], split_inds)])
    return new_docs

def filter_docs(original_docs):
    new_docs = []
    # keep tokens that are not stop words, punctuation, docs that contain less than some threshold on punctuation/ numbers
    for doc in original_docs:
        if len(doc) < 4:  # less than 5 tokens in the doc skip it
            continue
        values = [(token.is_punct, token.is_digit, token.pos_ == 'SYM', not token.has_vector, token.like_url) for
                  token in doc]
        sums = np.sum(values, axis=0)
        if sums[0] + sums[1] + sums[2] >= .5 * len(doc):  # more than half of doc is punctuation, digits and symbols
            continue
        if sums[3] + sums[4] > 0:  # at least one of the tokens does not have a vector or is a url
            continue
        new_docs.append(' '.join([token.text for token in doc if not (token.is_punct or token.pos_ == 'SYM')]))

    return new_docs


def preprocess_context_ARC():
    ARC_context_filename = 'ARC/ARC-V1-Feb2018-2/ARC_Corpus.txt'

    ARC_filename = os.path.join(head, ARC_context_filename)
    if not os.path.exists(ARC_filename):
        raise Exception('The filename {} does not exist!'.format(ARC_filename))

    with open(ARC_filename, 'r', encoding='utf-8') as corpus:
        temp_text = []

        for ind, line in enumerate(corpus):
            temp_text.append(line.strip())

            if ind >= 2000:
                break

        spacy_docs = doc_to_spans(temp_text)
        text_docs = filter_docs(spacy_docs)

        corpus.close()

    words_d = {}  # dict of all words with key 'word' and value id number to be used in sparse matrix
    sentences_d = {}  # dict of all sentences with key id number and value list of word id numbers in order of usage

    # indices to plug into the sparse matrix
    word_indices = []
    sentence_indices = []

    word_index = 0

    with open(os.path.join(head, 'hf_transformers/data/ARC_corpus_new.txt'), 'w') as wf:
        for i, sentence in enumerate(text_docs):
            try:
                wf.write('*_{} {}\n'.format(i, sentence + '.'))
            except UnicodeEncodeError:
                continue

            sentence_split = sentence.split(' ')
            temp_sentence = []
            for word in sentence_split:
                word = word.lower()
                if word in words_d:
                    temp_sentence.append(words_d[word])
                    word_indices.append(words_d[word])
                else:
                    words_d[word] = word_index
                    word_indices.append(word_index)
                    temp_sentence.append(word_index)
                    word_index += 1
                sentence_indices.append(i)


            # TODO go through words_d to see if there's a simple way to get rid of the things I don't want
            # TODO put a try statement on writing the sentence and continue if there's an error

            sentences_d[i] = temp_sentence

        wf.close()

    # need to save the word dict to a json file still
    with open(os.path.join(head, 'hf_transformers/data/ARC_words.json'), 'w') as word_file:
        json.dump(words_d, word_file)
    with open(os.path.join(head, 'hf_transformers/data/ARC_sentences.json'), 'w') as sentence_file:
        json.dump(sentences_d, sentence_file)

    ones = np.ones((len(sentence_indices,)), np.int)
    sentence_arr = np.array(sentence_indices)
    word_arr = np.array(word_indices)

    sp_mat = csr_matrix((ones, (sentence_arr, word_arr)), (len(text_docs), len(word_indices)))
    save_npz(os.path.join(head, 'hf_transformers/data/ARC_context_sparse_matrix.npz'), sp_mat)


if __name__ == '__main__':
    preprocess_context_ARC()




