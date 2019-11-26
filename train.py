import numpy as np
import json_lines
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
import torch
import multiprocessing

# TODO some sort of fold cross validation with vectors as is, assume each line individually but group them to measure accuracy
# TODO also train an MLP

def labels(json_filename):

    correct_list = []
    id_list = []
    # assuming the structure of the file is the same as the ARC context
    with open(json_filename, 'r', encoding='utf-8') as f:
        json_reader = json_lines.reader(f)
        for i, item in enumerate(json_reader):
            id = item['id']
            for c in item['question']['choices']:
                choice_label = c['label']
                correct = choice_label == item['answerKey']

                correct_list.append(correct)
                id_list.append('{}*-*{}'.format(id, choice_label))

    return correct_list, id_list

def QA_groupings(id_list):

    unique_ids = list(np.unique([id[:id.index('*-*')] for id in id_list]))
    grouped_ids = []
    for id in id_list:
        ui_index = unique_ids.index(id[:id.index('*-*')])
        grouped_ids.append(ui_index)

    return grouped_ids

def evaluate(y_pred, y_test, y_groupings):
    unique_grouping = list(np.unique(y_groupings))
    count_grouping = [y_groupings.count(ug) for ug in unique_grouping]
    assert sum(count_grouping) == y_pred.shape[0] == y_test.shape[0]

    correct = 0

    for i in range(len(count_grouping)):
        if i == len(count_grouping) - 1:
            current_pred = y_pred[sum(count_grouping[:i]):]
            current_test = y_test[sum(count_grouping[:i]):]
        else:
            current_pred = y_pred[sum(count_grouping[:i]):sum(count_grouping[:(i + 1)])]
            current_test = y_test[sum(count_grouping[:i]):sum(count_grouping[:(i + 1)])]

        if np.argmax(current_pred) == np.argmax(current_test):
            correct += 1

    percentage_correct = correct/len(unique_grouping)

    return percentage_correct

    # TODO output more info to logging, also not sure what info to save from here, the predictions maybe?

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    import logging
    import glob
    import getpass
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='cls')
    parser.add_argument('--train_nn', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='BertModel')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    if getpass.getuser() == 'Mitch':
        head = 'C:/Users/Mitch/PycharmProjects'
    else:
        head = '/home/kinne174/private/PythonProjects/'

    all_train_filenames = glob.glob(os.path.join(head, 'hf_transformers/log/logging-train*'))

    logging.basicConfig(filename=os.path.join(head, 'hf_transformers/log/logging-train{}.log'.format(len(all_train_filenames))),
                        level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info(args)

    logging.info('Collecting features')
    dev_features_filename = os.path.join(head, 'hf_transformers/data/{}_features_{}_dev.pt'.format(args.model_name, args.features))
    if os.path.exists(dev_features_filename):
        dev_features = torch.load(dev_features_filename, map_location=get_device())
    else:
        logging.info('Dev features are random')
        dev_features = torch.rand(50, 100)

    train_features_filename = os.path.join(head, 'hf_transformers/data/{}_features_{}_train.pt'.format(args.model_name, args.features))
    if os.path.exists(train_features_filename):
        train_features = torch.load(train_features_filename, map_location=get_device())
    else:
        logging.info('Train features are random')
        train_features = torch.rand(100, 100)

    test_features_filename = os.path.join(head, 'hf_transformers/data/{}_features_{}_test.pt'.format(args.model_name, args.features))
    if os.path.exists(test_features_filename):
        test_features = torch.load(test_features_filename, map_location=get_device())
    else:
        logging.info('Test features are random')
        test_features = torch.rand(75, 100)

    # X = torch.cat((train_features, dev_features, test_features), dim=0).numpy()

    logging.info('Number of features: {}'.format(train_features.shape[1]))

    correct_dict, id_dict = {}, {}
    for p in ['train', 'dev', 'test']:
        filename = os.path.join(head, 'ARC/ARC-with-context/{}.jsonl'.format(p))
        temp_correct_list, temp_id_list = labels(filename)
        correct_dict[p] = np.array(temp_correct_list, dtype=np.int)
        id_dict[p] = temp_id_list

    # y = np.array(correct_list, dtype=np.int)
    if getpass.getuser() == 'Mitch':
        for p, f in zip(['train', 'dev', 'test'], [train_features, dev_features, test_features]):
            correct_dict[p] = correct_dict[p][:f.shape[0]]
            id_dict[p] = id_dict[p][:f.shape[0]]
        # y = y[:X.shape[0]]
        # id_list = id_list[:X.shape[0]]
    else:
        for p, f in zip(['train', 'dev', 'test'], [train_features, dev_features, test_features]):
            assert correct_dict[p].shape[0] == f.shape[0]
            logging.info('Number of {} observations: {}'.format(p, f.shape[0]))

    # X_train, X_test, y_train, y_test, y_groupings_te, y_groupings_tr = train_test_split_QA(X, y, id_list, test_size=0.2,
    #                                                                                        random_state=args.seed)

    # logging.info('training size: {}, testing size: {}'.format(X_train.shape[0], X_test.shape[0]))

    if args.train_nn:
        logging.info('Start: CUDA available: {}'.format(torch.cuda.is_available()))
        assert torch.cuda.is_available()

    else:
        logging.info('Start: Using the cpu')

        num_cores = 2 if getpass.getuser() == 'Mitch' else multiprocessing.cpu_count()
        logging.info('Using {} cores'.format(num_cores))

        C_params = {'C': np.exp(np.arange(-8, 9, 2))}

        svm_linear = SVC(kernel='linear', probability=True)
        svm_rbf = SVC(kernel='rbf', gamma='scale', probability=True)
        logistic_l1 = LogisticRegression(max_iter=10000, penalty='l1')
        logistic_l2 = LogisticRegression(max_iter=10000, penalty='l2')
        logistic_elastic = LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga')

        solvers = {'log_l1': logistic_l1, 'log_l2': logistic_l2, 'log_en': logistic_elastic}

        logging.info('Starting to fit models')
        for i, (model_name, model) in enumerate(solvers.items()):
            logging.info('Evaluating {}'.format(model_name))
            if model_name == 'log_en':
                C_params = {'C': np.exp(np.arange(-8, 9, 2)), 'l1_ratio': np.arange(.1, 1, .2)}

            logging.info('All parameters: {}'.format(C_params))
            X_train = np.concatenate((train_features, dev_features), axis=0)
            y_train = np.concatenate((correct_dict['train'], correct_dict['dev']))
            test_fold = [-1]*train_features.shape[0] + [0]*dev_features.shape[0]
            ps = PredefinedSplit(test_fold=test_fold)

            groupings_dev = QA_groupings(id_dict['dev'])

            my_scorer = make_scorer(evaluate, y_groupings=groupings_dev)
            clf = GridSearchCV(model, C_params, scoring=my_scorer, cv=ps, n_jobs=num_cores)
            clf.fit(X_train, y_train)

            logging.info('Best parameters: {}'.format(clf.best_params_))
            logging.info('Dev Scores: {}'.format(clf.best_score_))

            X_test = test_features
            y_test = correct_dict['test']

            grouping_test = QA_groupings(id_dict['test'])

            y_pred = clf.predict_proba(X_test)
            y_pred = y_pred[:, 1].reshape((-1,))

            assert y_pred.shape[0] == y_test.shape[0]

            percentage_correct_te = evaluate(y_pred, y_test, grouping_test)
            logging.info('Testing accuracy for {}: {}'.format(model_name, percentage_correct_te))

            y_pred = clf.predict_proba(X_train)
            y_pred = y_pred[:, 1].reshape((-1,))

            assert y_pred.shape[0] == y_train.shape[0]

            grouping_train = QA_groupings(id_dict['train'] + id_dict['dev'])

            percentage_correct_tr = evaluate(y_pred, y_train, grouping_train)
            logging.info('Training accuracy for {}: {}'.format(model_name, percentage_correct_tr))















