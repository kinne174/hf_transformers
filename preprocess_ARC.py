import json_lines
import json
import getpass
import os

def preprocess_ARC():
    if getpass.getuser() == 'Mitch':
        # directory on my computer
        head = 'C:/Users/Mitch/PycharmProjects'
    else:
        # directory on compute
        head = '/home/kinne174/private/PythonProjects'

    difficulties = ['Easy', 'Challenge']
    partitions = ['Train', 'Dev', 'Test']

    # all_filenames = ['ARC-' + '-'.join(dp) + '.jsonl' for dp in product(difficulties, partitions)]

    for d in difficulties:
        for p in partitions:
            output = []

            ARC_filename = 'ARC/ARC-V1-Feb2018-2/ARC-{}/ARC-{}-{}.jsonl'.format(d, d, p)
            dataset_filename = os.path.join(head, ARC_filename)

            if os.path.exists(dataset_filename):
                with open(dataset_filename, 'r', encoding='utf-8') as df:
                    data = {}
                    for ind, item in enumerate(json_lines.reader(df)):
                        data['id'] = item['id']
                        data['question'] = item['question']['stem']
                        data['choices_text'] = [choice['text'] for choice in item['question']['choices']]
                        data['choices_labels'] = [choice['label'] for choice in item['question']['choices']]
                        data['answer'] = item['answerKey']

                        output += [data]

            else:
                raise Exception("Filename {} does not exist!".format(dataset_filename))

            with open(os.path.join(head, 'hf_transformers/data/{}-{}.json'.format(d, p)), 'w') as of:
                json.dump(output, of)


if __name__ == '__main__':
    preprocess_ARC()