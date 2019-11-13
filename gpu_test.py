import torch
import logging
import os
import getpass

if getpass.getuser() == 'Mitch':
    head = 'C:/Users/Mitch/PycharmProjects'
else:
    head = '/home/kinne174/private/PythonProjects/'

logging.basicConfig(filename=os.path.join(head, 'hf_transformers/log/logging-gpu.log'),
                    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info('gpu available {}'.format(torch.cuda.is_available()))
logging.info('gpu info {}'.format(torch.cuda.current_device()))
logging.info('{}'.format(torch.cuda.device_count()))