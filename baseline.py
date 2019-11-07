import getpass
import os
import numpy as np
from datetime import datetime

import logging

logging.basicConfig(filename='logging.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


if getpass.getuser() == 'Mitch':
    os.chdir('C:/Users/Mitch/PycharmProjects')
else:
    os.chdir('/home/kinne174/private/PythonProjects/')

    #