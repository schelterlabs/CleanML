"""Main function"""

import numpy as np
import utils
import json
import argparse
import datetime
import time
import config
from experiment import experiment
from relation import populate

# Changes annotated with CHANGES-SSC

datasets = [utils.get_dataset('USCensus')]

experiment(datasets, log=False, n_jobs=1, nosave=False, error_type='missing_values')
