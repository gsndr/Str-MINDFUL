import configparser
import sys
import numpy as np

np.random.seed(12)
import tensorflow

tensorflow.random.set_seed(12)

from CNN1D import RunCNN1D
from RunStream import RunStream
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def datasetException():
    try:
        dataset = sys.argv[1]

        if (dataset is None):
            raise Exception()
        if not ((dataset == 'KDDCUP99') or (dataset == 'UNSW_NB15') or (dataset == 'CICIDS2017') or (
                dataset == 'AAGM')):
            raise ValueError()
    except Exception:
        print("The name of dataset is null: use KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    except ValueError:
        print("Dataset not exist: must be KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    return dataset

def main():
    dataset = datasetException()

    config = configparser.ConfigParser()
    config.read('MINDFUL.conf')

    dsConf = config[dataset]
    configuration = config['setting']

    if int(configuration.get('EXECUTION_TYPE')) == 0:

        execution = RunCNN1D(dsConf, configuration)
        execution.run()

    else:

        execution = RunStream(dsConf, configuration)
        execution.run()


if __name__ == "__main__":
    main()
