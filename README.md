# Str-MINDFUL
# 

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Corrado Loglisci, Vincenzo Belvedere, Domenico Redavid and Donato Malerba_

[A Network Intrusion Detection System for Concept Drifting Network Traffic Data](https://github.com/gsndr/Str-MINDFUL) 

Please cite our work if you find it useful for your research and work.
```
 
```
## Getting Started 

### Installation
Str-MINDFUL requires Python 3 (preferably >= 3.6) 
Packages need are:

* [Tensorflow 2.3.1](https://www.tensorflow.org/) 
* [Keras 2.4.3](https://github.com/keras-team/keras)
* [Matplotlib 3.3.2](https://matplotlib.org/)
* [Pandas 1.1.3](https://pandas.pydata.org/)
* [Numpy 1.18.5](https://www.numpy.org/)
* [Scikit-learn 0.23.2](https://scikit-learn.org/stable/)
* [Scikit-multiflow 0.5.3](https://scikit-multiflow.github.io/)
* [Hyperas 0.4.1](https://github.com/maxpumperla/hyperas)
* [Hyperopt 0.2.5](https://github.com/hyperopt/hyperopt)
* [Pillow 7.2.0](https://pillow.readthedocs.io/en/stable/)

Package dependencies can be installed by using the listing in `requirements.txt`. 

```shell 
pip install -r requirements.txt
```

## How to use
Global variables are stored in  *settings* section of the file MINDFUL.conf: 

    [setting]
    EXECUTION_TYPE              # 0=>ebatch execution 1=>streaming execution
    N_CLASSES                   # number of classes for the prediction
    PREPROCESSING1              # 0=>do not preprocessing 1=>do preprocessing
    LOAD_AUTOENCODER_ADV        # 0=>train autoencoder of attacks  1=> load trained autoencoder of attacks
    LOAD_AUTOENCODER_NORMAL     # 0=>train autoencoder of normal samples  1=> load trained autoencoder of normal samples
    LOAD_CNN                    # 0=>train the CNN1D   1=> load trained CNN1D
    VALIDATION_SPLIT            # split train/validation
    WINDOW_TYPE                 # 0=>count window  1=>time window
    DECREASE_BATCH_DATASET      # 0=>use the complet dataset from batch during the stream 1=>reduce the dimension of the batch dataset
    
In addition to configuring you must also enter information about the dataset:

    [dataset_name]
    pathModels # path to hold the models
    pathPlot # path of the plot 
    pathDataset # path where the original datasets are stored
    path # name of the batch dataset file
    pathTest # file name of the dataset used as test in the basic version
    pathStream # file name of the dataset used for streaming
    testPath # the same as the dataset name
    pathDatasetNumeric # path to contain the preprocessed numeric dataset files
    pathUtils # path to the folder where to put the streaming computation support files
    countWindowNormal # size of countWindow of normal examples
    countWindowAttack # size of the countWindow of the attack examples
    smallerSize # size of the batch dataset after sampling
    timeDeltaNormal # time allowed in the time window of normal examples
    timeDeltaAttack # time allowed in the timeWindow of the attack examples
    
Two steps are required for the execution of the stream algorithm: a batch training and ta streaming phase

### Batch training

Set the value *EXECUTION_TYPE=0*, the value *path* with the name of the dataset you want to use as a batch, and run main.py using the name of the dataset as input.

    python3 main.py dataset_name

This will run the basic version of MINDFUL, which will take care of the preliminary training of the models by choosing the best hyperparameters. If you want more information about this step, you can consult [this repository](https://github.com/gsndr/MINDFUL)

### Streaming execution

Set *EXECUTION_TYPE=1*, the value of *pathStream* with the name of the dataset and run main.py again with the command:

    python3 main.py dataset_name

PLEASE NOTE: The current version of STREAM MINDFUL has been implemented based on the use of the CICIDS2017 dataset. If you want to use another dataset, you need to add in the *Datasetconfig.preprocessing2()* the functions for the preprocessing  step.


## Download datasets
The dataset used for experiments can downloaded from here:

[CICIDS2017](hhttps://drive.google.com/file/d/1ENI6gvSH48-QOvppVTdtzJZhRSHPtCIJ/view?usp=sharing)
