# A Sequence Learning Model for Decision Making in the Brain
# Source code for the network model and the analyses.

The Python code for the analyses in Zhang et al.(2019)

### Installation
Download the source code into a local directory.

### Set up the path
Open and edit ./seqrnn_multitask/main.py Change variable 'path' to current folder.

### Environment
This code has been tested on python 3.7.3 in linux.

### Generating the dataset
Execute ./datagenerator/data_generatorRT_training.py and ./datagenerator/data_generatorRT_training.py for generating the trainingset and validataset, respectively.
The generated dataset is save in the folder ./data

### Running the code
Edit the ./seqrnn_multitask/config.py, rt_shape['data_file'] and rt_shape['validation_data_file'] should be the filename of the trainingset and validataset you have just generated.

Change other parameters you want in the config.py

Run main.py

### Datasaving

The trained model is saved in folder ./save_m
The log of the behavioral results and neuronal response is saved in ./log_m 

### Data analysis

All the analy code is available in the folder ./holmes_m













