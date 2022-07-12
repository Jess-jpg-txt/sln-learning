# sln-learning

This repository hosts code used to obtain results in our paper: [Predicting Learning Interactions in Social Learning Networks: A Deep Learning Enabled Approach.]()

## Repository Structure

* `data/`: In this folder, you can find the 4 of the 6 datasets we used in our paper, `algo004`, `comp`, `ml` and `virtualshakespeare`. Due to personal identifiable information in the other two datasets (Piazza discussion platform data), we are not making them available. 
  * `metadata.txt` contains information on what's included in each of the data files
  * Rest of the files with naming convention `w_removal_{}` are data files. They are obtained after feature engineering and filtering out isolation nodes.
* `algo004/`, `comp/`, `ml/` and `virtualshakespeare/`: Each of these folders pertains to one dataset. Each folder includes:
  * multiple code files named as `kfold_{}.py` corresponding to different feature selections. Each file includes model architectures, training process, cross validated accuracy and AUC calculation.
  * a `Makefile` to facilitate running the experiments
* `gnn.py`: A standalone script that implements Graph Neural Network (GNN) for SLN and produces accuracy and AUC results. This script runs the GNN and calculates performance for all the datasets.
* `requirements.txt`: a snapshot of the Python package versions the experiments were run with.

## Get Started

### How to Run the Experiments

To run all the experiments for one dataset, simply go into the folder corresponding to this dataset, then run in the terminal:
```
make run_all
```
Or to run any of the individual files, simply go into an experiment folder and run:
```
python3 <filename>
```
Example:
```
> pwd
sln-learning/comp
> python3 kfold_all_feats.py
...
```

### Notes on Reproducability

You might obtain slightly different result by running code, due to the following sources of randomness:
* k-fold cross validation creates random splits
* model parameters are initialized randomly


### Package Requirements

To ensure success running of the program, the versions Python packages we used are listed in `requirements.txt`. To align the versions of your packages to this file, simply run:

```
pip install -r requirements.txt
```
