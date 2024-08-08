# DFPDDI
For our preprint paper "Learning Personalized Drug Features and Differentiated Drug-pair Interaction Information for Drug-drug Interaction Prediction"

The full code is coming soon ..

![AppVeyor](https://img.shields.io/badge/python-3.7.16-blue)
![AppVeyor](https://img.shields.io/badge/pytorch-1.7.1-brightgreen)

## Requirements

The code was trained on Ubuntu 18.04, Intel(R) Xeon(R) Gold 6132 CPU

including:
- python==3.7
- pytorch==1.7.0
- scikit-learn==1.0.2
- torch-geometric==2.0.0
- torch-scatter==2.0.5
- torch-cluster==1.5.9
- CUDA==12.0

## Run code

Note that: You can directly execute shell scripts to run the code

```
conda create -n DFPDDI python==3.7
conda activate DFPDDI
pip install -r ./requirements.txt
python run.py
```

Here is the logic:

1.To learn the structural features of a drug from its molecular graphs, you need to first modify the path in 'drugfeature_fromMG. py'.

```
python drugfeature_fromMG.py
```

2.Training/validating/testing for n times.
```
python run.py
```

3.You can see the final results in 'results/'

