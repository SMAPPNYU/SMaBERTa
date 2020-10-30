# SMaBERTa
This repository contains the code for SMaBERTa, a wrapper for the huggingface transformer libraries.
It was developed by Zhanna Terechshenko and Vishakh Padmakumar through research at the Center for 
Social Media and Politics at NYU.

## Setup

To download the repository, run 

```
git clone https://github.com/SMAPPNYU/SMaBERTa.git
```

To install the dependencies for this repo, run
```
cd SMaBERTa
pip install -r requirements.txt
python setup.py install
```

## Repository Contents

smaberta.py - main file.

For the example on how to use the model for the classification task follow Tutorial.ipynb.

For language model finetuning follow test_finetuning.py.

# Acknowledgements 

Code for this project was adapted from version 0.6 of https://github.com/ThilinaRajapakse/simpletransformers

Zhanna Terechshenko and Vishakh Padmakumar contributed to the software writing, implementation, and testing.

Megan Brown contributed to documentation and publication.