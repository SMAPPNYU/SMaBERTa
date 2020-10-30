## Credit
Code for this project was adapted from version 0.6 of https://github.com/ThilinaRajapakse/simpletransformers

## Setup
Aside from PyTorch, Scikit Learn and Scipy the implementation comes from huggingface/transformers

A dockerfile is provided to execute the scripts with the caveat of using PyTorch in CPU mode which is pretty slow. 

Additionally you could install the dependencies at the right version using the Requirements file. 

```
pip install -r requirements.txt
```

## Usage
smaberta.py - main file.

For the example on how to use the model for the classification task:
Follow Tutorial.ipynb

For language model finetuning:
Follow test_finetuning.py


