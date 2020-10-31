# SMaBERTa
This repository contains the code for SMaBERTa, a wrapper for the huggingface transformer libraries.
It was developed by Zhanna Terechshenko and Vishakh Padmakumar through research at the Center for 
Social Media and Politics at NYU.

## Setup

To install using pip, run
```
pip install smaberta
```

To install from the source, first download the repository by running 

```
git clone https://github.com/SMAPPNYU/SMaBERTa.git
```

Then, install the dependencies for this repo and setup by running
```
cd SMaBERTa
pip install -r requirements.txt
python setup.py install
```

## Using the package

Basic use:

```
from smaberta import TransformerModel

epochs = 3
lr = 4e-6

training_sample = ['Today is a great day', 'Today is a terrible day']
training_labels = [1, 0]

model = TransformerModel('roberta', 'roberta-base', num_labels=25, reprocess_input_data=True, num_train_epochs=epochs, learning_rate=lr,    
                         output_dir='./saved_model/', overwrite_output_dir=True, fp16=False)

model.train_model(training_sample, training_labels)

```

For further details, see `Tutorial.ipynb` in the [examples](https://github.com/SMAPPNYU/SMaBERTa/tree/master/examples) directory.

# Acknowledgements 

Code for this project was adapted from version 0.6 of https://github.com/ThilinaRajapakse/simpletransformers

Vishakh Padmakumar and Zhanna Terechshenko contributed to the software writing, implementation, and testing.

Megan Brown contributed to documentation and publication.