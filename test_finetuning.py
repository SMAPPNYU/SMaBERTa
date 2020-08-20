import pandas as pd
import numpy as np
import random
import torch
import pickle
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

from smaberta import TransformerModel

model = TransformerModel('roberta', 'roberta-base', finetune=True, args={"num_train_epochs":1, 'fp16':False})

model.finetune("./data/lm_train", "./data/lm_eval")

model.lm_evaluate()