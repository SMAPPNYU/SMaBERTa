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

train_df = pd.read_csv("~/smapp_stance_classifier/data/clinton_2016_smapp/train.csv")

mapping = {"Positive":2, "Negative":0, "Neutral":1}

temp = train_df.applymap(lambda s: mapping.get(s) if s in mapping else s)

labels = list(temp["label"])
texts = list(temp["text"])

train_data = [[texts[i], labels[i]] for i in range(len(texts))]

train_data = pd.DataFrame(train_data)

test_df = pd.read_csv("~/smapp_stance_classifier/data/clinton_2016_smapp/test.csv")

test_temp = test_df.applymap(lambda s: mapping.get(s) if s in mapping else s)

labels = list(test_temp["label"])
texts = list(test_temp["text"])
test_data = [[texts[i], labels[i]] for i in range(len(texts))]
test_data = pd.DataFrame(test_data)

lrs = [1e-6, 4e-6, 1e-5, 4e-5, 1e-4, 1e-3]
train_epochs = [1, 5, 10, 20]
best_accuracy = 0
results = {}
count = 0
#for lr in lrs:
#    for epochs in train_epochs:
print("-----------------------------------------------")
lr = 1e-5
epochs = 5
print("Learning Rate ", lr)
print("Train Epochs ", epochs)

model = TransformerModel('roberta', 'roberta-base', num_labels=3, args={'reprocess_input_data': True, "num_train_epochs":epochs, "learning_rate":lr, 'overwrite_output_dir': True, 'fp16':False})

model.train_model(train_data)
"""
        test_df = pd.read_csv("~/smapp_stance_classifier/data/clinton_2016_smapp/test.csv")

        test_temp = test_df.applymap(lambda s: mapping.get(s) if s in mapping else s)

        labels = list(test_temp["label"])
        texts = list(test_temp["text"])
        test_data = [[texts[i], labels[i]] for i in range(len(texts))]
        test_data = pd.DataFrame(test_data)
"""
result, model_outputs, wrong_predictions = model.eval_model(test_data)
preds = np.argmax(model_outputs, axis = 1)

correct = 0
for i in range(len(labels)):
    if preds[i] == labels[i]:
        correct+=1

accuracy = correct/len(labels)
print("Accuracy: ", accuracy)
#results[count] = {'lr':lr, 'epochs':epochs, 'accuracy':accuracy}
#count+=1
#pickle.dump(results, open("cv.pkl", "wb"))
if accuracy > best_accuracy:
    best_params = (lr, epochs)
    best_preds = preds
    best_accuracy = accuracy
    pickle.dump(model_outputs, open("model_outputs.pkl", "wb"))

#import code
#code.interact(local = locals())
