
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset

from helper import evaluate,data_loader

import pandas as pd
import pickle

from sklearn.metrics import roc_auc_score

outcome = ["MORTALITY_30_DAY", "MORTALITY_1_YEAR", "READMISSION_30_DAY", "READMISSION_1_YEAR"][0]
predictor =["PROCEDURE_ICD", "DIAGNOSIS_ICD", "PROCEDURE_AND_DIAGNOSIS_ICD"][2]

X = pd.read_csv("admit_modified.csv")[[outcome,'LOS', 'AGE', 'GENDER_M', "ETHNICITY_Asian", 
     "ETHNICITY_Black", "ETHNICITY_Hispanic", "ETHNICITY_Native_Hawaiian", "ETHNICITY_Other", 
     "ETHNICITY_White", predictor]]
X = X.dropna()
X = X.reset_index().drop(columns = ["index"])
y = X[outcome].values
X = X.drop(columns = outcome)    

for i in range(len(X)): 
    X[predictor][i] = X[predictor][i].replace("'", "")[1:-1].split(", ")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 1, stratify = y) #stratify
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
train_index = X_train.index
test_index = X_test.index

# Skip-Gram

NEGATIVE_SAMPLES = 20
LR                    = 0.001


BATCH_SIZE            = 256
batch_size = 2**5

NUM_EPOCHS            = 10 #int(1e+3)  

weight_cnn = 0.8
EMBEDDING_DIM = 100
DEVICE = 'cpu'

# add other variables with ICD code together
class_weights = torch.tensor(compute_class_weight( class_weight ="balanced", classes =  np.unique(y_train),y =  y_train ), dtype = torch.float)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')



dataset = word2vec_dataset(predictor, X, train_index, test_index)

'''
with open('dataset.pkl', 'wb') as outp:
    
    pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

'''



X_train = X_train.drop(columns = predictor).dropna()
X_test = X_test.drop(columns = predictor).dropna()


# takes time
vocab = dataset.vocab

word_to_ix = dataset.word_to_ix


vocab_size = len(word_to_ix.keys())
embedding_dict = word_to_ix


train_dataloader, val_dataloader =  data_loader( dataset.code_same_len[train_index],  np.array(X_train), dataset.code_same_len[test_index], np.array(X_test), y_train, y_test, batch_size=BATCH_SIZE)

train_loader_sg = torch.utils.data.DataLoader(dataset.training_data, batch_size = batch_size, shuffle = not True)
test_loader_sg = torch.utils.data.DataLoader(dataset.testing_data, batch_size = batch_size, shuffle = not True)

print('len(dataset): ', len(dataset))
print('len(train_loader_sg): ', len(train_loader_sg))
print('len(train_dataloader): ', len(train_dataloader))
print('len(vocab): ', len(vocab), '\n')


# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))


losses = []
word2vec = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
print('\nWe have {} Million trainable parameters here in the word2vec'.format(count_parameters(word2vec)))

optimizer = optim.Adam(word2vec.parameters(), lr = LR)




for epoch in range(NUM_EPOCHS):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    print('\nTRAINING...')

    train_accuracy = [ ]
    train_auroc = []
    
    train_loss = []
    word2vec.train()
    
    for item1, item2 in zip(train_loader_sg, train_dataloader): 
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        
        # X is the input ids and X1 is other features
        X, X1, y = item2

        optimizer.zero_grad()
        
        
        loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        
        loss_cnn = criterion_cnn(logits,y)
        loss =( 1-weight_cnn)*loss_word2vec + weight_cnn* loss_cnn  # * weight
        train_loss.append(loss.item())


        loss.backward(retain_graph=True)
        optimizer.step()    
        
                
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        proba =logits[:,1].detach().numpy()


        accuracy = (preds == y).cpu().numpy().mean() * 100
        
        
        auroc =  roc_auc_score(y, proba)
        
        train_accuracy.append(accuracy)
        train_auroc.append(auroc)
    


    print("average train loss", np.mean(train_loss), '\n')
    print("average train accuracy", np.mean(train_accuracy), '\n')
    print("train auroc", np.mean(train_auroc)*100, '\n')
    
   
    print("VALIDATION... \n")
    val_loss, val_accuracy, val_auroc = evaluate(word2vec, val_dataloader, test_loader_sg, word2vec, criterion_cnn, weight_cnn)
    print("val loss", val_loss, '\n')
    print("val accuracy", val_accuracy, '\n')
    print("val auroc", np.mean(val_auroc)*100, '\n')

