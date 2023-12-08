
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import compute_class_weight
from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset
from helper import evaluate,data_loader, plot_and_save_roc_curve, print_result, load_hyperparameters, load_and_split




hyperparameters = load_hyperparameters('config.yaml')
NEGATIVE_SAMPLES = hyperparameters.get('negative_samples')
LR = hyperparameters.get('lr')
BATCH_SIZE = hyperparameters.get('batch_size1')
batch_size = hyperparameters.get('batch_size2')
NUM_EPOCHS = hyperparameters.get('num_epochs')
weight_cnn = hyperparameters.get('weight_cnn')
EMBEDDING_DIM = hyperparameters.get('embedding_dim')
DEVICE = hyperparameters.get('device')
outcome = hyperparameters.get('outcome')
predictor = hyperparameters.get('predictor')


X, X_train, X_test, y_train, y_test, train_index, test_index = load_and_split(outcome, predictor)
dataset = word2vec_dataset(predictor, X, train_index, test_index)
# add other variables with ICD code together
class_weights = torch.tensor(compute_class_weight(class_weight ="balanced", classes =  np.unique(y_train),y =  y_train ), dtype = torch.float)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

X_train = X_train.drop(columns = predictor).dropna()
X_test = X_test.drop(columns = predictor).dropna()


# takes time
vocab = dataset.vocab
embedding_dict = dataset.word_to_ix
vocab_size = len(embedding_dict.keys())


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



train_roc_l = []
val_roc_l = []
train_loss_l = []
val_loss_l = []
for epoch in range(NUM_EPOCHS):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    print('\nTRAINING...')
    word2vec.train()
    max_auroc = float('-inf')
    
    for item1, item2 in zip(train_loader_sg, train_dataloader): 
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        
        # X is the input ids and X1 is other features
        X, X1, y = item2
        optimizer.zero_grad()
        loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        loss_cnn = criterion_cnn(logits,y)
        loss =( 1-weight_cnn)*loss_word2vec + weight_cnn* loss_cnn  # * weight
        loss.backward(retain_graph=True)
        optimizer.step()    


    print('EPOCH END TRAINING...\n')
    train_loss, train_accuracy, train_auroc, train_auprc, train_true, train_proba = evaluate(word2vec, train_dataloader, train_loader_sg, word2vec, criterion_cnn, weight_cnn)
    print_result(train_loss, train_accuracy, train_auroc, train_auprc, 'train')

    print("VALIDATION... \n")
    val_loss, val_accuracy, val_auroc, val_auprc, val_true, val_proba = evaluate(word2vec, val_dataloader, test_loader_sg, word2vec, criterion_cnn, weight_cnn)
    print_result(val_loss, val_accuracy, val_auroc, val_auprc, 'val')

    train_roc_l.append(np.mean(train_auroc))
    val_roc_l.append(np.mean(val_auroc))
    train_loss_l.append(train_loss)
    val_loss_l.append(val_loss)

    # Call the function to plot and save the ROC curve
    if np.mean(val_auroc) > max_auroc:
        plot_and_save_roc_curve(val_true, val_proba, epoch, 'plots/roc.png')
        max_auroc = np.mean(val_auroc)
        
print(train_loss_l) 
print(val_loss_l)

print(train_roc_l) 
print(val_roc_l)
