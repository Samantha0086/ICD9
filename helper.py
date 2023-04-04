import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
import pandas as pd
from sklearn.metrics import roc_auc_score



def evaluate(model, val_dataloader, test_loader_sg,word2vec,criterion_cnn, weight_cnn):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    val_auroc = []

    # For each batch in our validation set...
    for item1, item2 in zip(test_loader_sg, val_dataloader):
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        X, X1,y = item2

        # Compute logits
        with torch.no_grad():
            loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        loss_cnn = criterion_cnn(logits,y)
        loss = (1-weight_cnn)*loss_word2vec +weight_cnn* loss_cnn  # * weight
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        proba = logits[:, 1].detach().numpy()

        # Calculate the accuracy rate
        accuracy = (preds == y).cpu().numpy().mean() * 100
        
        auroc =  roc_auc_score(y, proba)
        

        val_auroc.append(auroc)
        
      
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    

    return val_loss, val_accuracy, val_auroc




def data_loader(train_inputs,train_inputs2,  val_inputs,val_inputs2, train_labels, val_labels,
                batch_size):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """
    # train_inputs = np.concatenate((train_inputs, train_inputs2), axis=1)
    # val_inputs = np.concatenate((val_inputs, val_inputs2), axis=1)
    # Convert data type to torch.Tensor
    train_inputs, train_inputs2, val_inputs, val_inputs2, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, train_inputs2.astype(np.float32), val_inputs, val_inputs2.astype(np.float32), train_labels, val_labels])



    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_inputs2, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs,val_inputs2, val_labels)
    val_sampler = SequentialSampler(val_data)
    
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader



