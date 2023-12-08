import numpy as np
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_and_split(outcome, predictor):
    X = pd.read_csv("admit_modified.csv")[[outcome,'LOS', 'AGE', 'GENDER_M', "ETHNICITY_Asian", 
     "ETHNICITY_Black", "ETHNICITY_Hispanic", "ETHNICITY_Native_Hawaiian", "ETHNICITY_Other", 
     "ETHNICITY_White", predictor]]
    X = X.dropna()
    X = X.reset_index().drop(columns = ["index"])
    y = X[outcome].values
    X = X.drop(columns = outcome)    
    X[predictor] = X[predictor].apply(lambda x: x.replace("'", "")[1:-1].split(", "))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 1, stratify = y) 
    train_index = X_train.index
    test_index = X_test.index
    return X, X_train, X_test, y_train, y_test, train_index, test_index



def load_hyperparameters(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_result(loss, accuracy, auroc, auprc, mode):


    print(f"{mode} loss", loss, '\n')
    print(f"{mode} accuracy", accuracy, '\n')
    print(f"{mode} auroc", np.mean(auroc)*100, '\n')
    print(f"{mode} auprc", np.mean(auprc)*100, '\n')


def plot_and_save_roc_curve(y_true, y_proba, epoch, file_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Validation ROC - Epoch {epoch+1}')
    plt.legend(loc="lower right")
    # Save the figure
    plt.savefig(file_path)
    plt.close()  # Close the figure to avoid displaying it in the notebook

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
    val_auprc = []
    val_proba = []
    val_true = []

    # For each batch in our validation set...
    for item1, item2 in zip(test_loader_sg, val_dataloader):
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        X, X1,y = item2
        val_true.extend(y.numpy())

        # Compute logits
        with torch.no_grad():
            loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        loss_cnn = criterion_cnn(logits,y)
        loss = (1-weight_cnn)*loss_word2vec +weight_cnn* loss_cnn  # * weight
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        proba = logits[:, 1].detach().numpy()
        val_proba.extend(proba)

        # Calculate the accuracy rate
        accuracy = (preds == y).cpu().numpy().mean() * 100
        
        auroc =  roc_auc_score(y, proba)
        auprc = average_precision_score(y, proba)
        

        val_auroc.append(auroc)
        val_auprc.append(auprc)
        
      
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
   
    

    return val_loss, val_accuracy, val_auroc, val_auprc, val_true, val_proba




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



