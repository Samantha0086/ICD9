import os
import warnings
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')



feature_columns = ['LOS', 'AGE', 'GENDER_M', 'ETHNICITY_Asian', 'ETHNICITY_Black', 'ETHNICITY_Hispanic', 'ETHNICITY_Native_Hawaiian', 'ETHNICITY_Other', 'ETHNICITY_White']
data = pd.read_csv('admit_modified.csv', index_col=False)
data.dropna(subset = feature_columns, inplace = True)
data.reset_index(drop=True, inplace = True)
data['codes'] =  data['PROCEDURE_AND_DIAGNOSIS_ICD'].apply(lambda x: x.replace("'", "")[1:-1].split(", "))
X = data.drop(['MORTALITY_30_DAY'], axis = 1)
y = data['MORTALITY_30_DAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
X_train.reset_index(drop=True, inplace = True)
X_test.reset_index(drop=True, inplace = True)
y_train.reset_index(drop=True, inplace = True)
y_test.reset_index(drop=True, inplace = True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# train skip-gram model
window_size = 71 # max(X_train['codes'].apply(lambda x: len(x))) = 71
skipgram = Word2Vec(vector_size=100, window=window_size, min_count=1, sg=1)
skipgram.build_vocab(data['codes'])
skipgram.train(X_train['codes'], total_examples=skipgram.corpus_count, epochs=skipgram.epochs)
# Extract embeddings
embeddings = {word: skipgram.wv[word] for word in skipgram.wv.index_to_key}




def codes_to_emb(codes):
    return np.mean(np.array([embeddings.get(c) for c in codes]), axis=0)

X_train['embedding'] = X_train['codes'].apply(lambda x: codes_to_emb(x))
X_test['embedding'] = X_test['codes'].apply(lambda x: codes_to_emb(x))
X_train = pd.concat([X_train['embedding'].apply(pd.Series), X_train[feature_columns]], axis=1)
X_test = pd.concat([X_test['embedding'].apply(pd.Series), X_test[feature_columns]], axis=1)




class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        auroc_train = roc_auc_score(self.y_train, y_train_pred)
        auroc_test = roc_auc_score(self.y_test, y_test_pred)
        auprc_train = average_precision_score(self.y_train, y_train_pred)
        auprc_test = average_precision_score(self.y_test, y_test_pred)
        print(f"\nEpoch {epoch+1}: AUROC Train: {auroc_train}, AUROC Test: {auroc_test}, "
              f"AUPRC Train: {auprc_train}, AUPRC Test: {auprc_test}")

def cnn(X_train, y_train, X_test, y_test):
    # Build and compile the neural network
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=109))  # 100 embedding dims + 9 other features
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(8))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Suitable for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Initialize metrics callback
    metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test)

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[metrics_callback])

    return model

# Example usage
model = cnn(X_train, y_train, X_test, y_test)
