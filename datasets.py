
#import os, glob, cv2, sys, torch, pdb, random
from torch.utils.data import Dataset

import torch
import numpy as np
import time
from array import array
import pickle  
from itertools import permutations, chain


class word2vec_dataset(Dataset):
    def __init__(self, TYPE, data, train_index, test_index):
        self.data = data[TYPE].dropna().reset_index().drop(columns = ["index"])[TYPE]
        del(data)
   
        self.vocab, self.word_to_ix  = self.gather_word_freqs(self.data) 
        
        self.train_index = train_index
        self.test_index = test_index
        
        self.training_data,self.testing_data, self.tokenized_data, max_code_count = self.gather_training_data(self.data, self.word_to_ix)
        
        
        self.code_same_len = self.encode(self.tokenized_data, self.word_to_ix, max_code_count )
        
        # training_data is a list of list of 2 indices
      
        
    def __getitem__(self, index):  
        x = self.data[index, 0]
        y = self.data[index, 1]
        return x, y

    def __len__(self):
        return len(self.data)              
    
    def gather_training_data(self, tokenized_data, word_to_ix):       
        print("gather training data")
        tic = time.perf_counter()
    

        training_data = []
        max_code_count =  tokenized_data.str.len().max()

        
        tokenized_data = tokenized_data.map(lambda x: [word_to_ix[y] for y in x if y in word_to_ix])
        
        

        training_data = list(chain.from_iterable([list(permutations(sentence,2)) for sentence in tokenized_data[self.train_index]]))
        training_data = torch.tensor(training_data, dtype = torch.long)
        
        testing_data = list(chain.from_iterable([list(permutations(sentence,2)) for sentence in tokenized_data[self.test_index]]))
        testing_data = torch.tensor(testing_data, dtype = torch.long)
            
                        
           
        
        #all_vocab_indices = list(range(len(word_to_ix)))
        toc = time.perf_counter()
        
        print(f"gather training data takes {toc - tic:0.4f} seconds")
         
        # data is the tokenized data
        
        return training_data, testing_data, tokenized_data, max_code_count
            
    def load_data(self):
        

        vocab, word_to_ix  = self.gather_word_freqs(self.data) 
        
        
        training_data, testing_data, tokenized_data, max_code_count = self.gather_training_data(self.data, word_to_ix)
        
        self.data = []
        
        padding = self.encode(tokenized_data, word_to_ix, max_code_count )
        
        return vocab, word_to_ix,  training_data, testing_data, padding

    def gather_word_freqs(self, data): #here split_text is sent_list
        
        print("gather_word_freqs")
        tic = time.perf_counter()
        
        #vocab = {"<pad>":1}
        vocab =[1]
 
        word_to_ix = {'<pad>':0}
        #total = 0.0

 
        
        for i in range(len(data)):
            for j in range(len(data[i])):
            #for every word in the word list(split_text), which might occur multiple times
                code = data[i][j]
                if code not in word_to_ix: #only new words allowed
                    vocab.append(0)
                    
                    word_to_ix[code] = len(vocab)-1
                vocab[word_to_ix[code]] += 1 #count of the word stored in a dict
                #total += 1.0 #total number of words in the word_list(split_text)
        

        toc = time.perf_counter()
        print(f"gather word freqs takes {toc - tic:0.4f} seconds")
        

    
        return vocab, word_to_ix
    
    def encode(self, tokenized_texts, word2idx, max_len):
        """Pad each sentence to the maximum sentence length and encode tokens to
        their index in the vocabulary.

        Returns:
            input_ids (np.array): Array of token indexes in the vocabulary with
                shape (N, max_len). It will the input of our CNN model.
        """
        
        print("encode beginning")
        tic = time.perf_counter()

        input_ids = []
       
        for i in range(len(tokenized_texts)):
            tokenized_sent = tokenized_texts[i].copy()
            # Pad sentences to max_len
            
            tokenized_sent += [0] * (max_len - len(tokenized_sent))
            
            input_ids.append(tokenized_sent)
            
        input_ids = np.array( input_ids)
            
        toc = time.perf_counter()
        print(f"encode takes {toc - tic:0.4f} seconds")
        

            
        return input_ids