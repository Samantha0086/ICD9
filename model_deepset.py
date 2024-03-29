
import torch, sys, pdb


#import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import torch.optim as optim

class Word2Vec_neg_sampling(nn.Module):

    def __init__(self, embedding_size, vocab_size, device, noise_dist = None, negative_samples = 10,
                 embeddings=None,
                 freeze_embedding=False,
                 filter_sizes=[3, 4, 5], #
                 num_filters=[50, 50, 50],
                 num_classes=2,
                 dropout=0.3):
        super(Word2Vec_neg_sampling, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device
        self.noise_dist = noise_dist

        # Initialize both embedding tables with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1,1)
        self.embeddings_context.weight.data.uniform_(-1,1)
        
      
        self.deepset = True
        
        
        # deepset
        self.fc_1 = nn.Linear(embedding_size, 128)
        #self.fc_2 = nn.Linear(embedding_size-64, embedding_size//2)
        #self.fc_3 = nn.Linear(embedding_size//2, embedding_size//4)
        self.relu = nn.ReLU()
     

        #self.embedding = nn.Embedding.from_pretrained(self.embeddings_input.weight.data,freeze=freeze_embedding)

        
         # Conv Network
        #self.conv1d_list = nn.ModuleList([
            #nn.Conv1d(in_channels=self.embed_dim,
                      #out_channels=num_filters[i],
                      #kernel_size=filter_sizes[i])
           # for i in range(len(filter_sizes))
       # ])
        # Fully-connected layer and Dropout
        self.fc1 = nn.Linear(128+9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        #self.fc4 = nn.Linear(32, 16)
        #self.fc5 = nn.Linear(16, 8)
        #self.fc6 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=dropout) # tune

    def forward(self, input_word, context_word,input_ids, other_features):
        debug =  not True
        if debug:
            print('input_word.shape: ', input_word.shape)        # bs
            print('context_word.shape: ', context_word.shape)    # bs

        # computing out loss
        emb_input = self.embeddings_input(input_word)     # bs, emb_dim
       

        emb_context = self.embeddings_context(context_word)  # bs, emb_dim
        

        emb_product = torch.mul(emb_input, emb_context)     # bs, emb_dim
       
        
        emb_product = torch.sum(emb_product, dim=1)          # bs


        out_loss = F.logsigmoid(emb_product)                      # bs


        # computing negative loss
        if self.noise_dist is None:
            noise_dist = torch.ones(self.vocab_size)  
        else:
            noise_dist = self.noise_dist

        if debug:print('noise_dist.shape: ', noise_dist.shape)
        
        num_neg_samples_for_this_batch = context_word.shape[0]*self.negative_samples
        negative_example = torch.multinomial(noise_dist, num_neg_samples_for_this_batch, replacement = True) # coz bs*num_neg_samples > vocab_size
        if debug:print('negative_example.shape: ', negative_example.shape)

        negative_example = negative_example.view(context_word.shape[0], self.negative_samples).to(self.device) # bs, num_neg_samples
        if debug:print('negative_example.shape: ', negative_example.shape)

        emb_negative = self.embeddings_context(negative_example) # bs, neg_samples, emb_dim
        if debug:print('emb_negative.shape: ', emb_negative.shape)

        if debug:print('emb_input.unsqueeze(2).shape: ', emb_input.unsqueeze(2).shape) # bs, emb_dim, 1
        emb_product_neg_samples = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2)) # bs, neg_samples, 1
        if debug:print('emb_product_neg_samples.shape: ', emb_product_neg_samples.shape)

        noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1) # bs
        if debug:print('noise_loss.shape: ', noise_loss.shape)

        total_loss = -(out_loss + noise_loss).mean()
        if debug:print('total_loss.shape: ', total_loss.shape)
        
    
        
        
        
            # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
            # Output shape: (b, embed_dim, max_len)
            #x_reshaped = x_embed.permute(0, 2, 1)
    
            # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
            #x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
    
            # Max pooling. Output shape: (b, num_filters[i], 1)
            #x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
            
            # Concatenate x_pool_list to feed the fully connected layer.
            # Output shape: (b, sum(num_filters))
            #x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
            
            # Concatenate with other features 
            # Output shape: (b, sum(num_filters + number of other features)
            #x_fc =torch.cat((x_fc, other_features ), dim = 1) 
        
        
        self.ds = self.relu(self.fc_1(self.embeddings_input.weight.data))
        
            
        self.ds_embedding = nn.Embedding.from_pretrained(self.relu(self.fc_1(self.embeddings_input.weight.data)),freeze=True)
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.ds_embedding(input_ids).float()
        
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        
        #  Output shape: (b,  embed_dim,1)
        x_embed = torch.sum(x_reshaped, axis =2)
        
        # Concatenate with other features 
        # Output shape: (b, sum(num_filters + number of other features)
        x_fc =torch.cat((x_embed, other_features ), dim = 1) 
        
        
        x = (self.fc1(x_fc))
        x = self.fc2(x)
        x = self.dropout(self.fc3(x))
        #x = self.fc4(x)
        #x = self.dropout(self.fc5(x))
        # Compute logits. Output shape: (b, n_classes)
        #logits = self.fc6(x)
        

        return total_loss, x
           


      
