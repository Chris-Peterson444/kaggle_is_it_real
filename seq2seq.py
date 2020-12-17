#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data import Field, BucketIterator
import torchtext

import spacy

import sys, os, io, random, math
import time
import numpy as np
import pandas as pd
# from jupyterplot import ProgressPlot
import matplotlib.pyplot as plt


# Where to save
location = 'new/slang/'

# Defining some of the hyperparameters here so I can use them in the check
ENC_EMB_DIM = 300
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5

FC_IN_DIM = HID_DIM * N_LAYERS * 2 # CELL and HIDDEN for each layer
# INTERMEDIATE_DIM = 25 # See what works best here


# # Tokenizer Setup
# I am using spaCy to tokenize since it's a little more robust than the default pytorch tokenizer
# Seq2Seq reverses the input sentences, but since this techincally is more semantic analysis
# than sequence to sequence, we're going to leave it in the forward order.

# In[3]:


spacy_en = spacy.load('en')


# In[4]:


def tokenize_input_lang(text):
    '''
    Tokenize input from a string into a list of tokens and reverses it
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]


# ## Fields setup

# In[5]:


def stoi(x):
    return [int(x[0])]

TEXT = Field(tokenize = tokenize_input_lang, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

LABEL = Field(dtype = torch.int, use_vocab=False, is_target=True, preprocessing=stoi)
# The original file contains: id    keyword    location    target
# However I haven't decided how I want to include the keywords in this model yet (probably concatentaiton)
# So for now it is being trained without
#
# To train with
fields = [(None, None), (None, None), (None,None), ('text', TEXT), ('label', LABEL)]


# # Importing and Loading Data To Use In PyTorch

# In[6]:


twitter_dataset = torchtext.data.TabularDataset('train.csv','csv',fields,skip_header=True)


# In[7]:


# torchtext.data.split() returns default a 70-30 split for training-testin
# but since testing is provided by kaggle we will treat this as our 
# training-validation split
train_data, valid_data = twitter_dataset.split()


# ## Double check we've loaded the right number and split correctly

# In[8]:


print(f"Number of total examples: {len(twitter_dataset.examples)}")
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")


# ## Example of one of the training data (Tokenized correctly and reversed (?) )

# In[9]:


print(vars(train_data.examples[1]))
print(type(vars(train_data.examples[1])['label'][0]))


# ## Slang Embeddings setup

# In[10]:


# def load_embeddings(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = list(map(float, tokens[1:]))
#     return data

# slang_emb = load_embeddings('ud_embeddings/ud_basic.vec')
slang_emb = torchtext.vocab.Vectors(name = '../chris_nlp_data/ud_embeddings/ud_basic.vec',
                                   cache = '../chris_nlp_data/ud_embeddings',
                                   unk_init = torch.Tensor.normal_)


# ### Some helpful functions to verify things are working correctly

# Next block from https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/B%20-%20A%20Closer%20Look%20at%20Word%20Embeddings.ipynb#scrollTo=DMkoy7iFMeN3

# In[11]:


def get_vector(embeddings, word):
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]

def closest_words(embeddings, vector, n = 10):
    
    distances = [(word, torch.dist(vector, get_vector(embeddings, word)).item())
                 for word in embeddings.itos]
    
    return sorted(distances, key = lambda w: w[1])[:n]

def print_tuples(tuples):
    for w, d in tuples:
        print(f'({d:02.04f}) {w}') 
        

def analogy(embeddings, word1, word2, word3, n=5):
    
    #get vectors for each word
    word1_vector = get_vector(embeddings, word1)
    word2_vector = get_vector(embeddings, word2)
    word3_vector = get_vector(embeddings, word3)
    
    #calculate analogy vector
    analogy_vector = word2_vector - word1_vector + word3_vector
    
    #find closest words to analogy vector
    candidate_words = closest_words(embeddings, analogy_vector, n+3)
    
    #filter out words already in analogy
    candidate_words = [(word, dist) for (word, dist) in candidate_words 
                       if word not in [word1, word2, word3]][:n]
    
    print(f'{word1} is to {word2} as {word3} is to...')
    
    return candidate_words


# In[12]:


# print_tuples(analogy(slang_emb, 'man', 'actor', 'woman'))


# ## Build Vocab

# In[13]:


TEXT.build_vocab(twitter_dataset,
                vectors = slang_emb)
LABEL.build_vocab(train_data)


# In[14]:


print(f"Unique tokens in text vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in label vocabulary: {len(LABEL.vocab)}")


# In[15]:


print(LABEL.vocab.freqs)
print(LABEL.vocab.itos)


# In[16]:


b = TEXT.vocab.vectors[TEXT.vocab.stoi['man']]
a = get_vector(slang_emb,'man')
print (a - b)
# If loaded correctly all should be 0



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell


# ## Fully Connected
# 
# Small fully connected layer to help with encapsulation

# In[20]:


class FullyConnected(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, intermediate_dim)
        self.activation = nn.functional.relu
        self.fc_out = nn.Linear(intermediate_dim, output_dim)
    
    def forward(self, input_x):
        
        x = self.fc_in(input_x.squeeze(0))
        x = self.activation(x)
        x = self.fc_out(x)
        return x


# ## Model class

# In[89]:


class CustomModel(nn.Module):
    def __init__(self, encoder, fc, device):
        super().__init__()
        
        self.encoder = encoder
        self.fc = fc
        self.device = device
        
    def forward(self, src):
        
        hidden, cell = self.encoder(src)
        fc_input = torch.cat((hidden[0],hidden[1],cell[0],cell[1]),1)
        output = self.fc(fc_input)
       
        return output
     
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
#         print(batch)
        src = batch.text
#         print("src",batch.text)
        trg = batch.label
#         print("trg",batch.label)
        optimizer.zero_grad()
        
        output = model(src)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
#         print(output.shape)
#         print(trg.shape)
#         print(output)
#         print("target")
#         print(trg)
        output_dim = output.shape[-1]
#         print(output_dim)
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
#         print(len(trg))
        trg = trg.long()
        #trg = [(trg len ) * batch size]
        #output = [(trg len ) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# # Evaluation loop

# In[126]:


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.text
            trg = batch.label

            output = model(src)

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            trg = trg.long()

            #trg = [(trg len ) * batch size]
            #output = [(trg len ) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# Calculate how long an epoch takes

# In[127]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM =  len(LABEL.vocab) - 2
# Loop time

BATCH_SIZES = [3,4,5,10,15,20,25,50,52,64,128,256,512]
INNER_DIM = [256,128,64,32]

for batch_size in BATCH_SIZES:
    for inner_dim in INNER_DIM:

        BATCH_SIZE = batch_size
        INTERMEDIATE_DIM = int(inner_dim)
        N_EPOCHS = 30

        train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), 
            batch_size = BATCH_SIZE,
            device = device,
            sort_key = lambda x: len(x.text),
            sort_within_batch=False)


        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        fc = FullyConnected(FC_IN_DIM,INTERMEDIATE_DIM,OUTPUT_DIM)

        embed_weights = TEXT.vocab.vectors
        enc.embedding.weight.data.copy_(embed_weights)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        enc.embedding.weight.data[UNK_IDX] = torch.zeros(ENC_EMB_DIM)
        enc.embedding.weight.data[PAD_IDX] = torch.zeros(ENC_EMB_DIM)

        #freeze embeddings
        enc.embedding.weight.requires_grad = False

        model = CustomModel(enc,fc, device ).to(device)

        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss().to(device)
       
        file_name = str(N_EPOCHS)+"-epochs-"+str(BATCH_SIZE)+"-batch-"+str(INTERMEDIATE_DIM)+"-dim"
        training_loss_data = []
        validation_loss_data = []
        CLIP = 1
        f = open(location+"log_data/"+file_name+".txt","w")
        best_valid_loss = float('inf')
        print("running "+file_name)

        # Train

        for epoch in range(N_EPOCHS):
            
            start_time = time.time()
            
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            validation_loss_data.append(valid_loss)
            training_loss_data.append(train_loss)
            # pp.update([[train_loss,valid_loss]])
            
            #save model

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), location+"models/"+file_name+'.pt')
            
            #save log info

            f.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
            f.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
            f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n')
        f.close()

        # Save plot
        # print(training_loss_data)
        x_data = []
        for v in range(N_EPOCHS):
            x_data.append(v+1)
        plt.plot(x_data,validation_loss_data)
        plt.plot(x_data,training_loss_data)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend(['valid','train'], loc="upper right")
        plt.savefig(location+"loss_plots/"+file_name+'.png')
        plt.clf() # clear plot for next iteration

        #save loss in easily readable way
        f = open(location+"loss_data/"+file_name+".txt", "w")
        # write validation loss as an array
        for loss in validation_loss_data:
            f.write(f'{loss:.3f}')
            f.write(', ')
        f.write('\n')
        # write train loss
        for loss in training_loss_data:
            f.write(f'{loss:.3f}')
            f.write(', ')
        f.write('\n')
        f.close()



