import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data import Field, BucketIterator
import torchtext

import spacy

from os import listdir
from os.path import isfile, join

import sys, os, io, random, math
import time
import numpy as np
import pandas as pd
from jupyterplot import ProgressPlot
import matplotlib.pyplot as plt


spacy_en = spacy.load('en')

def tokenize_input_lang(text):
    '''
    Tokenize input from a string into a list of tokens and reverses it
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]

def stoi(x):
    return [int(x[0])]

TEXT = Field(tokenize = tokenize_input_lang, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

LABEL = Field(dtype = torch.int, use_vocab=False, is_target=True, preprocessing=stoi)
ID = Field(dtype = torch.int, use_vocab=False, preprocessing=stoi)
# The original file contains: id    keyword    location    target
# However I haven't decided how I want to include the keywords in this model yet (probably concatentaiton)
# So for now it is being trained without
#
# To train with
fields = [('id', ID), (None, None), (None,None), ('text', TEXT), ('label', LABEL)]
twitter_dataset = torchtext.data.TabularDataset('train.csv','csv',fields,skip_header=True)
train_data, valid_data = twitter_dataset.split()
print(f"Number of total examples: {len(twitter_dataset.examples)}")
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(vars(train_data.examples[1]))
print(type(vars(train_data.examples[1])['label'][0]))

slang_emb = torchtext.vocab.GloVe(name = '6B', dim = 300)

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
     


def getPrecision(model, iterator):
    
    # print('slow down so we dont crash')
    precision = 0
    true_pos = 0
    false_pos = 0

    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            # print(i)
            src = batch.text
            trg = batch.label
            
#             number = batch.id
            # print(src.shape)
            output = model(src)
            # print(output.shape[0])
            # if output.shape[0] == 48:
            # 	print(output)
#             for word in src[:,0]:
#                 print(TEXT.vocab.itos[word.item()])
#             print(number[0][0].item())
#             print(number[0].shape[0])
#             print(output.shape)

            for j in range(output.shape[0]):
#                 print(output[i,:].cpu().numpy())
#                 print(output.shape)
                predicted = np.argmax(output[j,:].cpu().numpy())
                expected = trg[0,j].item()
                # If it was negative
                if expected == 0:
                    # if false positive
                    if predicted == 1:
                        false_pos = false_pos +1
                
                # if it was positive
                elif expected == 1:
                    # if true pos
                    if predicted == 1:
                        true_pos = true_pos + 1

                
    if (true_pos + false_pos) == 0:
    	precision = -1
    	# print("Something went wrong")
    else:
    	precision = true_pos/(true_pos + false_pos)
                
                
            

            #trg = [(trg len ) * batch size]
            #output = [(trg len ) * batch size, output dim]
    return precision

def getRecall(model, iterator):
            
    model.eval()
    
    recall = 0
    true_pos = 0
    false_neg = 0
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            src = batch.text
            trg = batch.label
            
#             number = batch.id
            
            output = model(src)
#             for word in src[:,0]:
#                 print(TEXT.vocab.itos[word.item()])
#             print(number[0][0].item())
#             print(number[0].shape[0])
#             print(output.shape)
            for j in range(output.shape[0]):
#                 print(output[i,:].cpu().numpy())
#                 print(output.shape)
                predicted = np.argmax(output[j,:].cpu().numpy())
                expected = trg[0,j].item()
                
                # if it was positive
                if expected == 1:
                    # if true pos
                    if predicted == 1:
                        true_pos = true_pos + 1
                    # if false neg
                    elif predicted == 0:
                        false_neg = false_neg + 1
                
    if (true_pos + false_neg) == 0:
    	recall = -1
    else:
    	recall = true_pos/(true_pos + false_neg)
                
                
    return recall

def getF1(precision, recall):
    return 2*(precision*recall)/(precision+recall)



TEXT.build_vocab(twitter_dataset,
                vectors = slang_emb)
LABEL.build_vocab(train_data)
print(f"Unique tokens in text vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in label vocabulary: {len(LABEL.vocab)}")
print(LABEL.vocab.freqs)
print(LABEL.vocab.itos)
b = TEXT.vocab.vectors[TEXT.vocab.stoi['man']]
a = get_vector(slang_emb,'man')
print (a - b)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_iterator, valid_iterator = BucketIterator.splits(
	    (train_data, valid_data), 
	    batch_size = 52,
	    device = device,
	    sort_key = lambda x: len(x.text),
	    sort_within_batch=False)

location = "new/glove/"

files = [f for f in listdir(location+"models/") if isfile(join(location+"models/", f))]

for f in files:


	file_string = f.split('-')
	# batch_size = int(file_string[2])
	inner_dim = int(file_string[4])
	print('running '+f)
	# print(batch_size)
	print(inner_dim)
	print(len(valid_iterator))




	INPUT_DIM = len(TEXT.vocab)
	OUTPUT_DIM =  len(LABEL.vocab) - 2
	ENC_EMB_DIM = 300
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5

	FC_IN_DIM = HID_DIM * N_LAYERS * 2 # CELL and HIDDEN for each layer
	INTERMEDIATE_DIM = inner_dim # See what works best here

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

	model = CustomModel(enc,fc, device )
	model.load_state_dict(torch.load(location+'/models/'+f))
	model.eval()
	model.to(device)

	# print(f'The model has {count_parameters(model):,} trainable parameters')

	# optimizer = optim.Adam(model.parameters())

	# criterion = nn.CrossEntropyLoss().to(device)

	print("model loaded")
	print('getting precision')
	precision =  getPrecision(model,valid_iterator)
	print('finished getting precision')
	print('getting recall')
	recall = getRecall(model, valid_iterator)
	print('finished getting recall')
	f1_score = getF1(precision, recall)

	with open(location+'f1/'+f.replace('.pt','-precision.txt'),'w') as file:
		file.write("precision "+str(precision)+'\n')
		file.write('recall '+str(recall)+'\n')
		file.write('f1 '+str(f1_score)+'\n')
	print('deleting model')
	del model
	# time.sleep(3)