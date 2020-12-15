#!/usr/bin/env python3
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    # print(n,d)
    data = {}
    # sum = 0
    for line in fin:
    	# sum += 1
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        # data[tokens[0]] = []
        # for token in tokens[1:]:
        # 	data[tokens[0]].append(token)
    # fin.close()
    # print(sum)
    return data

def get_size(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    # print(n,d)
    data = {}
    # sum = 0
    total_size = 0
    for line in fin:
    	# sum += 1
        tokens = line.rstrip().split(' ')
        total_size += 49 + len(tokens[0])
       	total_size += 2536 
    fin.close()
    # print(sum)
    return total_size





# # data = load_vectors('ud_embeddings/ud_basic.vec')
# import csv
# missing_loc = 0
# missing_keyword = 0
# total_size = 0
# with open('train.csv', mode='r') as csvf:
#   reader = csv.reader(csvf, delimiter=',', quotechar='"')
#   for row in reader:
#     if row[1] == '':
#     	missing_keyword +=1
#     if row[2] == '':
#     	missing_loc +=1
#     total_size +=1
# print(missing_keyword,missing_loc, total_size)