# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2020-12-16 10:15:43
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2020-12-16 10:46:17
import sys, os, io, time
import numpy as numpy
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join

validation_loss = []
training_loss = []
min_file = {}
top_number = 5

run = 'new/slang'

def getLoss(file_name):
	global validation_loss, training_loss
	# empty arrays
	validation_loss = []
	training_loss = []
	f = open(run+"/loss_data/"+file_name, "r")
	s = f.readline()
	s = s.split(', ')
	s = s[:-1]
	for v in s:
		validation_loss.append(float(v))


files = [f for f in listdir(run+"/loss_data") if isfile(join(run+"/loss_data", f))]

for file in files:
	getLoss(file)
	min_file[file] = min(validation_loss)
sorted_dict = sorted(min_file.items(), key= lambda item: item[1])
sorted_dict.reverse()
for v in sorted_dict:
	print(v)

	