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


def getLoss(file_name):
	global validation_loss, training_loss
	# empty arrays
	validation_loss = []
	training_loss = []
	f = open("slang/loss_data/"+file_name, "r")
	s = f.readline()
	s = s.split(', ')
	s = s[:-1]
	for v in s:
		validation_loss.append(v)
	s = f.readline()
	s = s.split(', ')
	s = s[:-1]
	for v in s:
		training_loss.append(v)

files = [f for f in listdir("slang/loss_data") if isfile(join("slang/loss_data", f))]

for file in files:
	getLoss(file)
	x_data = []
	for v in range(len(validation_loss)):
	    x_data.append(v+1)
	plt.plot(x_data,validation_loss)
	plt.plot(x_data,training_loss)
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.legend(['valid','train'], loc="upper right")
	plt.savefig("slang/loss_plots/"+file.replace('.txt','')+'.png')
	plt.clf()