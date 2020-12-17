# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2020-12-16 10:15:43
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2020-12-16 10:46:17
import sys, os, io, time
import numpy as numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
# each index is a different file
precision = []
recall = []
f1 = []
file_names = []
batch_size = []
inner_dim = []

unique_batches = []
unique_dim = []

embed = 'glove'
location = "new/" + embed


def getf1(file_name):
	global precision, recall, f1, batch_size, inner_dim
	# empty arrays

	f = open(location+"/f1/"+file_name, "r")
	s = f.readline()
	s = s.split(' ')
	# check if a bad sample
	if float(s[1]) == -1:
		return
	precision.append(float(s[1]))
	s = f.readline()
	s = s.split(' ')
	recall.append(float(s[1]))
	s = f.readline()
	s = s.split(' ')
	f1.append(float(s[1]))
	f.close()
	file_string = file.split('-')
	batch_size.append(int(file_string[2]))
	inner_dim.append(int(file_string[4]))
	file_names.append(file_name)
	
	
	
	
	


files = [f for f in listdir(location+"/f1/") if isfile(join(location+"/f1/", f))]

for file in files:
	getf1(file)


# get unique batches and inner dims
for i in range(len(file_names)):
	if batch_size[i] not in unique_batches:
		unique_batches.append(batch_size[i])
	if inner_dim[i] not in unique_dim:
		unique_dim.append(inner_dim[i])


# print(len(file_names))
a = len(unique_batches)
b = len(unique_dim)

sorted_batch = sorted(unique_batches)
sorted_dim = sorted(unique_dim)

# print(a, b, a*b)

# make f1 table
f1_table = np.zeros((a,b))
for i in range(len(file_names)):
	f1_table[sorted_batch.index(batch_size[i]),sorted_dim.index(inner_dim[i])] = f1[i]
# print(f1_table)
# f1_table = (f1_table-f1_table.mean())/f1_table.std()
ax = sns.heatmap(f1_table, cmap=sns.cm.rocket_r, xticklabels=sorted_dim, yticklabels=sorted_batch) #, yticklabels=sorted_dim, xticklabels=sorted_batch)
plt.xlabel('hidden layer dimension')
plt.ylabel('batch size')
plt.title(embed+": f1 score heatmap")
plt.savefig(location+'/'+embed+'-f1-heatmap.png')
plt.clf()

precision_table = np.zeros((a,b))
for i in range(len(file_names)):
	precision_table[sorted_batch.index(batch_size[i]),sorted_dim.index(inner_dim[i])] = precision[i]
# print(precision_table)
ax = sns.heatmap(precision_table, cmap=sns.cm.rocket_r, xticklabels=sorted_dim, yticklabels=sorted_batch) #, yticklabels=sorted_dim, xticklabels=sorted_batch)
plt.xlabel('hidden layer dimension')
plt.ylabel('batch size')
plt.title(embed+": precision heatmap")
plt.savefig(location+'/'+embed+'-precision-heatmap.png')
plt.clf()

recall_table = np.zeros((a,b))
for i in range(len(file_names)):
	recall_table[sorted_batch.index(batch_size[i]),sorted_dim.index(inner_dim[i])] = recall[i]
# print(precision_table)
ax = sns.heatmap(recall_table, cmap=sns.cm.rocket_r, xticklabels=sorted_dim, yticklabels=sorted_batch) #, yticklabels=sorted_dim, xticklabels=sorted_batch)
plt.xlabel('hidden layer dimension')
plt.ylabel('batch size')
plt.title(embed+": recall heatmap")
plt.savefig(location+'/'+embed+'-recall-heatmap.png')
plt.clf()


print('Best f1 score for '+embed,np.max(np.array(f1)))
name = np.argmax(np.array(f1))
print(name)
print(file_names[name])

# 	for j in range(len(inner_dim)):
# 		f1_table[i,j] = f1_



# # plot by unique dim:
# for inner in unique_dim:
# 	plot_data = [[],[],[]] # precision, recal, f1
# 	x_data = []
# 	for j in range(len(file_names)):
# 		if inner_dim[j] == inner:
# 			plot_data[0].append(precision[j])
# 			plot_data[1].append(recall[j])
# 			plot_data[2].append(f1[j])
# 			x_data.append(batch_size[j])

# 	plt.scatter(x_data, plot_data[2])
# 	plt.scatter(x_data, plot_data[0])
# 	plt.scatter(x_data, plot_data[1])
# 	plt.ylabel('metrics')
# 	# plt.legend(['precision', 'recall'], loc="upper right")
# 	plt.legend(['F1','precision','recall'], loc="upper right")
# 	plt.title('hidden layer dim: '+str(inner))
# 	plt.show()

# # plot by unique batch:
# for inner in unique_batches:
# 	plot_data = [[],[],[]] # precision, recal, f1
# 	x_data = []
# 	for j in range(len(file_names)):
# 		if batch_size[j] == inner:
# 			plot_data[0].append(precision[j])
# 			plot_data[1].append(recall[j])
# 			plot_data[2].append(f1[j])
# 			x_data.append(inner_dim[j])

# 	plt.scatter(x_data, plot_data[2])
# 	plt.scatter(x_data, plot_data[0])
# 	plt.scatter(x_data, plot_data[1])
# 	plt.ylabel('metrics')
# 	# plt.legend(['precision', 'recall'], loc="upper right")
# 	plt.legend(['F1','precision','recall'], loc="upper right")
# 	plt.title('batch_size: '+str(inner))
# 	plt.show()






# plt.plot(x_data,validation_loss)
# plt.plot(x_data,training_loss)
# plt.xlabel('')
# plt.ylabel('metrics')
# plt.legend(['F1','precision','recall'], loc="upper right")
# plt.savefig(location+"/f1_plots/"+file.replace('.txt','')+'.png')
# plt.clf()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# print(len(batch_size))
# print(len(inner_dim))
# print(len(f1))
# ax.plot_trisurf(X=batch_size,Y=inner_dim,Z=f1)
# plt.show()