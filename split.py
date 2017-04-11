
import h5py
from scipy import misc
import numpy as np
import os 
import sys
import random
from sys import getsizeof

trainNum = list(range(22232))
random.shuffle(trainNum)
testNum = list(range(640))
random.shuffle(testNum)

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def readFloToFile(floFile):
	with open(floFile, 'rb') as f:
	    magic = np.fromfile(f, np.float32, count=1)
	    if 202021.25 != magic:
		print 'Magic number incorrect. Invalid .flo file'
	    else:
		w = np.fromfile(f, np.int32, count=1)
		h = np.fromfile(f, np.int32, count=1)
		print 'Reading %d x %d flo file' % (w, h)
		data = np.fromfile(f, np.float32, count=2*w*h)
		# Reshape data into 3D array (columns, rows, bands)
		data2D = np.resize(data, (2, h, w))
		return data2D


def writeToFile(listName,flag1):
	print('writing to file ...')
	imgList = open(listName).readlines()
	X_data = []	
	for i in trainNum:
		print(i)
		imgToRead = str(imgList[i].split()[0])
		print(imgToRead)
		if flag1 == 'true':
			X_data.append(misc.imread(imgToRead))
		else:
			X_data.append(readFloToFile(imgToRead))
	
  	if flag1 == 'true':
    		return np.transpose(np.asarray(X_data), (0,3,1,2))
	else:
		return np.asarray(X_data)

def writeTestData():
	f = h5py.File("testData.h5", 'w')		    

	imData1 = writeToFile("Test_Img1.list",'true')
	#f.create_dataset("img1", data=finalData, compression='gzip', compression_opts=9)

	imData2 = writeToFile("Test_Img2.list",'true')
	#f.create_dataset("img2", data=finalData, compression='gzip', compression_opts=9)

	flowData = writeToFile("Test_Flow.list",'false')
	#f.create_dataset("flow", data=finalData, compression='gzip', compression_opts=9)
  	finalData = np.concatenate((imData1, imData2, flowData), axis=1)
  	print(finalData.shape)
  	f.create_dataset("data", data=finalData, compression='gzip', compression_opts=9)
	f.close()

def writeTrainData():
	f = h5py.File("trainData1.h5", 'w')		    

	imData1 = writeToFile("Train_Img1.list",'true')
#	f.create_dataset("img1", data=finalData, compression='gzip', compression_opts=9)

	imData2 = writeToFile("Train_Img2.list",'true')
#	f.create_dataset("img2", data=finalData, compression='gzip', compression_opts=9)
	
	finalData = np.concatenate((imData1, imData2), axis=1)
	print(finalData.shape)
#	f.create_dataset("data1", data=finalData, compression='gzip', compression_opts=9)

	flowData = writeToFile("Train_Flow.list",'false')
	
	finalData2 = np.concatenate((finalData, flowData), axis=1)
	print(finalData2.shape)
	
	f.create_dataset("data", data=finalData2, compression='gzip', compression_opts=9)
	  	
	f.close()

#writeTestData()
writeTrainData()





