
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


def writeTrainData():
	print('writing to file ...')
	f = h5py.File("trainData1.h5", 'r+')
	img1List = open("Train_Img1.list").readlines()
	img2List = open("Train_Img2.list").readlines()
	flowList = open("Train_Flow.list").readlines()
	for i in trainNum:
		print(i)
		img1ToRead = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		img2ToRead = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		flowToRead = np.asarray(readFloToFile(str(flowList[i].split()[0])))
		print(img1ToRead.shape)
		print(img2ToRead.shape)
		print(flowToRead.shape)
		X_data = np.concatenate((img1ToRead, img2ToRead, flowToRead), axis=0)
		print(X_data.shape)
		dSetName = "trainData/data"+str(i)
		f.create_dataset(dSetName, data=X_data, compression='gzip', compression_opts=9)
	
  	f.close()

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


#writeTestData()
writeTrainData()





