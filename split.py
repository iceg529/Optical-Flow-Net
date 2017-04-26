
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
	j = 0
	batInd = 1
	X_data = np.empty([8, 8, 384, 512])
	for i in trainNum:
		print(i)
		X_data[j, :3] = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		X_data[j, [3,4,5]] = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		X_data[j, 6:] = np.asarray(readFloToFile(str(flowList[i].split()[0])))
		if j == 7:
			print(X_data.shape)
			dSetName = "/data"+str(batInd)
			f.create_dataset(dSetName, data=X_data, compression='gzip', compression_opts=9)
			j = 0
		else:
			j = j+1
		batInd = batInd +1 #remember to shift this to abov if statement to fix the key name issues
	print batInd
	
  	f.close()

def writeTestData():
	print('writing to file ...')
	f = h5py.File("testData.h5", 'r+')
	img1List = open("Test_Img1.list").readlines()
	img2List = open("Test_Img2.list").readlines()
	flowList = open("Test_Flow.list").readlines()
	j = 0
	batInd = 1
	X_data = np.empty([640, 8, 384, 512])
	for i in testNum:
		print(i)
		X_data[j, :3] = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		X_data[j, [3,4,5]] = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		X_data[j, 6:] = np.asarray(readFloToFile(str(flowList[i].split()[0])))
		if j == 639:
			print(X_data.shape)
			dSetName = "/data"+str(batInd)
			f.create_dataset(dSetName, data=X_data, compression='gzip', compression_opts=9)
			j = 0
		else:
			j = j+1
	
  	f.close()

def sampleForColorCoding():
	print('writing to file ...')
	f = h5py.File("FreshCode/sampleForColorCoding2.h5", 'w')
	data1 = np.asarray(readFloToFile('FreshCode/Evaluate/flownets-pred-0000000.flo'))
	f.create_dataset("/data", data=data1)

#writeTestData()
#writeTrainData()
sampleForColorCoding()





