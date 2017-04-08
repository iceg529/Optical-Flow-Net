
import h5py
from scipy import misc
import numpy as np
import os 
import sys
from sys import getsizeof

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
	
	for i, x in enumerate(imgList):
		print(i)
		imgToRead = str(x.split()[0])
		if flag1 == 'true':
			X_data.append(misc.imread(imgToRead))
		else:
			X_data.append(readFloToFile(imgToRead))

	finalData = np.asarray(X_data)
  	if flag1 == 'true':
    		finalData = np.transpose(finalData, (0,3,1,2))
    	print(getsizeof(finalData))
	return finalData

def writeTestData():
	f = h5py.File("testData.h5", 'w')		    

	finalData = writeToFile("Test_Img1.list",'true')
	f.create_dataset("img1", data=finalData, compression='gzip', compression_opts=9)

	finalData = writeToFile("Test_Img2.list",'true')
	f.create_dataset("img2", data=finalData, compression='gzip', compression_opts=9)

	finalData = writeToFile("Test_Flow.list",'false')
	f.create_dataset("flow", data=finalData, compression='gzip', compression_opts=9)

	f.close()

def writeTrainData():
	f = h5py.File("trainData1.h5", 'w')		    

#	finalData = writeToFile("Train_Img1.list",'true')
#	f.create_dataset("img1", data=finalData, compression='gzip', compression_opts=9)

#	finalData = writeToFile("Train_Img2.list",'true')
#	f.create_dataset("img2", data=finalData, compression='gzip', compression_opts=9)

	finalData3 = writeToFile("Train_Flow.list",'false')
	f.create_dataset("flow", data=finalData3, compression='gzip', compression_opts=9)

	f.close()

#writeTestData()
writeTrainData()





