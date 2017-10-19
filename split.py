
import h5py
from scipy import misc
import numpy as np
import os 
import sys
import random
import cPickle
from sys import getsizeof
import torchfile


trainNum = list(range(904*2)) #22232
random.shuffle(trainNum)
testNum = list(range(137*1))  #640
random.shuffle(testNum)

cPickle.dump(trainNum, open('trainNum.p', 'wb')) 
#obj = cPickle.load(open('save.p', 'rb'))

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def readFloToFile(floFile):
	with open(floFile, 'rb') as f:
	    magic = np.fromfile(f, np.float32, count=1)
	    if 202021.25 != magic:
		print 'Magic number incorrect. Invalid .flo file'
	    else:
		w = np.fromfile(f, np.int32, count=1)
		h = np.fromfile(f, np.int32, count=1)
		#print 'Reading %d x %d flo file' % (w, h)
		data = np.fromfile(f, np.float32, count=2*w*h)
		# Reshape data into 3D array (columns, rows, bands)
		data2D = np.reshape(data, (2, w, h), order='F') #data2D = np.resize(data, (2, h, w))
		return data2D


def writeTrainData():
	print('writing to file ...')
	f = h5py.File("FreshCode/trainData_Sintel.h5", 'w')
	img1List = open("Sintel_Train_Img1.list").readlines()
	img2List = open("Sintel_Train_Img2.list").readlines()
	flowList = open("Sintel_Train_Flow.list").readlines()
	j = 0
	batInd = 1
	X_data = np.empty([8, 8, 436, 1024]) #change to img size (448 not orig size , multiple of 32)
	for i in trainNum:
		print(i)
		temp = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		X_data[j, :3] = temp
		temp = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		X_data[j, [3,4,5]] = temp
		temp = np.transpose(np.asarray(readFloToFile(str(flowList[i].split()[0]))), (0,2,1))
		X_data[j, 6:] = temp
		if j == 7:
			print(X_data.shape)
			dSetName = "/data"+str(batInd)
			f.create_dataset(dSetName, data=X_data, compression='gzip', compression_opts=9)
			print batInd
			j = 0
			batInd = batInd +1
		else:
			j = j+1		
		
  	f.close()

def writeTestData():
	print('writing to file ...')
	f = h5py.File("FreshCode/testData_SintelClean.h5", 'w')
	img1List = open("Sintel_Test_Img1.list").readlines()
	img2List = open("Sintel_Test_Img2.list").readlines()
	flowList = open("Sintel_Test_Flow.list").readlines()
	j = 0
	batInd = 1
	X_data = np.empty([137*1, 8, 436, 1024]) # [640, 8, 384, 512]
	for i in testNum:
		print(i)
		X_data[j, :3] = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		X_data[j, [3,4,5]] = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		X_data[j, 6:] = np.transpose(np.asarray(readFloToFile(str(flowList[i].split()[0]))), (0,2,1))
		if j == ((137*1) - 1):
			print(X_data.shape)
			dSetName = "/data"+str(batInd)
			f.create_dataset(dSetName, data=X_data, compression='gzip', compression_opts=9)
			j = 0
		else:
			j = j+1
	
  	f.close()

def sampleForColorCoding():
	print('writing to file ...')
	f = h5py.File("FreshCode/sampleForColorCoding.h5", 'w')

	#comment this section later
	"""
	imgRead = misc.imresize(misc.imread('FreshCode/Evaluate/OtherExamples/frame_0007.png'), (384,512), interp='bilinear', mode=None)
	data1 = np.transpose(np.asarray(imgRead), (2,0,1))
	imgRead = misc.imresize(misc.imread('FreshCode/Evaluate/OtherExamples/frame_0008.png'), (384,512), interp='bilinear', mode=None)
        data2 = np.transpose(np.asarray(imgRead), (2,0,1))
	imgRead = readFloToFile('FreshCode/Evaluate/OtherExamples/frame_0007.flo')
	tempFlow = np.zeros((3, 1024, 436))
	tempFlow2 = np.zeros((3, 512, 384))
	tempFlow[0] = imgRead[0]
	tempFlow[1] = imgRead[1]
	tempFlow[2] = imgRead[1]
	tempFlow2 = np.transpose(np.asarray(tempFlow), (1,2,0))
	tempFlow2 = misc.imresize(tempFlow2, (384,512), interp='bilinear', mode=None)
	imgRead = tempFlow2[:,:,[0,1]]
	data3 = np.transpose(np.asarray(imgRead), (2,0,1))
	"""
		
	
	data1 = np.transpose(np.asarray(misc.imread('../MPI-Sintel-complete/training/clean/bamboo_2/frame_0015.png')), (2,0,1)) #bamboo_2 alley_1
        data2 = np.transpose(np.asarray(misc.imread('../MPI-Sintel-complete/training/clean/bamboo_2/frame_0016.png')), (2,0,1))
	data3 = np.transpose(np.asarray(readFloToFile('../MPI-Sintel-complete/training/flow/bamboo_2/frame_0015.flo')), (0,2,1))
	data4 = np.transpose(np.asarray(readFloToFile('FreshCode/Evaluate/sintel4_caffe.flo')), (0,2,1)) #sintel3_epic.flo

	#data3 = np.transpose(np.asarray(readFloToFile('../../FlowNet/dispflownet-release/models/FlowNetS/data/0000000-gt.flo')), (0,2,1))
	#data1 = np.transpose(np.asarray(misc.imread('../../FlowNet/dispflownet-release/models/FlowNetS/data/0000000-img0.ppm')), (2,0,1))
        #data2 = np.transpose(np.asarray(misc.imread('../../FlowNet/dispflownet-release/models/FlowNetS/data/0000000-img1.ppm')), (2,0,1))
	#data4 = np.transpose(np.asarray(readFloToFile('../../FlowNet/dispflownet-release/models/FlowNetS/data/0000000-gt.flo')), (0,2,1))

	f.create_dataset("/data1", data=data1)
	f.create_dataset("/data2", data=data2)
	f.create_dataset("/data3", data=data3)	
	f.create_dataset("/data4", data=data4)

def createMeanImg():
	f = h5py.File("FreshCode/meanDataSintel.h5", 'w')
	img1List = open("Sintel_Train_Img1.list").readlines()
	img2List = open("Sintel_Train_Img2.list").readlines()
        meanIm1 = np.zeros((3, 436, 1024))  # 384, 512   448 , 1024
	meanIm2 = np.zeros((3, 436, 1024))
	meanSqredIm1 = np.zeros((3, 436, 1024))
	meanSqredIm2 = np.zeros((3, 436, 1024))
	data1 = np.empty([4, 3, 436, 1024])
	temp = np.empty([4, 3, 436, 1024])
	totalImgs = 904*2 #22232
	for i in range(0, (totalImgs-1)):
		meanIm1 = meanIm1 + np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		meanIm2 = meanIm2 + np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		print(i)
	 			
        meanIm1 = meanIm1/totalImgs
        meanIm2 = meanIm2/totalImgs

	for i in range(0, (totalImgs-1)):
		temp = np.transpose(np.asarray(misc.imread(str(img1List[i].split()[0]))), (2,0,1))
		meanSqredIm1 = meanSqredIm1 + np.square(np.subtract(temp,meanIm1))
		temp = np.transpose(np.asarray(misc.imread(str(img2List[i].split()[0]))), (2,0,1))
		meanSqredIm2 = meanSqredIm2 + np.square(np.subtract(temp,meanIm2))
		print(i)

	meanSqredIm1 = meanSqredIm1/totalImgs
        meanSqredIm2 = meanSqredIm2/totalImgs
	print((meanSqredIm1 >= 0).all())
	print((meanSqredIm2 >= 0).all())
	data1[0] = meanIm1
	data1[1] = meanIm2
	data1[2] = np.sqrt(meanSqredIm1)
	data1[3] = np.sqrt(meanSqredIm2)
	f.create_dataset("/data", data=data1)

def writeSintelFeatures():
	print('writing to file ...')
	f = h5py.File("FreshCode/trainData_SintelFeatures.h5", 'w')
	f2 = h5py.File("FreshCode/trainData_SintelFlowData.h5", 'w')
	for i in range(1,227): #range(1,227) (227,453)
		X_data = torchfile.load('FreshCode/sintelFeat/sintelFeatures'+str(i)+'.t7')
		flo_data = torchfile.load('FreshCode/sintelFeat/flow'+str(i)+'.t7')
		print(i)
		dSetName = "/data"+str(i)
		dSetName2 = "/flow"+str(i)
		dSetName3 = "/tmpflow"+str(i)
		f.create_dataset(dSetName, data=X_data[0], compression='gzip', compression_opts=9)
		f2.create_dataset(dSetName2, data=flo_data, compression='gzip', compression_opts=9)
		f.create_dataset(dSetName3, data=X_data[1], compression='gzip', compression_opts=9)
  	f.close()
	f2.close()


#writeTestData()
#writeTrainData()
#sampleForColorCoding()
#createMeanImg()
writeSintelFeatures()


