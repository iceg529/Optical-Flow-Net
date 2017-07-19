
import torchfile
import numpy as np
import matplotlib.pyplot as plt

floFile = 'down.flo' #flow_sample2_pred
floatCheck = np.float32(202021.25)
width = np.int32(1024) #128 512  1024
height = np.int32(448) #96 384  448
flowSample = torchfile.load('down.t7') #flow_sample2_pred
flowCh = np.empty([2, 448*1024])
flowCh[0] = flowSample[0].flatten()
flowCh[1] = flowSample[1].flatten()
flowArray = flowCh.flatten('F')

#print(flowArray.shape)

#a = np.array([[1,2], [3,4]])
#b = a.flatten('F')
#print(b.shape)

#plt.figure()
#plt.title('Flow fields')
#plt.quiver(flowSample[0],flowSample[1])
#plt.show()

flowFinal = np.float32(flowArray)


def writeFloData():
	print(floFile)
	with open(floFile, 'w') as f:
		floatCheck.tofile(f)
		width.tofile(f)
		height.tofile(f)
		flowFinal.tofile(f)
 		print(flowFinal.shape)
		#d2 = d1.flatten()
		#np.float32(d2).tofile(f)

def readFloData1():
	with open('0000000-gt.flo', 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		if 202021.25 != magic:
			print 'Magic number incorrect. Invalid .flo file'
		else:
			w = np.fromfile(f, np.int32, count=1)
			h = np.fromfile(f, np.int32, count=1)
			print 'Reading %d x %d flo file' % (w, h)
			data = np.fromfile(f, np.float32, count=2*w*h)
			print(data.shape)
			# Reshape data into 3D array (columns, rows, bands)
			data2D = np.resize(data, (2, h, w))
			print(data2D.shape)
	return data2D

#d1 = readFloData1()
writeFloData()

#x = np.linspace(-200,2,100)
#y = np.linspace(2,-200,100)
#u = flowSample[0].flatten()
#v = flowSample[1].flatten()
#plt.quiver(u, v)
#plt.show()
