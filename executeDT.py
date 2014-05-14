# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

import sys
import pickle
from decisionTree import Node, DecisionTree
import matplotlib.pyplot as plt
import numpy as np

# Print confusion matrix
def printConfusionMatrix(confMatrix):
	print '%20s %30s' % ('', 'Actual class')
	print '%20s %10s %10s %10s %10s' % ('Assigned Class', 'bolt', 'nut', 'ring', 'scrap')
	print '%20s %10d %10d %10d %10d' % ( 'bolt', confMatrix[0][0], confMatrix[0][1], confMatrix[0][2], confMatrix[0][3] )
	print '%20s %10d %10d %10d %10d' % (  'nut', confMatrix[1][0], confMatrix[1][1], confMatrix[1][2], confMatrix[1][3] )
	print '%20s %10d %10d %10d %10d' % ( 'ring', confMatrix[2][0], confMatrix[2][1], confMatrix[2][2], confMatrix[2][3] )
	print '%20s %10d %10d %10d %10d' % ('scrap', confMatrix[3][0], confMatrix[3][1], confMatrix[3][2], confMatrix[3][3] )

# Read data from file
def readData(filename):
	data = []
	fr = open(filename)
	for line in fr:
		data.append([])
		instance = line.split(',')
		instance[0],instance[1],instance[2] = float(instance[0]), float(instance[1]), int(instance[2])
		data[-1] = instance
	return data

# Main entry point
if __name__ == '__main__':
	data = readData(sys.argv[2])
	f = open(sys.argv[1])
	ensemble = pickle.load(f)

	profitMat = [[0.20, -0.07, -0.07, -0.07],
				 [-0.07, 0.15, -0.07, -0.07],
				 [-0.07, -0.07, 0.05, -0.07],
				 [-0.03, -0.03, -0.03, -0.03]]
	confusionMat = [[0.00,0.00,0.00,0.00],[0.00,0.00,0.00,0.00],[0.00,0.00,0.00,0.00],[0.00,0.00,0.00,0.00]]

	correct = 0.0
	incorect = 0.0
	profit = 0.00

	for y in data:
		d = {1:0.00, 2:0.00, 3:0.00, 4:0.00}
		for x in ensemble:
			ans = x.applyDT(y)
			for b in d.keys():
				d[b] += ans[b]
		classification = max(d, key=d.get)
		if classification != y[2]:
			incorect += 1.00
		else:
			correct += 1.00
		profit += profitMat[classification-1][y[2]-1]
		confusionMat[classification-1][y[2]-1] += 1.00

	redx, bluex, greenx, yellowx = [],[],[],[]
	redy, bluey, greeny, yellowy = [],[],[],[]
	for xc in np.arange(0,1,0.006):
		for yc in np.arange(0,1,0.006):
			y = [xc,yc]
			d = {1:0.00, 2:0.00, 3:0.00, 4:0.00}
			for x in ensemble:
				ans = x.applyDT(y)
				for b in d.keys():
					d[b] += ans[b]
			classification = max(d, key=d.get)
			if classification==4:
				redx.append(xc)
				redy.append(yc)
			if classification==1:
				bluex.append(xc)
				bluey.append(yc)
			if classification==3:
				greenx.append(xc)
				greeny.append(yc)
			if classification==2:
				yellowx.append(xc)
				yellowy.append(yc)
	plt.plot(redx,redy,'ro',bluex,bluey,'bo',greenx,greeny,'go',yellowx,yellowy,'yo')
	plt.xlabel("Six-fold rotational symmetry")
	plt.ylabel("Eccentricity")
	plt.title("No. of trees in ensemble = " + str(len(ensemble)))
	plt.show()
			
	
	accuracy = correct/(correct+incorect)
	col_totals = [ sum(x) for x in zip(*confusionMat)]


	print "[" + sys.argv[2] + "]"
	print ""
	print "No. of correctly classified samples: " + str(int(correct))
	print "No. of incorrectly classified samples: " + str(int(incorect))
	print "Recognition rate: " + str(accuracy*100.00) + " %"
	print "Profit obtained: " + str(profit)
	print "Confusion Matrix:"

	printConfusionMatrix(confusionMat)

	print "________________________________________________________________"