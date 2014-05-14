# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

import sys
from decisionTree import Node, DecisionTree
from math import log
import random
import pickle
import matplotlib.pyplot as plt

# Method for calculating the entropy of a given sample
def entropy(samples):
	ans = 0.00
	p = {1:0.00, 2:0.00, 3:0.00, 4:0.00}
	for sample in samples:
		p[sample[2]] += 1
	for x in p:
		prob = float(p[x])/float(len(samples))
		logval = 0.00 if prob <= 0.00 else log(prob,2)
		ans += -prob*logval
	return ans

# Method for calculating the information gain after a split
def IG(left,right):
	ileft = entropy(left)
	iright = entropy(right)
	itotal = entropy(left + right)
	ans = itotal - ((float(len(left))*ileft + float(len(right))*iright)/float(len(left) + len(right)))
	return ans

# Method for training a single decision tree on given sample with a given depth limit
def trainDecisionTree(currNode,samples,currDepth,depthLimit):

	if len(samples)==0:
		x = random.randint(1,4)
		currNode.probDist[x] = 1.00
		currNode.isLeaf = True
		return

	count = {1:0, 2:0, 3:0, 4:0}
	for x in samples:
		count[x[2]] += 1

	for x in count.keys():
		if count[x]==len(samples):
			currNode.probDist[x] = 1.00
			currNode.isLeaf = True
			return

	if currDepth == depthLimit:
		n = float(len(samples))
		for x in samples:
			currNode.probDist[x[2]] += 1.00

		for x in currNode.probDist.keys():
			currNode.probDist[x] /= n

		currNode.isLeaf = True
		return

	sorted_by_first = sorted(samples, key=lambda samples: samples[0])
	sorted_by_second = sorted(samples, key=lambda samples: samples[1])

	maxig = -1.00
	bestslpit = 0.00
	flag = 0
	for x in range(len(samples)-1):
		left, right = sorted_by_first[0:x+1], sorted_by_first[x+1:len(samples)]
		ig = IG(left,right)
		if ig > maxig:
			maxig = ig
			bestslpit = (sorted_by_first[x][0] + sorted_by_first[x+1][0])/2.00
			maxleft = left
			maxright = right

	for x in range(len(samples)-1):
		left, right = sorted_by_second[0:x+1], sorted_by_second[x+1:len(samples)]
		ig = IG(left,right)
		if ig > maxig:
			flag = 1
			maxig = ig
			bestslpit = (sorted_by_second[x][0] + sorted_by_second[x+1][0])/2.00
			maxleft = left
			maxright = right

	currNode.left, currNode.right = Node(), Node()
	currNode.setSplitAt(flag)
	currNode.setSplitVal(bestslpit)

	trainDecisionTree(currNode.left,maxleft,currDepth+1,depthLimit)
	trainDecisionTree(currNode.right,maxright,currDepth+1,depthLimit)

# Calculate the squared error between two vectors
def se(v1,v2):
	ans = 0.00
	for i in v1.keys():
		ans += (v1[i]-v2[i])**2.00
	return ans;
	
# Calculate the sum-squared error of given samples(yprime) and actual output(y)
def sse(y,yprime):
	ans = 0.00
	for i in range(len(y)):
		ans += se(y[i],yprime[i])
	return ans/float(len(y))

# Generate and save the random forest for a given ensemble size and depth limit
def generateRandomForest(samples,ensembleSize,randomness,depthLimit):
	ensemble = []
	error = []
	k = int(len(samples)*randomness)

	y = []
	for x in samples:
		v1 = {1:0.00, 2:0.00, 3:0.00, 4:0.00}
		v1[x[2]] = 1.00
		y.append(v1)

	saveresultsat = 1
	for n in range(ensembleSize):
		dt = DecisionTree()
		trainDecisionTree(dt.root,random.sample(samples,k),0,depthLimit)
		ensemble.append(dt)
		norm = float(len(ensemble))
		ind = 0
		c,i = 0.00,0.00
		yprime = []		
		for x in samples:
			vote = {1:0.00, 2:0.00, 3:0.00, 4:0.00}
			for dtx in ensemble:
				latestClass = dtx.applyDT(x)
				for key in latestClass.keys():
					vote[key] += latestClass[key]
			

			norm = float(len(ensemble))
			for key in vote.keys():
				vote[key] = vote[key]/norm

			result = max(vote,key=vote.get)
			yprime.append(vote)

			if result == x[2]:
				c += 1.00
			else:
				i += 1.00

		error.append(sse(yprime,y))
		# error.append(c/(c+i))


		if (n+1)==saveresultsat or (n+1)==ensembleSize:
			saveresultsat *= 10
			name = 'ensemble' + str(n+1)
			f = open(name,'w')
			pickle.dump(ensemble,f)

	plt.plot(error)
	plt.ylabel('SSE')
	plt.xlabel('Ensemble size')
	plt.show()

	return ensemble

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

# Main function
if __name__ == '__main__':
	data = readData(sys.argv[1])
	
	ensemble = generateRandomForest(data,150,0.2,1)

	f = open("ensemble5","w")
	pickle.dump(ensemble,f)


