# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali


# Class representing a node of a decision tree
class Node:

	def __init__(self):
		self.left = None
		self.right = None
		self.splitAt = None
		self.splitVal = None
		self.isLeaf = False
		self.probDist = {1:0.00, 2:0.00, 3:0.00, 4:0.00}

	def setLeft(self,left):
		self.left = left

	def setRight(self,right):
		self.right = right

	def setSplitAt(self,spa):
		self.splitAt = spa

	def setSplitVal(self,spb):
		self.splitVal = spb

	def makeLeaf(self,dist):
		self.isLeaf = True
		self.probDist = dist

# Class representing a decision tree
class DecisionTree:

	def __init__(self):
		self.root = Node()

	# Use this decision tree to classify an input
	def applyDT(self,x):
		curr = self.root
		while not curr.isLeaf:
			if x[curr.splitAt] > curr.splitVal:
				curr = curr.right
			else:
				curr = curr.left
		return curr.probDist