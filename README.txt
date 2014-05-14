INSTRUCTIONS for trainDT.py:

Usage: python trainDT.py <datafile>

Train a random forest with 150 stumps,
Plot the SSE curve,
Save the ensemble object for ensemble size=1,10,100,150 using pickle,
The ensemble of size n is stored in a file named ensemble<n> e.g. for ensemble of size 100, the output file storing the
corresponding ensemble will be named "ensemble100"
The parameters to this algorithm are:
	- Randomness: set to 0.2 initially, indicating that 20% of the samples are chosen randomly and used to train a stump
	- Depth limit: set to 1 initially, indicating that the trees in the random forests are of height 1 i.e. stumps

Datafile format:
x1,y1,c1
x2,y2,c2
...

INSTRUCTIONS for executeDT.py:

Usage: python executeDT.py <ensemblefile> <datafile>
ensemblefile is the file containing the ensemble object which is read using pickle
datafile is the file containing the data to be tested

It tests the data from the datafile on the ensemble read from the ensemblefile and outputs the following:
- No. of misclassifications
- Recognition rate
- Profit
- Confusion matrix
And it plots the classification boundaries.