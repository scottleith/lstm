import numpy as np
from copy import deepcopy
import pandas as pandas
import matplotlib.pyplot as plt
# matplotlib is imported so you can visualize the sequences you want to predict, if you want.

def return_datasets( data, timesteps ):
	'''
	General function to take a long sequences and break it up into
	X and Y components with the desired timestep length, both train and test sets. 

	PARAMS Single long sequence, desired timesteps. Note that input/output timesteps are
	set to be the same.

	RETURNS normalized X/Y train and validation sets.
	'''

	sequences_X = []
	sequences_Y = []
	init_offset = (timesteps*2)-1

	# Iterates through our list or array, pulling the current and subsequent
	# n-timesteps of data for x and y.
	for i in range(0,10000-init_offset):
	    x = data[ 0+i : timesteps+i ]
	    y = data[ timesteps+i : (timesteps*2)+i]
	    sequences_X.append(x)
	    sequences_Y.append(y)

	sequences_X = np.array( sequences_X )
	sequences_Y = np.array( sequences_Y )

	# Normalize our X and Y according to the trainset maxes
	Xtrain_norm = sequences_X[:9000] / (np.amax(sequences_X[:9000]))
	Xval_norm = sequences_X[9000:] / (np.amax(sequences_X[:9000]))
	Ytrain_norm = sequences_Y[:9000] / (np.amax(sequences_Y[:9000]))
	Yval_norm = sequences_Y[9000:] / (np.amax(sequences_Y[:9000]))

	return Xtrain_norm, Ytrain_norm, Xval_norm, Yval_norm

# Function that turns a sequences into x*sinx 
def x_sin(x): return x*np.sin(x)

# Function that returns both x*sinx and x*cosx
def x_sin_cos(x):
    return x*np.sin(x), x*np.cos(x)

data = x_sin( np.linspace(0,100,10000))

data_s, data_c = x_sin_cos(np.linspace(0,100,10000))

test_s = np.reshape( np.array(data_s), [10000,1])
test_c = np.reshape( np.array(data_c), [10000,1])
# Stack x*sinx with a reversed x*cosx.
test = np.hstack( (test_s, test_c[::-1] ) )

Xtrain, Ytrain, Xval, Yval = return_datasets( test, timesteps = 12 )

