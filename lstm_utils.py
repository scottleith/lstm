import numpy as np

def _calculate_cost( loss_func = 'rmse', output, y ):

	''' Returns squared cost. //TODO: Add other loss functions.
	Args: 
	    loss_func: Loss function type. Currently only has 'rmse'.
	    output: The output from an LSTM.
	    y: The target values.
	Returns:
	    The cost of the same shape as the targets y.
	'''

	# If y isn't the right shape, reshapes it to match the output.
	if y.shape != output.shape:
		y = np.reshape(y, [ y.shape[0], -1])
	if loss_func == 'rmse':
		cost = np.sqrt( np.mean( np.power( output - y, 2 ), axis = 0 ) )

	return cost

def _gen_batches( data_x, data_y, batch_size ):

	''' 
	Splits X and Y into batches of batch_size. Last batch is the remainder. 
	Note that this function assumes an even number of batches. 
	
	Args:
	    data_x: Input data of shape [ batch_size, num_steps, num_features ].
	    data_y: The target data of shape [ batch_size, num_steps, num_features ].
	    batch_size: The desired size of batches. 
	'''

	batches_x, batches_y = [], []

	for i in range( 0, data_x.shape[0], batch_size ):
		batches_x.append( data_x[ i: i+batch_size ] )
		batches_y.append( data_y[ i : i+batch_size ] )

	return np.array(batches_x), np.array(batches_y)

def _shuffle( X, Y, axis = 0 ):

	'''
	Shuffles X and Y together, returns shuffled X and Y.
	Args:
	    X: The input data. 
	    Y: The target data.
	Returns:
	    Shuffled X and Y matrices.
	'''

	assert( X.shape[0] == Y.shape[0] )
	s = np.arange(X.shape[0])
	np.random.shuffle(s)
	return X[s], Y[s]

def _initialize_weights( shape, initializer = 'glorot'):

	'''
	Returns matrix of [shape] filled with random values 
	according to given initialization approach.
	Args:
	    shape: The desired shape of the weight matrix to
	           be generated.	
	    initializer: The type of initializer desired.
	Returns:
	    The weight matrix of desired shape, filled with
	    values produced by the indicated initializer.
	'''

	if initializer == 'random_normal':
		return np.random.randn( shape ) * 0.1
	elif initializer == 'random_uniform':
		return np.random.uniform( low = -.08, high = .08)
	elif initializer == 'glorot':
		base = np.sqrt(6) / np.sqrt( shape[0]+shape[1] )
		return np.random.uniform( low = -1*base, high = base, size = shape ) 
	elif initializer == 'he':
		return np.random.normal( loc = 0, scale = 2 / np.sqrt(shape[0]+shape[1]), size = shape )
	elif initializer == 'xavier':
		return np.random.normal( loc = 0, scale = 1 / shape[0], size = shape )


#-----Basic activation functions and their derivatives. 

def sigmoid( x ):
	return 1. / ( 1. + np.exp(-x) )

def tanh( x ):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dsigmoid( x ):
	return x * (1. - x)

def dtanh( x ):
	return (1. - x**2)
