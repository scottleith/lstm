import numpy as np
from copy import deepcopy


#-------LSTM functions


def _predict( parameters, X, Y ):

	''' Runs new data through an LSTM with given params. Returns predictions.'''

	if len(X.shape) > 2:
			curr_batch_size, n_timesteps, n_features = X.shape
	else:
		curr_batch_size, n_timesteps = X.shape
		n_features = 1
		X = np.reshape( X, [ curr_batch_size, n_timesteps, n_features ] )
	y = np.reshape( Y, [ curr_batch_size, -1 ] )
	cache, gradcache = _create_activity_caches( n_timesteps, curr_batch_size, h_size, n_features )
	cache, output = _run_forward_pass( X, Y, n_timesteps, cache, parameters )
	output = np.reshape(output, [ curr_batch_size, n_timesteps, n_features ] )
	cost = _calculate_cost( output, Y )	
	return cost, output

def _initialize_lstm_params( mini_batch_size, h_size, output_timesteps, output_features ):

	''' Initializes and returns LSTM parameters. Weights are initialized with desired approach. '''
	
	parameters = {}

	parameters['Wf'] = _initialize_weights( [ (n_features+h_size), h_size] )
	parameters['Wi'] = _initialize_weights( [ (n_features+h_size), h_size] )
	parameters['Wc'] = _initialize_weights( [ (n_features+h_size), h_size] )
	parameters['Wo'] = _initialize_weights( [ (n_features+h_size), h_size] )
	parameters['Wy'] = _initialize_weights( [ h_size, output_timesteps*output_features ] )

	parameters['bf'] = np.ones( ( 1, h_size ) ) - 2
	parameters['bi'] = np.zeros( ( 1, h_size ) )
	parameters['bc'] = np.zeros( ( 1, h_size ) )
	parameters['bo'] = np.zeros( ( 1, h_size ) )
	parameters['by'] = np.zeros( (  1, output_timesteps*output_features ) ) 

	return parameters

def _create_activity_caches( n_timesteps, curr_batch_size, h_size, n_features ):

	''' Initialize the activity and gradient caches for backward pass '''

	cache = {}

	cache['f'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['i'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['c'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['o'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['h_in'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['h_out'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )
	cache['hx'] = np.zeros( (n_timesteps, curr_batch_size, h_size+n_features ) )
	cache['cell_state'] = np.zeros( (n_timesteps, curr_batch_size, h_size ) )

	cache['h0'] = np.zeros( (curr_batch_size, h_size) )
	cache['c0'] = np.zeros( (curr_batch_size, h_size) )

	gradcache = {}

	gradcache['df'] = np.zeros_like( cache['f'] )
	gradcache['di'] = np.zeros_like( cache['i'] )
	gradcache['dc'] = np.zeros_like( cache['c'] )
	gradcache['do'] = np.zeros_like( cache['o'] )
	gradcache['dh_in'] = np.zeros_like( cache['h_in'] )
	gradcache['dh_out'] = np.zeros_like( cache['h_out'] )
	gradcache['dhx'] = np.zeros_like( cache['hx'] )
	gradcache['dcell_state'] = np.zeros_like( cache['cell_state'] )

	gradcache['dWf'] = np.zeros_like( parameters['Wf'] )
	gradcache['dWi'] = np.zeros_like( parameters['Wi'] )
	gradcache['dWc'] = np.zeros_like( parameters['Wc'] )
	gradcache['dWo'] = np.zeros_like( parameters['Wo'] )
	gradcache['dWy'] = np.zeros_like( parameters['Wy'] )

	gradcache['dbf'] = np.zeros_like( parameters['bf'] )
	gradcache['dbi'] = np.zeros_like( parameters['bi'] )
	gradcache['dbc'] = np.zeros_like( parameters['bc'] )
	gradcache['dbo'] = np.zeros_like( parameters['bo'] )
	gradcache['dby'] = np.zeros_like( parameters['by'] )

	return cache, gradcache



def _run_batch( train_set_X, train_set_Y, parameters, mini_batch_size, h_size, val_X = None, val_Y = None ):

	''' For easy one-function running of an LSTM. Runs a full batch. '''

	# Shuffle dataset at the start of every batch run, so the model isn't seeing the same series 
	# each time. 
	shuffx,shuffy  = _shuffle( train_set_X, train_set_Y )
	batches_x, batches_y = gen_batches( shuffx, shuffy, mini_batch_size )

	for x_batch, y_batch in zip( batches_x, batches_y ):

		# Assumed size is [ samples, timesteps, features ]
		# If size fits, go with it. If not, it is probably a time series, so we reshape.
		if len(x_batch.shape) > 2:
			curr_batch_size, n_timesteps, n_features = x_batch.shape
		else:
			curr_batch_size, n_timesteps = x_batch.shape
			x_batch = np.reshape(x_batch, [ curr_batch_size, n_timesteps, 1 ] )

		# Reset the caches to be empty at the start of each batch run / iteration.
		cache, gradcache = _create_activity_caches( n_timesteps, curr_batch_size, h_size, n_features )

		# Forward pass, backward pass, parameter update.
		cache, output = _run_forward_pass( x_batch, y_batch, n_timesteps, cache, parameters )
		parameters, gradcache = _run_backward_pass( output, y_batch, n_timesteps, cache, gradcache, parameters  )
		parameters = _apply_gradients( gradcache, learning_rate = .001, parameters )

	# If y was multidimensional (M features at each timepoint), reshape to fit output layer.
	y = np.reshape( y_batch,  [y_batch.shape[0], -1 ] )
	# Cost for reporting.
	traincost = np.sqrt( np.mean( np.power(output - y,2), axis = 0 ) )

	# Calculating cost on validation set for reporting, if provided. Could be its own func. 
	if val_X is not None:
		if len(val_X.shape) > 2:
			curr_batch_size, n_timesteps, n_features = val_X.shape
		else:
			curr_batch_size, n_timesteps = val_X.shape
			n_features = 1
			val_X = np.reshape(val_X, [ curr_batch_size, n_timesteps, n_features ] )

		cache, _ = _create_activity_caches( n_timesteps, curr_batch_size, h_size, n_features )
		_, output = _run_forward_pass( val_X, val_Y, n_timesteps, cache, parameters )
		y = np.reshape( val_Y,  [val_Y.shape[0], -1 ] )
		valcost = np.sqrt( np.mean( np.power(output - y,2), axis = 0 ) )

		return parameters, traincost, valcost

	else: return parameters, traincost

def _run_forward_pass( x_batch, y_batch, n_timesteps, cache, parameters ):
	''' Runs the forward pass through repeated calls to _run_forward_cell_step'''
	for t in range(n_timesteps):

		cache = _run_forward_cell_step( x_batch, t, cache, parameters )

	output = np.dot( cache['h_out'][ n_timesteps-1 ], parameters['Wy'] ) + parameters['by']

	return cache, output


def _run_forward_cell_step( x_batch, t, cache, parameters ):
	''' Moves through a single node / cell step. '''
	cache['h_in'][t] = cache['h_out'][t-1] if t > 0 else cache['h0']
	cache['hx'][t] = np.hstack( (x_batch[:,t], cache['h_in'][t]) )

	cache['cell_state'][t] = cache['cell_state'][t-1] if t > 0 else cache['c0']

	cache['f'][t] = sigmoid( np.dot( cache['hx'][t], parameters['Wf'] ) + parameters['bf'] ) 
	cache['i'][t] = sigmoid( np.dot( cache['hx'][t], parameters['Wi'] ) + parameters['bi'] ) 
	cache['c'][t] = tanh( np.dot( cache['hx'][t], parameters['Wc'] ) + parameters['bc'] ) 
	cache['cell_state'][t] =  ( cache['f'][t] * cache['cell_state'][t] ) + ( cache['i'][t] * cache['c'][t] )
	cache['o'][t] = sigmoid( np.dot( cache['hx'][t], parameters['Wo'] ) + parameters['bo'] ) 
	cache['h_out'][t] = cache['o'][t] * tanh( cache['cell_state'][t] )

	return cache

def _run_backward_pass( output, y_batch, n_timesteps, cache, gradcache, parameters ):

	''' Runs the backward pass through repeated calls to _run_backward_cell_step'''

	#Compute initial gradient from the cost back to final h_out.
	curr_batch_size = y_batch.shape[0]
	dy = output - np.reshape( y_batch, [ curr_batch_size, -1 ] )
	dWy = np.dot( cache['h_out'][ n_timesteps-1 ].T, dy )
	dby = deepcopy( dy )
	gradcache['dh_out'][ n_timesteps-1 ] = dy @ parameters['Wy'].T
	# Initialize the hidden and cell state gradients coming from the next step
	# to zero, because there are no steps after the final step. 
	dh_next, dc_next = 0,0

	for t in reversed( range(n_timesteps) ):

		dh_next, dc_next, gradcache = _run_backward_cell_step( t, dh_next, dc_next, cache, gradcache, parameters )

	return parameters, gradcache

def _run_backward_cell_step( t, dh_next, dc_next, cache, gradcache, parameters ):

	'''Computes the gradients for a single cell step and returns the current gradient totals
	for the hidden and cell states (note they are passed along the constant error carousel)
	'''

	gradcache['dh_out'][t] += dh_next
	gradcache['dcell_state'][t] = ( cache['o'][t] * gradcache['dh_out'][t] ) * dtanh(cache['cell_state'][t]) + dc_next
	
	#The final cell state required is from before the first cell step, so we use c0. 
	if t > 0:
		gradcache['df'][t] = ( cache['cell_state'][t-1] * gradcache['dcell_state'][t] ) * ( cache['f'][t] * ( 1. - cache['f'][t] ) )
	else: 
		gradcache['df'][t] = ( cache['c0'] * gradcache['dcell_state'][t] ) * ( cache['f'][t] * ( 1. - cache['f'][t] ) )

	gradcache['di'][t] = ( cache['c'][t] * gradcache['dcell_state'][t] ) * ( cache['i'][t] * ( 1. - cache['i'][t] ) )
	gradcache['dc'][t] = ( cache['i'][t] * gradcache['dcell_state'][t] ) * ( 1. - cache['c'][t]**2 )
	gradcache['do'][t] = tanh( cache['cell_state'][t] ) * gradcache['dh_out'][t] * ( cache['o'][t] * ( 1. - cache['o'][t] ) )

	gradcache['dWf'] += cache['hx'][t].T @ gradcache['df'][t]
	gradcache['dWi'] += cache['hx'][t].T @ gradcache['di'][t]
	gradcache['dWc'] += cache['hx'][t].T @ gradcache['dc'][t]
	gradcache['dWo'] += cache['hx'][t].T @ gradcache['do'][t]

	gradcache['dbf'] += np.sum( gradcache['df'][t], axis = 0 )
	gradcache['dbi'] += np.sum( gradcache['di'][t], axis = 0 )
	gradcache['dbc'] += np.sum( gradcache['dc'][t], axis = 0 )
	gradcache['dbo'] += np.sum( gradcache['do'][t], axis = 0 )

	dhxf = gradcache['df'][t] @ parameters['Wf'].T
	dhxi = gradcache['di'][t] @ parameters['Wi'].T
	dhxc = gradcache['dc'][t] @ parameters['Wc'].T
	dhxo = gradcache['do'][t] @ parameters['Wo'].T

	# Recover the hx gradient by summing the gradients for the four gates.
	dhx = dhxf + dhxi + dhxc + dhxo
	# Split out the hidden state gradient and discard the x_t gradient we don't need. 
	dh_next = dhx[:, n_features:]
	# Calculate the cell state gradient for use in the next step back.
	dc_next = cache['f'][t] * gradcache['dcell_state'][t]

	return dh_next, dc_next, gradcache

def _apply_gradients( gradcache, learning_rate, parameters ):

	''' Performs normal gradient descent updates on the parameters. '''

	parameters['Wf'] -= ( learning_rate * gradcache['dWf'])
	parameters['Wi'] -= ( learning_rate * gradcache['dWi'])
	parameters['Wc'] -= ( learning_rate * gradcache['dWc'])
	parameters['Wo'] -= ( learning_rate * gradcache['dWo'])
	parameters['Wy'] -= ( learning_rate * gradcache['dWy'])

	parameters['bf'] -= ( learning_rate * gradcache['dbf'])
	parameters['bi'] -= ( learning_rate * gradcache['dbi'])
	parameters['bc'] -= ( learning_rate * gradcache['dbc'])
	parameters['bo'] -= ( learning_rate * gradcache['dbo'])
	parameters['by'] -= ( learning_rate * gradcache['dby'])

	return parameters





#------------------------Time to test

h_size = 25
mini_batch_size = 128
learning_rate = .001
if len( train_set_Y.shape) > 2:
	n_features = train_set_X.shape[2]
else:
	n_features = 1
n_timesteps = train_set_X.shape[1]
output_timesteps = train_set_Y.shape[1]
if len( train_set_Y.shape) > 2:
 	output_features = train_set_Y.shape[2]
else: output_features = 1
mem_cell_size = n_features+h_size

parameters = _initialize_lstm_params( n_features, h_size, output_timesteps, output_features )
train_err, val_err = [], []


for i in range(1,2000):
	
	parameters, traincost, valcost = _run_batch( train_set_X, train_set_Y, parameters, mini_batch_size, h_size, val_set_X, val_set_Y )

	train_err.append(traincost)
	val_err.append(valcost)

	if i % 10 == 0:
		print( "Running epoch "+str(i) )
		print( 'Train set cost: %s' % (traincost) )
		print( 'Validation set cost: %s' % (valcost) )
		print(" ")
	

