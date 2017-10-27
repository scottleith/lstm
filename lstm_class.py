import numpy as np
from copy import deepcopy

class LstmSimpleClass( object ):

	''' Class that can run a basic single-layer LSTM front to back, or piece by piece using the component functions. 
	Assumes single output node.'''

	def __init__( self, X = 'none', Y = 'none', output_act_func = 'linear', loss_func = 'rmse', initializer = 'xavier', 
					forget_bias = 1, optimizer = 'momentum', num_epochs = 1000, mini_batch_size = 128, hidden_size = 50, 
					learning_rate = .001, beta1 = .9, beta2 = .999, verbose = 1, seed = 1234, train_size = .9,
					normalize = True ):

		self.verbose = verbose
		self.seed = seed
		if self.seed >= 2:
			np.random.seed( self.seed )
		# Note that data is stored in the class.
		self.X = X
		self.Y = Y
		# If you want a validation set and haven't prenormalized your data
		self.train_size = train_size
		self.normalize = normalize

		self.h_size = hidden_size
		self.mini_batch_size = mini_batch_size
		self.output_act_func = output_act_func
		self.loss_func = loss_func
		self.num_epochs = num_epochs
		self.initializer = initializer
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2

		# Assumed shape is [ samples, timesteps, features ]. If the input data do not fit that template,
		# they are reshaped.
		if len(X.shape) < 3:
			self.X = np.reshape( X, [ X.shape[0], X.shape[1], 1 ] )
		if len(Y.shape) < 3:
			self.Y = np.reshape( Y, [ Y.shape[0], Y.shape[1], 1 ] )
		self.n_timesteps = self.X.shape[1]
		self.n_features = self.X.shape[2]
		self.output_size = self.Y.shape[1] * self.Y.shape[2]


	def predict( self, X ):

		''' Takes in an X array, makes a prediction using the class object parameters. '''

		self.create_activity_caches( X.shape[0] )
		output = self.run_forward_pass( X )
		return output

	def fit( self ):

		''' All-in-one fitting function. '''

		if self.normalize == True:
			self.normalize_data()
		self.split_data()
		self.initialize_parameters()
		self.train_model()

	def normalize_data( self ):

		''' Normalizes the data according to the trainset maxes.
		Alters the object's X and Y. '''

		trainsize = round( self.X.shape[0] * self.train_size )
		Xnorm = np.amax(self.X[:trainsize])
		Ynorm = np.amax(self.Y[:trainsize])

		self.X = self.X / Xnorm
		self.Y = self.Y / Ynorm

	def split_data( self ):

		''' Splits data into train and validation sets. '''

		trainsize = round( self.X.shape[0] * self.train_size )
		self.Xtrain = self.X[:trainsize]
		self.Ytrain = self.Y[:trainsize]
		self.Xval = self.X[trainsize:]
		self.Yval = self.Y[trainsize:]

	def initialize_parameters( self ):

		''' Initializes parameters using the "_initialize_weights" utility function. 
		If the optimizer required a record of past gradients, those are initialized as well.'''

		self.Wf = _initialize_weights( [self.n_features+self.h_size, self.h_size], initializer = self.initializer )
		self.Wi = _initialize_weights( [self.n_features+self.h_size, self.h_size], initializer = self.initializer )
		self.Wc = _initialize_weights( [self.n_features+self.h_size, self.h_size], initializer = self.initializer )
		self.Wo = _initialize_weights( [self.n_features+self.h_size, self.h_size], initializer = self.initializer )
		self.Wy = _initialize_weights( [self.h_size, self.output_size], initializer = self.initializer )

		self.bf = np.zeros( ( 1, self.h_size ) )
		self.bi = np.zeros( ( 1, self.h_size ) )
		self.bc = np.zeros( ( 1, self.h_size ) )
		self.bo = np.zeros( ( 1, self. h_size ) )
		self.by = np.zeros( ( 1, self.output_size ) )

		if self.optimizer is 'momentum' or self.optimizer is 'adam':

			self.dWf_last = np.zeros_like( self.Wf )
			self.dWi_last = np.zeros_like( self.Wi )
			self.dWo_last = np.zeros_like( self.Wo )
			self.dWc_last = np.zeros_like( self.Wc )
			self.dWy_last = np.zeros_like( self.Wy )

			self.dbf_last = np.zeros_like( self.bf )
			self.dbi_last = np.zeros_like( self.bi )
			self.dbo_last = np.zeros_like( self.bo )
			self.dbc_last = np.zeros_like( self.bc )
			self.dby_last = np.zeros_like( self.by )

	def train_model( self ):

		''' Runs model through desired number of epochs and stores records of train/validation error.'''

		self.train_err = []
		self.val_err = []

		for i in range( self.num_epochs ):
			self.run_epoch( i )
			if self.verbose == 1 and i % 10 == 0:
				print( "Train cost at epoch %s: %s" % ( i, self.train_err[i] ) )
				print( "Validation cost at epoch %s: %s" % ( i, self.val_err[i] ) )
				print(" ")

	def run_epoch( self, epoch ):

		''' Shuffles datasets, splits it into batches, and runs the forward/backward passes on those batches.'''

		shuffx, shuffy = _shuffle( self.Xtrain, self.Ytrain )

		batches_x, batches_y = _gen_batches( shuffx, shuffy, self.mini_batch_size )

		for x_batch, y_batch in zip( batches_x, batches_y ):

			output = self.run_forward_pass( x_batch )
			self.run_backward_pass( y_batch, output )

		# Update error lists.
		self.train_err.append( _calculate_cost( self.loss_func, self.run_forward_pass( self.Xtrain ), self.Ytrain ) )
		self.val_err.append( _calculate_cost( self.loss_func, self.run_forward_pass( self.Xval ), self.Yval) )


	def create_activity_caches( self, curr_batch_size ):

		''' Create activity caches for forward pass. '''

		self.h0 = np.zeros( ( curr_batch_size, self.h_size ) )
		self.c0 = np.zeros( ( curr_batch_size, self.h_size ) )
		self.f = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.i = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.c = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.o = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.h_in = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.h_out = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )
		self.hx = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size+self.n_features ) )
		self.cell_state = np.zeros( (curr_batch_size, self.n_timesteps, self.h_size ) )



	def run_forward_pass( self, x ):

		''' Runs forward pass through repeated calls of _run_forward_cell_step. 
		Returns prediction. '''

		self.create_activity_caches( x.shape[0] )

		for t in range(self.n_timesteps):

			self._run_forward_cell_step( x, t )

		output = np.dot( self.h_out[ :, self.n_timesteps-1, : ], self.Wy ) + self.by

		return output

	def run_forward_cell_step( self, x, t ):

		''' Runs forward cell step. '''

		# Quick check to make sure our x_t and h_t are of appropriate shapes.
		assert x[:,t,:].shape[0] == self.h_in[:,t,:].shape[0], \
			"x does not match dimensions with h, x shape is {0} and h shape is {1}".format( str(x[:,t,:].shape), \
            str(self.h_in[:,t,:].shape) )

		self.h_in[:,t,:] = self.h_out[:,t-1,:] if t > 0 else self.h0
		self.hx[:,t,:] = np.hstack( (x[:,t,:], self.h_in[:,t,:]) )
		self.cell_state[:,t,:] = self.cell_state[:,t-1,:] if t > 0 else self.c0
		self.f[:,t,:] = sigmoid( np.dot( self.hx[:,t,:], self.Wf ) + self.bf )
		self.i[:,t,:] = sigmoid( np.dot( self.hx[:,t,:], self.Wi ) + self.bi )
		self.c[:,t,:] = tanh( np.dot( self.hx[:,t,:], self.Wc ) + self.bc )
		self.cell_state[:,t,:] = ( self.f[:,t,:] * self.cell_state[:,t,:] ) + ( self.i[:,t,:] * self.c[:,t,:] )
		self.o[:,t,:] = sigmoid( np.dot( self.hx[:,t,:], self.Wo ) + self.bi )
		self.h_out[:,t,:] = self.o[:,t,:] * tanh( self.cell_state[:,t,:] )



	def _run_backward_pass( self, y_batch, output ):

		''' Runs backward pass through repeated calls to _run_backward_cell_step. '''

		# Starting gradients calculated.
		self.initialize_gradient_caches()
		self.dy = output - np.reshape( y_batch, [ y_batch.shape[0], -1 ] )
		self.dWy = np.dot( self.h_out[ :, self.n_timesteps-1, : ].T, self.dy )
		self.dby = np.sum(self.dy, axis = 0)
		self.dh_out[ :, self.n_timesteps-1, : ] = self.dy @ self.Wy.T
		# Initialize these to zero because there are no gradients after the last timestep.
		dh_next, dc_next = 0, 0

		for t in reversed( range( 0, self.n_timesteps ) ):

			dh_next, dc_next = self.run_backward_cell_step( t, dh_next, dc_next )

		self._apply_gradients()

	def initialize_gradient_caches( self ):

		''' Creates caches for the gradients as we go along. 
		We do this so we can eventually sum them at the end. '''

		self.df = np.zeros_like( self.f )
		self.di = np.zeros_like( self.i )
		self.dc = np.zeros_like( self.c )
		self.do =np.zeros_like( self.o )
		self.dh_in = np.zeros_like( self.h_in )
		self.dh_out = np.zeros_like( self.h_out )
		self.dhx = np.zeros_like( self.hx )
		self.dcell_state = np.zeros_like( self.cell_state )

		self.dWf = np.zeros_like( self.Wf )
		self.dWi = np.zeros_like( self.Wi )
		self.dWc = np.zeros_like( self.Wc )
		self.dWo = np.zeros_like( self.Wo )
		self.dWy = np.zeros_like( self.Wy )

		self.dbf = np.zeros_like( self.bf )
		self.dbi = np.zeros_like( self.bi )
		self.dbc = np.zeros_like( self.bc )
		self.dbo = np.zeros_like( self.bo )
		self.dby = np.zeros_like( self.by )

	def run_backward_cell_step( self, t, dh_next, dc_next ):

		''' Runs each backward step. Note that dh_next and dc_next are constantly passed
		along the error carousel. '''

		self.dh_out[:,t,:] += dh_next

		self.do[:,t,:] = ( tanh( self.cell_state[:,t,:] ) * self.dh_out[:,t,:] ) * ( self.o[:,t,:] * (1-self.o[:,t,:]) ) 
		self.dcell_state[:,t,:] = ( self.o[:,t,:] * self.dh_out[:,t,:] ) * (1. - self.cell_state[:,t,:]**2 ) + dc_next

		# Once we reach t = 0, there is no t-1 to draw from so we use c0. 
		if t > 0:
			self.df[:,t,:] = (self.cell_state[:,t-1,:] * self.dcell_state[:,t,:]) * ( self.f[:,t,:] * (1-self.f[:,t,:]) ) 
		else:
			self.df[:,t,:] = ( self.c0 * self.dcell_state[:,t,:] ) * ( self.f[:,t,:] * (1-self.f[:,t,:]) ) 

		self.di[:,t,:] = ( self.c[:,t,:] * self.dcell_state[:,t,:] ) * ( self.i[:,t,:] * (1-self.i[:,t,:]) ) 
		self.dc[:,t,:] = ( self.i[:,t,:] * self.dcell_state[:,t,:] ) * (1. - self.c[:,t,:]**2 )

		self.dWf += self.hx[:,t,:].T @ self.df[:,t,:]
		self.dWi += self.hx[:,t,:].T @ self.di[:,t,:]
		self.dWc += self.hx[:,t,:].T @ self.dc[:,t,:]
		self.dWo += self.hx[:,t,:].T @ self.do[:,t,:]

		self.dbf = self.dbf + np.sum( self.df[:,t,:], axis = 0 )
		self.dbi = self.dbi + np.sum( self.di[:,t,:], axis = 0 )
		self.dbo = self.dbo + np.sum( self.do[:,t,:], axis = 0 )
		self.dbc = self.dbc + np.sum( self.dc[:,t,:], axis = 0 )

		dhxf = self.df[:,t,:] @ self.Wf.T
		dhxi = self.di[:,t,:] @ self.Wi.T
		dhxo = self.do[:,t,:] @ self.Wo.T 
		dhxc = self.dc[:,t,:] @ self.Wc.T

		# Recover the hidden state gradient by summing the gate gradients and splitting it out of hx.
		dhx = dhxf + dhxi + dhxo + dhxc
		dh_next = dhx[:, :self.h_size]

		dc_next = self.f[:,t,:] * self.dcell_state[:,t,:]

		return dh_next, dc_next

	def apply_gradients( self ):

		''' Apply gradient updates according to selected optimizer. '''

		self.dWf = np.mean( self.dWf, axis = 0 )
		self.dWi = np.mean( self.dWi, axis = 0 )
		self.dWc = np.mean( self.dWc, axis = 0 )
		self.dWo = np.mean( self.dWo, axis = 0 )

		self.dbf = np.mean( self.dbf, axis = 0 )
		self.dbi = np.mean( self.dbi, axis = 0 )
		self.dbc = np.mean( self.dbc, axis = 0 )
		self.dbo = np.mean( self.dbo, axis = 0 )

		if self.optimizer == 'gradient descent':

			self.Wf -= ( self.learning_rate * self.dWf )
			self.Wo -= ( self.learning_rate * self.dWo ) 
			self.Wi -= ( self.learning_rate * self.dWi ) 
			self.Wc -= ( self.learning_rate * self.dWc )
			self.Wy -= ( self.learning_rate * self.dWy )

			self.bf -= ( self.learning_rate * self.dbf )
			self.bi -= ( self.learning_rate * self.dbi )
			self.bo -= ( self.learning_rate * self.dbo )
			self.bc -= ( self.learning_rate * self.dbc )
			self.by -= ( self.learning_rate * self.dby )

		elif self.optimizer == 'momentum':

			self.Wf -= self.learning_rate * ( self.beta1*self.dWf + (1-self.beta1)*self.dWf_last )
			self.Wi -= self.learning_rate * ( self.beta1*self.dWi + (1-self.beta1)*self.dWi_last )
			self.Wo -= self.learning_rate * ( self.beta1*self.dWo + (1-self.beta1)*self.dWo_last )
			self.Wc -= self.learning_rate * ( self.beta1*self.dWc + (1-self.beta1)*self.dWc_last )
			self.Wy -= self.learning_rate * ( self.beta1*self.dWy + (1-self.beta1)*self.dWy_last )

			self.bf -= self.learning_rate * ( self.beta1*self.dbf + (1-self.beta1)*self.dbf_last )
			self.bi -= self.learning_rate * ( self.beta1*self.dbi + (1-self.beta1)*self.dbi_last )
			self.bo -= self.learning_rate * ( self.beta1*self.dbo + (1-self.beta1)*self.dbo_last )
			self.bc -= self.learning_rate * ( self.beta1*self.dbc + (1-self.beta1)*self.dbc_last )
			self.by -= self.learning_rate * ( self.beta1*self.dby + (1-self.beta1)*self.dby_last )

			self.dWf_last = deepcopy( self.dWf )
			self.dWi_last = deepcopy( self.dWi )
			self.dWo_last = deepcopy( self.dWo )
			self.dWc_last = deepcopy( self.dWc )
			self.dWy_last = deepcopy( self.dWy )

			self.dbf_last = deepcopy( self.dbf )
			self.dbi_last = deepcopy( self.dbi )
			self.dbo_last = deepcopy( self.dbo )
			self.dbc_last = deepcopy( self.dbc )
			self.dby_last = deepcopy( self.dby )
