import numpy as np

class LstmLayer( object ):
	"""
	A single LSTM layer that can run forward and backward passes.

	NB: When dealing with the caches, a prefix of "d" means it stores 
	gradients, and a suffix of "_f" indicates it stores activated values 
	(e.g., a gate's values after being passed through the sigmoid function).
	"""

	def __init__( self, X, Y, initializer = 'glorot', learning_rate = .001,
		optim = 'momentum', forget_bias = 1, hidden_size = 50, 
		fully_connected_output = True, return_params = False ):
		"""
		Initialize the parameters and hyperparameters of the LSTM layer.
		Args:
			X: The inputs. Can be raw inputs or outputs from a lower layer.
			output_size: Needed for shape information if you want a fully-connected
				output layer.
			initializer: The initializer ('glorot','he','xavier','random_normal',
				'random_uniform') for this layer's weights.
			optim: The optimizer to use with this layer.
			forget_bias: If you want to tweak the forget gate bias in step with
				CITE, to discourage forgetting early in training.
			hidden_size: The number of neurons/nodes in the hidden state.
			fully_connected_output: Set to 'True' if you want the output layer
				to be fully connected (usually used for final/top layer).
			iter: The current iteration (required for momentum and adam
					optimizers).
		Updates:
			output: A numpy array containing the layer's outputs, node by node.
			FIOC and FIOC_f: A cache of the activity within the layer, node by node.
			dFIOC and dFIOC_f: A cache with the gradients of this layer, node by node.
		Returns:
			params: The parameters for the layer. 
		"""

		assert len( X.shape ) == 3 and max(X.shape) == X.shape[1], "Required shape\
		of X is [ timesteps, batch_size, features ]. This is so we can easily\
		reference a given cell step with the indexer [t] rather than [:,t,:]."

		self.n_timesteps = X.shape[0]
		self.input_size = X.shape[2]
		if Y is None:
			self.output_size = hidden_size
		else:
			self.output_size = Y.shape[0]*Y.shape[2]
		self.initializer = initializer
		self.h_size = hidden_size
		self.forget_bias = forget_bias
		self.optim = optim
		
		# Order of the gates is Forget, Input, Output, Candidate (FIOC). 
		# The +1 is for the biases.
		self.params = _initialize_weights( [(self.input_size+self.h_size+1), 
							(self.h_size*4)], initializer = self.initializer )
		self.params[0,:] = 1

		if fully_connected_output is True:
			self.Wy = _initialize_weights( [ self.h_size+1, self.output_size ], 
				initializer = self.initializer )
			self.by = np.zeros( (1, self.output_size ) )

		if return_params is True:
			return self.params

	def forward_pass( self, X ):
		"""
		Runs a forward pass of a single layer.
		Args: 
			- Self (layer parameters, activity caches).
			- Input data for this batch.
		Updates:
			- output
			- batch_size
			- All activity cache attributes (cellstate, cellstate_f, 
				hiddenstate, FIOC, FIOC_f, hx).
		Returns:
			- Output of the forward pass.
		"""

		self.batch_size = X.shape[1]
		self.initialize_fwd_caches()

		self.output = np.zeros( [self.n_timesteps, self.batch_size, 
			self.output_size] )

		for t in range(self.n_timesteps):
			self.forward_cell_step( X[t], t )
			self.output[t] = np.dot( self.hiddenstate[t], self.Wy ) + self.by

		return self.output

	def initialize_fwd_caches( self ):
		""" 
		Initializes the activity caches for this forward pass:
		The gate activations, the gate nonlinearities, the cell
		state and its nonlinearity, the hidden state, and the input
		vector. 
		Args:
			- Self.
		Updates:
			- Forward pass cache attributes (cellstate, cellstate_f, 
				hiddenstate, FIOC, FIOC_f, hx)
		Returns:
			- Nothing.
		"""

		# Gate activations. Forget, Input, Candidate, Output.
		self.FIOC = np.zeros( (self.n_timesteps, self.batch_size, self.h_size*4) )
		# Gate nonlinearities.
		self.FIOC_f = np.zeros( (self.n_timesteps, self.batch_size, self.h_size*4) )

		# Set FIOC indices for easy/clear referencing later on.
		self.f = slice( 0, self.h_size )
		self.i = slice( self.h_size, self.h_size*2 )
		self.o = slice( self.h_size*2, self.h_size*3 )
		self.c = slice( self.h_size*3, self.h_size*4 )

		self.cellstate = np.zeros( (self.n_timesteps, self.batch_size, 
			self.h_size) )
		self.cellstate_f = np.zeros_like( self.cellstate )
		self.hiddenstate = np.zeros_like( self.cellstate )
		self.hx = np.zeros( (self.n_timesteps, self.batch_size, 
			(self.h_size+self.input_size+1)) )

		self.h0 = np.zeros( (self.batch_size, self.h_size) )
		self.c0 = np.zeros_like( self.h0 )

	def forward_cell_step( self, x, t ):
		""" 
		Runs a single forward step of the forward pass.
		Args:
			- Self.
			- Input data x (vector or scalar).
			- Current timestep t (scalar).
		Updates:
			- Forward pass cache attributes (cellstate, cellstate_f, 
				hiddenstate, FIOC, FIOC_f, hx)
		Returns:
			- Nothing.
		"""

		# Quick check that x_t and h_t are of compatible shapes.
		assert x.shape[1] == self.hiddenstate[t].shape[0], \
		"x does not match dimensions with h, x shape is {0} and h shape \
		is {1}".format( str(x.shape), str(self.hiddenstate[t].shape) )

		self.hx[t,:,0] = 1 #bias
		self.hx[t,:,1:self.input_size+1] = x
		self.hx[t,:,self.input_size+1:] = self.hiddenstate[t-1] \
			if t > 0 else self.h0

		self.FIOC[t] = self.hx[t].dot( self.params ) # Compute gate activations.

		self.FIOC_f[t,:,:3*self.h_size] = sigmoid( self.FIOC[t,:,:3*self.h_size] )
		self.FIOC_f[t,:,self.f] += self.forget_bias 
		self.FIOC_f[t,:,self.c] = np.tanh( self.FIOC[t,:,self.c] )

		prev_cellstate = self.cellstate[t-1] \
			if t > 0 else self.c0
		self.cellstate[t] = prev_cellstate * self.FIOC_f[t,:,self.f] + \
			self.FIOC_f[t,:,self.i] * self.FIOC_f[t,:,self.o]
		self.cellstate_f[t] = np.tanh( self.cellstate[t] )
		self.hiddenstate[t] = self.cellstate_f[t] * self.FIOC[t,:,self.o]

	def calculate_cost( self, Y, cost_metric = 'rmse', return_gradient = True ):
		"""
		Calculates cost after the forward pass. NOTE: This is a fairly specific
		function that only accepts a certain shape of Y, and then flattens it 
		out. 
		Args:
			- Self.
			- Y.
			- cost_metric: The cost metric ('rmse', 'mae', 'softmax', 
				'logistic', 'sampled_softmax')
			- return_gradient: calculates and returns the gradient for the
				chosen cost metric.
		NOTES: 
			- Y is expected to be padded to span the entire series of timesteps, 
				so there is one gradient for each cell step. This is so the backward 
				pass can be used flexibly for multiple layerS (i.e., for lower 
				layers in a multilayer network, there is a gradient for each cell 
				step). 
		Updates:
			- None.
		Returns:
			- A vector of the cost.
			- A vector of the gradients.
		"""

		assert len( Y.shape ) is 3, "Y is expected to be of shape [ n_timesteps, \
		batch_size, n_features ]. For example, if Y is for classification, \
		rather than forecasting, the shape would be [ 1, batch_size, n_classes ]."

		y = Y[0]
		# Because we want a flattened Y (i.e., we want to turn [n_timesteps, batch_size,
		# n_features ] into [batch_size, n_timesteps*n_features]), we have to...
		for i in range(1,self.n_timesteps):
			y = np.concatenate( (y, Y[i]), axis = 1) # God forgive me.
		y = np.reshape( y, [1, y.shape[0], y.shape[1]] )

		if cost_metric is 'rmse':
			cost = np.sqrt( np.mean( np.power( self.output[self.n_timesteps-1] - y, 2 ), axis = 0 ) )
			grad = self.output[self.n_timesteps-1] - y

			# Pad targets with 0s so our gradient matrix works for every layer. 
			pad = np.zeros( ( self.n_timesteps-1, self.batch_size, self.output_size) ) 
			grad = np.concatenate( (pad, grad) )

		return cost, grad


	def backward_pass( self, dh_above, iteration ):
		"""
		Runs the backward pass for a single layer.
		Args: 
			- Self.
			- dh_above, the gradient from the layer above (output or
				otherwise).
		Updates:
			- Gradient cache attributes.
		Returns: 
			- The updated parameters.
			- Parameter gradients for the weights and biases.
			- The gradients for the input, dh_above.
		"""

		assert dh_above is not None, "The backward pass requires a\
			complete set of gradients from the layer above (output or\
			subsequent layer), dh_above. If this is the layer that produces\
			output,it will be the output gradients for output nodes and zeros\
			otherwise. If this is not the output layer, dh_above will simply\
			be the input (X) gradients from the layer above."

		# The hidden state gradients are initialized with the gradients
		# from the layer above. We will add in this layer's gradients
		# as we go. 
		self.d_hiddenstate = deepcopy( dh_above )
		self.initialize_grad_caches()
		dh_next = 0 #There are no gradients after the final cell step.
		dc_next = 0

		for t in reversed( range(self.n_timesteps) ):
			dh_next, dc_next = self.backward_cell_step( dh_next, dc_next, t )
			
		self.update_parameters( iteration = iteration )

		return self.params, self.dparams, self.dx

	def initialize_grad_caches( self ):
		""" 
		Initializes the gradient caches for the backward pass.
		Args: Self.
		Returns: Self (refreshed cache attributes).
		"""
		if self.optim is 'momentum' or 'adam':
			# Stores exponentially weighted avg of the gradient.
			self.v = np.zeros_like( self.params ) 
		if self.optim is 'adam':
			# Stores exponentially weighted avg of the squared gradient.
			self.s = np.zeros_like( self.params ) 

		self.dFIOC = np.zeros_like( self.FIOC )
		self.dFIOC_f = np.zeros_like( self.FIOC )
		self.dparams = np.zeros_like( self.params )
		self.dhx = np.zeros_like( self.hx )
		self.dx = np.zeros( (self.n_timesteps, self.batch_size, self.input_size) )
		self.dhiddenstate = np.zeros_like( self.hiddenstate )
		self.dcellstate = np.zeros_like( self.cellstate )
		self.dinputx = np.zeros( (self.n_timesteps, self.batch_size, 
			self.input_size) )

	def backward_cell_step( self, dh_next, dc_next, t ):
		"""
		Runs a single backward pass step.
		Args:
			- Self.
			- dh_next, the hidden state gradient from the prior step.
			- dc_next, the cell state gradient from the prior step.
			- t, the index of the current step.
		Updates:
			- All gradient cache attributes.
		Returns:
			- dh_next and dc_next for passage to next backward step.
		"""

		self.dhiddenstate[t] += dh_next
		self.dcellstate += (1 - self.cellstate_f[t]**2) * ( self.dhiddenstate[t] *\
			self.FIOC_f[t,:,self.o] )

		self.dFIOC_f[t,:,self.o] = self.cellstate_f[t] * self.dhiddenstate[t]
		if t > 0:
			self.dFIOC_f[t,:,self.f] = self.cellstate[t-1] * self.dcellstate[t]
			self.dcellstate[t-1] += self.FIOC_f[t,:,self.f] * self.dcellstate[t]
		else:
			self.dFIOC_f[t,:,self.f] = self.c0 * self.dcellstate[t]
			self.dc0 = self.FIOC_f[t,:,self.f] * self.dcellstate[t]
		self.dFIOC_f[t,:,self.i] = self.FIOC_f[t,:,self.c] * \
			self.dcellstate[t]
		self.dFIOC_f[t,:,self.c] = self.FIOC_f[t,:,self.i] * \
			self.dcellstate[t]

		self.dFIOC[t,:,self.c] = (1 - self.FIOC_f[t,:,self.c]**2) * self.FIOC_f[t,:,self.c]
		y = self.FIOC_f[t,:,:3*self.h_size]
		# Using 3*h_size below to compactly get Forget, Input, Output all in one.
		self.dFIOC[t,:,:3*self.h_size] = ( y * (1.-y) ) * self.dFIOC_f[t,:,:3*self.h_size]
			
		self.dparams +=  np.dot( self.hx[t].T, self.dFIOC[t] )
		self.dhx[t] = self.dFIOC[t].dot( self.params.T )
		self.dx[t] = self.dhx[t,:,1:self.input_size+1]
		
		dh_next = self.dhx[t,:,self.input_size+1:]
		if t == 0: self.dh0 = deepcopy( dh_next )
		dc_next = self.dcellstate[t]

		return dh_next, dc_next	

	def update_parameters( self, epsilon = 1e-7, iteration = None ):
		"""
		Updates the params of this LSTM layer using its gradients.
		Args:
			- Self.
			- Optimizer. Optimization algorithm to use - 'gradient descent',
					'momentum', 'adam'.
		Returns: 
			- A matrix containing this layer's updated parameters.
		Updates:
			- This layer's parameters, params.
			- The matrix containing the expoentially weighted avg of the 
				gradients, v.
			- The matrix containing the squared exponentially weighted avg 
				of the gradients, s.
		"""

		if self.optim is 'gradient_descent':
			self.params -= (self.learning_rate*self.dparams)
		elif self.optim is 'momentum':
			self.v = self.beta1*self.v + (1.-self.beta1)*self.dparams
			self.params -= (self.learning_rate*self.v)
		elif self.optim is 'adam':
			self.v = self.beta1*self.v + (1.-self.beta1)*self.dparams
			self.s = self.beta2*self.s + (1.-self.beta2)*(self.dparams**2)
			vcorr = self.v / ( 1. - np.pow(self.beta1, iteration) )
			scorr = self.s / (1. - np.pow(self.beta2, iteration))
			update = vcorr / ( np.sqrt(scorr) + epsilon )
			self.params -= (self.learning_rate*update)

		return self.params

class LSTMnetwork( object ):
	"""
	Builds and fits a Long Short-Term Memory network. Mostly intended for
	multiple layers, but can be used for one layer (e.g., if you want fancy
	things like scheduled sampling, automatic validation cost checks, etc.).
	Attributes:
		- X. The inputs.
		- Y. The targets.
		- output_act_func. The activation function for the output layer.
		- loss_func. The loss/cost function for the output.
		- optimizer. The optimizer to use ('gradient_descent','momentum',
			'adam').
		- scheduled_sampling. Whether or not to use scheduled sampling. As a
			default, sampling is not conducted (0% of outputs). The original
			paper suggests a coin flip (0.5). 
		- num_epochs: Number of epochs to run during training.
		- mini_batch_size: Size of the minibatches.
		- learning_rate: Learning rate for parameter updates.
		- beta1: Used only if optimizer is 'momentum' or 'adam'.
		- beta2: Used only if optimizer is 'adam'.
		- verbose: Set to '1' if you want the cost printed to console as
			the model trains.
		- seed: Seed. Like in Final Fantasy 8.
		- train_size. Percent of X used for training vs. validation. Alternatively,
			train_size can be set to 1, and prediction can be performed using 
			component functions outside of the fit function.
		- normalize. Normalizes the data according to the max in the training set.
	"""

	def __init__( self, X = 'none', Y = 'none', layer_sizes = [50], 
		output_act_func = 'linear', loss_func = 'rmse', optimizer = 'momentum', 
		scheduled_sampling = 0, num_epochs = 1000, mini_batch_size = 128,
		learning_rate = .001, beta1 = .9, beta2 = .999, verbose = 1, 
		seed = 1234, train_size = .9, normalize = True ):
		"""

		"""
		self.X = X
		self.Y = Y

		self.base_layer = LSTMlayer( X = X, hidden_size = layer_sizes[0],
			output_act_func = output_act_func )

		if len( layer_sizes ) > 1:
			for layer in layer_sizes[1:]:


