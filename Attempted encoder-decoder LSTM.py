import numpy as np
import tensorflow as tf
import chakin
import gensim
from copy import deepcopy
import re

text_file = open("author-quote.txt", "r")
lines = text_file.read().split("	") # In the file, quotes are split by line breaks. We want to remove the author names, which are split by tabs.

# Lines is a list with each quote. At the end of each quote is '\n author_name'. So let's remove that.

quotes = [ i.split("\n")[0] for i in lines ] #done!!!

# Now to process the corpus. 
corpus_raw = [ i.lower() for i in quotes ]
# Clean the rest with a function:

def clean_text( text ):
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ", text)
	text = re.sub(r"\+", " ", text)
	text = re.sub(r"\-", " ", text)
	text = re.sub(r"\=", " ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " ", text)
	text = re.sub(r" e g ", "for example", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"\s{2,}", " ", text)
	return text

corpus_raw = [ clean_text(i) for i in corpus_raw ]
sentences = [ i.split() for i in corpus_raw ]
sentences.sort( key = lambda x: len(x) )

# Tokenize into words 'manually', create wordset and method of
# turning words into integers/indices.

words = []

for sentence in corpus_raw:
	tempwords = []
	for word in sentence.split():
		tempwords.append(word)
	words.extend(tempwords)

words = list( set(words) ) # A 'set' in Python is an unordered collection of unique elements. 
# NOTES: 
# - Set objects do not support indexing. 
# - Might be used for membership testing, elimination of duplicate elements. 
# - Set comprehension is also supported ( e.g., {x for x in 'derp' if x not in 'burbadurb'} returns {'p', 'e'} )

words.insert(0,"PAD") # Add the ID for our padding value (zero).
words.insert(1,"<EOS>") # Add the ID for our end-of-sentence token.


word2int = {}
int2word = {}
vocab_size = len(words)
# NOTES:
# - enumerate returns a list of tuples (index, item).
# - The list might be (0, 'if'), (1, 'to'), (2, 'from'), and so on. 
for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word

sentences = [i+"<EOS>" for i in corpus_raw] # Add in the end-of-sentence tokens.
sentences = [ i.split() for i in sentences ]
sentences.sort( key = lambda x: len(x) )
sentences.pop(0) # the stupid "aa milne" sentence that won't go away. 

# Create word ID dataset
word_ids = []
for sentence in sentences:
	temp = [ word2int[i] for i in sentence ]
	word_ids.append( temp )

# Time to get the embeddings.
def batch( inputs, max_seq_len = None ):
    seq_len = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_seq_len is None:
        max_seq_len = max(seq_len) 
    inputs_batch_major = np.zeros(shape=[batch_size, max_seq_len], dtype=np.int32) # == PAD 
    for i, seq in enumerate(inputs):
        pad = np.zeros( ( max_seq_len - len(seq ) ), dtype = np.int32 )
        insert = np.concatenate( (seq, pad) )
        inputs_batch_major[i] = insert
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, seq_len

batches_x = []
batches_x_seqlen = []
counter = 0
batch_size = 100
inputs = deepcopy(word_ids)
while counter < len(inputs):
	currbatch = inputs[ counter : min( counter+batch_size, len(inputs) ) ]
	currbatch, currbatch_seqlen = batch( currbatch )
	batches_x.append(currbatch)
	batches_x_seqlen.append( currbatch_seqlen)
	counter = counter+batch_size



enc_hidden_units = 1000
dec_hidden_units = enc_hidden_units*2 # because the encoder will be bidirectional.
embedding_size = 300

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Create placeholders for the encoder / decoder inputs, lengths, and targets.
enc_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "encoder_inputs" )
enc_inp_len = tf.placeholder( shape = (None,), dtype = tf.int32, name = "encoder_input_lengths" )
dec_inp = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_inputs" )
dec_inp_len = tf.placeholder( shape = (None,), dtype = tf.int32, name = "decoder_input_lengths" ) # Possibly unnecessary. 
dec_tgt = tf.placeholder( shape = (None, None), dtype = tf.int32, name = "decoder_targets") 

# For when embeddings are provided directly, instead of trained and looked up.
# enc_inp_emb = tf.placeholder( shape = (None, None), dtype = tf.float32, name = "encoder_input_embeddings")

# For when embeddings are trained and looked up.
# NOTE: In this case, enc_inp consists of sequences of word IDs. 
embeddings = tf.Variable( tf.random_uniform( [vocab_size, embedding_size], -1., 1.), dtype = tf.float32 )
enc_inp_emb = tf.nn.embedding_lookup( embeddings, enc_inp )

# Primary cell
enc_cell_fw = tf.contrib.rnn.LSTMCell( enc_hidden_units )
enc_cell_bw = tf.contrib.rnn.LSTMCell( enc_hidden_units )

# Encoder - Bidirectional RNN
( (enc_fw_output, enc_bw_output),(enc_fw_finalstate, enc_bw_finalstate) ) = tf.nn.bidirectional_dynamic_rnn( cell_fw = enc_cell_fw, 
									cell_bw = enc_cell_bw, 
									inputs = enc_inp_emb, 
									sequence_length = enc_inp_len,
									dtype = tf.float32, 
									time_major = True )


enc_outputs = tf.concat( (enc_fw_output, enc_bw_output), 2 ) # concatenate along third axis - the hidden state

enc_finalstate_c = tf.concat( (enc_fw_finalstate.c, enc_bw_finalstate.c), 1 ) # concat along second axis - cell state
enc_finalstate_h = tf.concat( (enc_fw_finalstate.h, enc_bw_finalstate.h), 1 ) # concat along second axis - hidden state

enc_finalstate = tf.contrib.rnn.LSTMStateTuple( c = enc_finalstate_c, h = enc_finalstate_h )

# Decoder - raw_rnn

dec_cell = tf.contrib.rnn.LSTMCell( dec_hidden_units )

enc_max_time, batch_size = tf.unstack( tf.shape( enc_inp ) ) # Assumes zero-padding has already occurred.
dec_len = enc_inp_len + 5 # Allow for longer sequences to be produced by the decoder, but cut it at input+5. 
# Might be reasonable to go higher (e.g. +10 or beyond).


# The decoder must work like this:
# output -> 
# output projection via fully-connected layer -> 
# prediction (argmax id) -> 
# get input embedding for next timestep -> 
# input the embedding

# To get the output words at each decoder timestep, we must produce an actual word id prediction each time.
# So, we create our own global output variables to call at each timestep:

Wout = tf.Variable( tf.random_uniform( [dec_hidden_units, vocab_size], -1., 1.), dtype = tf.float32, name = "weights_output" )
bout = tf.Variable( tf.zeros( [vocab_size] ), dtype = tf.float32, name = "biases_output" )

# We are using tf.raw_rnn rather than the tf.nn.dynamic_rnn because the dynamic rnn does not allow us to feed
# decoder-produced tokens as input for the next timestep. 

eos_time_slice = tf.ones( [batch_size], dtype = tf.int32, name = 'EOS' )
pad_time_slice = tf.zeros( [batch_size], dtype = tf.int32, name = 'PAD' )

eos_step_embedded = tf.nn.embedding_lookup( embeddings, eos_time_slice ) # All IDs are 1s.
pad_step_embedded = tf.nn.embedding_lookup( embeddings, pad_time_slice ) # All IDs are 0s. 


# The raw_rnn requires its initial state and transition behaviour to be defined.

# Initial state:
def loop_fn_initial():
	initial_elements_finished = (0 >= dec_len) # All False at first timestep.
	initial_inp = eos_step_embedded # Provide the <EOS> embedding as first x_t input.
	initial_cell_state = enc_finalstate # Provide final state of encoder as initial decoder state.
	initial_cell_output = None # No output yet.
	initial_loop_state = None # Don't have any other information to pass.
	return( 
		initial_elements_finished, 
		initial_inp, 
		initial_cell_state, 
		initial_cell_output, 
		initial_loop_state
		)

# Transition behaviour:
def loop_fn_transition( time, prev_output, prev_state, prev_loop_state ):
	def get_next_input():
		output_logits = tf.add( tf.matmul( prev_output, Wout) , bout)
		pred = tf.argmax( output_logits, axis = 1 ) # is an ID (index)
		next_input = tf.nn.embedding_lookup( embeddings, pred ) # Could also be word lookup + embedding lookup
		return next_input # The embedding for the word produced this timestep. 
	elements_finished = (time >= dec_len) # Produces a boolnea tensor of [batch_size] which defines if the sequence has ended
	finished = tf.reduce_all( elements_finished ) # Boolean scalar, False as long as there is one False
	# i.e., are we finished? Unless all time > current dec_len (e.g., the corresponding enc_len+5),
	# we continue. 
	input_next = tf.cond( finished, lambda: pad_step_embedded, lambda: get_next_input() ) # If finished = True, 
	# this returns pad_step_embedded (sequence is over), otherwise returns get_next_input() and 
	# continues the loop. 
	state = prev_state # beginning state for next timestep
	output = prev_output # beginning output for next timestep
	loop_state = None # As above, no other information to pass. 
	return( elements_finished, input_next, state, output, loop_state )

# Combine the initialization and transition functions into single call that will check 
# if the state is None and return init or transition.
def loop_fn( time, prev_output, prev_state, prev_loop_state ):
	if prev_state is None:
		assert prev_output is None and prev_state is None
		return loop_fn_initial()
	else:
		return loop_fn_transition( time, prev_output, prev_state, prev_loop_state )


dec_outputs_ta, dec_finalstate, _ = tf.nn.raw_rnn( dec_cell, loop_fn )

dec_outputs = dec_outputs_ta.stack()

# tf.unstack will take the provided tensor and divide it along the values of the given axis (base axis = 0).
# So if something has shape (A,B,C,D) and we just call tf.unstack, output is a tensor of A tensors of shape (B,C,D).
# This is supposed to take apart a [ time, batch, hidden_units ] shape tensor
dec_max_steps, dec_batch_size, dec_dim = tf.unstack( tf.shape(dec_outputs) )

dec_outputs_flat = tf.reshape( dec_outputs, [-1, dec_hidden_units] ) # Flattens to [ t * b, h ] from [ t, b, h ]. 

dec_logits_flat = tf.add( tf.matmul( dec_outputs_flat, Wout ), bout )
dec_logits = tf.reshape( dec_logits_flat, (dec_max_steps, dec_batch_size, vocab_size) )

dec_predict = tf.argmax( dec_logits, 2 ) # argmax over 3rd dimension (vocab_size) - which word is most likely?

# For tf.one_hot, the first argument contains the indices, and the second contains the length of the sparse vectors.
# So indices of [1,2,3] and a depth of 5 would produce 3 one-hot vectors of length 5, with 1s at position 1, 2, and 3, respectively.
# tf.one_hot also has on_value and off_value (base = 1 and 0 respectively) that you can set on your own (e.g. 2, -2).
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels = tf.one_hot( dec_tgt, depth = vocab_size,
	dtype = tf.float32), logits = dec_logits  )
# ERROR: logits_size=[1000,27994] labels_size=[500,27994]


loss = tf.reduce_mean( stepwise_cross_entropy )
train_op = tf.train.AdamOptimizer().minimize( loss )
loss_track = []

sess.run( tf.global_variables_initializer() )

try:
	for i in range( len(batches_x) ):
		fd = { 
			enc_inp: batches_x[i], 
			enc_inp_len: batches_x_seqlen[i], 
			dec_tgt: batches_x[i] 
		}
		_, l = sess.run( [train_op, loss], feed_dict = fd )
		loss_track.append(l)
		if i == 0 or i % len(batches_x) == 0:
			print( 'batch {}'.format(i) )
			print( '  minibatch loss: {}'.format( sess.run(loss,fd) ) )
			predict_ = sess.run( decoder_prediction, fd )
			for j, (inp, pred) in enumerate( zip( fd[enc_inp].T, predict_.T) ):
				print( '  sample {}:'.format(j+1) )
				print( '  input  > {}'.format(inp) )
				print( '  predicted > {}'.format(pred) )
				if j >= 2:
					break
except KeyboardInterrupt:
	print( 'Training Interrupted.' )
