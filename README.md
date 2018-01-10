# lstm
Vanilla LSTM implementations.

<b>lstm_utils.py</b> are basic functions for use across the different implementations (e.g., batch generation, shuffling, activation functions, etc.)

<b>lstm_sample_data_generation.py</b> is a quick way to produce sequences for training/testing using the LSTM implementations. 

<b>lstm_dicts.py</b> is an LSTM implementation that just uses numpy, and dictionaries function as the caches.

<b>lstm_class.py</b> Is like lstm_dicts, but implemented as a Python class.

<b>lstm_multilayer.py</b> Is a pair of funtions, one for individual LSTM layers, and one to implement layers together. Unlike the prior implementations, this class holds all gate weights and biases (forget, input, output, candidate) in a single matrix. 
