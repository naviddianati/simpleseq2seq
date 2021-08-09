'''
Created on Aug 7, 2021
@author: Navid Dianati

Based on an example from keras.io 
'''

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class Seq2SeqCharWithEmbedding(keras.Model):

    def __init__(self, input_sequence_length, target_sequence_length, n_input_tokens, n_target_tokens, latent_dim,
                input_token_index, target_token_index):
        super(Seq2SeqCharWithEmbedding, self).__init__()
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.n_input_tokens = n_input_tokens
        self.n_target_tokens = n_target_tokens
        self.latent_dim = latent_dim
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.reverse_target_token_index = dict([(i, char) for char, i in self.target_token_index.items() ])
        
        self.encoder_embedding = keras.layers.Embedding(input_dim=self.n_input_tokens, output_dim=self.n_input_tokens)
        self.decoder_embedding = keras.layers.Embedding(input_dim=self.n_target_tokens, output_dim=self.n_target_tokens)
    
        self.encoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, go_backwards=True)
        self.decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.decoder_dense = keras.layers.Dense(self.n_target_tokens, activation="softmax")
    
    def call_decoder(self, decoder_inputs, h, c, training=None):
        y = decoder_inputs
        y = self.decoder_embedding(y)
        y, h, c = self.decoder_lstm(y, initial_state=[h, c])
        y = self.decoder_dense(y)
        return y, h, c
    
    def call_encoder(self, encoder_inputs, training=None):
        x = encoder_inputs
        x = self.encoder_embedding(x)
        x, h, c = self.encoder_lstm(x)
        return x, h, c
    
    def call(self, inputs, training=None):
        """using fit(), during training this is automatically 
        called with training=True, but with training=False
        for validation at the end of each epoch. Will be
        needed if layers such as BatchNormalization are
        used."""
        encoder_inputs, decoder_inputs = inputs
        x, h, c = self.call_encoder(encoder_inputs, training=training)
        y, h, c = self.call_decoder(decoder_inputs, h, c, training=training)
        return y
    
    def predict(self, inputs, **kwargs):
        """Override predict(). We can't run our custom prediction
        logic simply by overriding predict_step() (which is called
        by predict()) because of our python sideeffects. So it seems 
        the only solution is to entirely override predict.
        Only encoder inputs are received at prediction time"""
        n_seqs = inputs.shape[0]
        predictions = []
        for i in range(n_seqs):
            predictions.append(self.decode_sequence(inputs[i: i + 1], **kwargs))
        return predictions
    
    def decode_sequence(self, seq, **kwargs):
        """Decode a single sequence. seq.shape == (1, input_sequence_length)
        seq elements are integers
        @kwarg join_token: if not None, join the list of decoded tokens using
        this token to produce a string instead."""
        _, h, c = self.call_encoder(tf.convert_to_tensor(seq))
        
        join_token = kwargs.get('join_token')
        
        target = np.zeros((1, 1))
        target[0, 0] = self.target_token_index["\t"]
        # Stop if this character is encountered
        stop_char_index = self.target_token_index["\n"]
        output = []
        while True:
            y, h_new, c_new = self.call_decoder(target, h, c)
            char_index = np.argmax(y)
            output.append(char_index)
            if (char_index == stop_char_index) or (len(output) > self.target_sequence_length):
                break
            target *= 0
            target[0, 0] = char_index
            h, c = h_new, c_new
            
        output = [self.reverse_target_token_index[x] for x in output]
        if join_token is not None:
            output = join_token.join(output)
        return output

