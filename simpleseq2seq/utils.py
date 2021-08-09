'''
Created on Aug 9, 2021

@author: Navid Dianati
'''
import time
import numpy as np
from nltk.tokenize import RegexpTokenizer


class Seq2SeqDataParser:
    """Parse one input/output pair of sequences at a time 
    and transform them into a format directly usable by a 
    seq2seq model."""

    def __init__(self, tokenizer, start_token="\t", end_token="\n"):
        self.start_token = start_token
        self.end_token = end_token
        
        self.input_tokens = set()
        self.output_tokens = set([self.start_token, self.end_token])
        
        self.tokenizer = tokenizer
        
        self.input_texts = []
        self.output_texts = []
        self.input_token_index = dict()
        self.output_token_index = dict()
        
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.max_input_length = 0
        self.max_output_length = 0
        
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        
    def parse(self, input_s, output_s):
        """Given a pair of strings, normalize, tokenize and 
        update token indexes."""
        # Normalize
        input_s = input_s.strip().lower()
        output_s = output_s.strip().lower()
        
        # Tokenize and update tokens
        for token in self.tokenizer.tokenize(input_s):
            self.input_tokens.add(token)
        for token in self.tokenizer.tokenize(output_s):
            self.output_tokens.add(token)
        self.input_texts.append(input_s)
        self.output_texts.append(output_s)
    
    def __str__(self):
        s = ""
        s += "Number of samples: {}\n".format(len(self.input_texts))
        s += "Number of unique input tokens: {}\n".format(self.num_input_tokens)
        s += "Number of unique output tokens: {}\n".format(self.num_output_tokens)
        s += "Max sequence length for inputs: {}\n".format(self.max_input_length)
        s += "Max sequence length for outputs: {}\n".format(self.max_output_length)
        return s
    
    def compile(self):
        # Build input and output token indexes
        self.input_token_index = {token: i for i, token in enumerate(sorted(list(self.input_tokens)))}
        self.output_token_index = {token: i for i, token in enumerate(sorted(list(self.output_tokens)))}
        
        self.num_input_tokens = len(self.input_tokens)
        self.num_output_tokens = len(self.output_tokens)
        
        self.max_input_length = max([len(x) for x in self.input_texts])
        self.max_output_length = max([len(x) for x in self.output_texts])
        
        # To include the start sequence and end sequence tokens, add 2
        self.max_output_length += 2
        
        print(str(self))
        
        # Build input and target data arrays.
        # encoder and decoder input data will be integer encoded: each
        # token is mapped to its integer index. decoder target data will
        # be one-hot encoded to simplify loss computation.
        self.encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_input_length),
            dtype="int"
        )
        self.decoder_input_data = np.zeros(
            (len(self.output_texts), self.max_output_length),
            dtype="int"
        )
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_output_length, self.num_output_tokens),
            dtype="float32"
        )

        for i, (input_text, output_text) in enumerate(zip(self.input_texts, self.output_texts)):
            input_tokens = self.tokenizer.tokenize(input_text)

            # Add start sequence and end sequence tokens
            output_tokens = [self.start_token] + self.tokenizer.tokenize(output_text) + [self.end_token]
            
            for t, token in enumerate(input_tokens):
                self.encoder_input_data[i, t] = self.input_token_index[token]
            self.encoder_input_data[i, t + 1:] = self.input_token_index[" "]
            for t, token in enumerate(output_tokens):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t] = self.output_token_index[token] 
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    # decoder_target_data will have shape (batch, seq_len, output_dim)
                    self.decoder_target_data[i, t - 1, self.output_token_index[token]] = 1.0 
            self.decoder_input_data[i, t + 1:] = self.output_token_index["\n"]
            self.decoder_target_data[i, t:, self.output_token_index["\n"]] = 1.0           


def print_sample_predictions(model, data, start_index=128, stop_index=192):
    """Apply the model to a given slice of the 
    input data, and output the translated results"""
    n1, n2 = start_index, stop_index
    
    # Slice a batch of encoder inputs
    d_in = data.encoder_input_data[n1:n2,:]

    out = model.predict(d_in)
    print("English    French    Model-translation")
    for x, y, z in zip(
        data.input_texts[n1:n2],
        data.output_texts[n1:n2],
        out
        ):
        print('"{}"   "{}"   "{}"'.format(x.strip(), y.strip(), "".join(z).strip()))


def timeit(fn):

    def fn_timed(*args, **kw):
        t1 = time.time()
        result = fn(*args, **kw)
        t2 = time.time()
        print((t2 - t1) * 1000)
        return result

    return fn_timed


class CharTokenizer(object):
    '''Simply convert a string to a list of
    its characters'''

    def tokenize(self, s):
        return list(s)
