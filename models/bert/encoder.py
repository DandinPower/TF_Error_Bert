import tensorflow as tf
import sys
import time 
from tensorflow.python.framework import ops
from dotenv import load_dotenv
import os
load_dotenv()

ERROR_RATE = float(os.getenv('ERROR_RATE'))
FLIP_START = int(os.getenv('FLIP_START'))
FLIP_END = int(os.getenv('FLIP_END'))

class AddParameter(tf.keras.layers.Layer):
    def __init__(self, nums,hiddens):
        super().__init__()
        self.w = self.add_variable(name='weight',shape=[nums,hiddens], initializer=tf.zeros_initializer())

    def call(self, inputs):
        return inputs + self.w

class BERTEncoder(tf.keras.Model):
    def __init__(self, config, parameters):
        super(BERTEncoder, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.token_embedding = tf.keras.layers.Embedding(config.vocabSize, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.token_embedding.weight"]])
        self.segment_embedding = tf.keras.layers.Embedding(2, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.segment_embedding.weight"]])
        self.pos_embedding = AddParameter(config.maxLen,config.numHiddens)
        self.config = config
        self.parameters = parameters

    def call(self, inputs):
        (tokens,segments) = inputs
        X = self.kernel.random_error(self.token_embedding(tokens), ERROR_RATE, FLIP_START, FLIP_END)
        X = self.kernel.random_error(X + self.kernel.random_error(self.segment_embedding(segments), ERROR_RATE, FLIP_START, FLIP_END), ERROR_RATE, FLIP_START, FLIP_END)
        X = self.kernel.random_error(self.pos_embedding(X), ERROR_RATE, FLIP_START, FLIP_END)
        return X

    def LoadParameters(self):
        self.pos_embedding.set_weights(self.parameters["encoder.pos_embedding"])
        
