import tensorflow as tf
from tensorflow.python.framework import ops
from dotenv import load_dotenv
import os
load_dotenv()

ERROR_RATE = float(os.getenv('ERROR_RATE'))
FLIP_START = int(os.getenv('FLIP_START'))
FLIP_END = int(os.getenv('FLIP_END'))

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = self.kernel.random_error(tf.matmul(inputs, self.w), ERROR_RATE, FLIP_START, FLIP_END) + self.kernel.random_error(self.b, ERROR_RATE, FLIP_START, FLIP_END)
        return self.kernel.random_error(y_pred, ERROR_RATE, FLIP_START, FLIP_END)

class AddNorm(tf.keras.Model):
    def __init__(self, dropout):
        super(AddNorm, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        return self.kernel.random_error(self.ln(self.dropout(Y) + X), ERROR_RATE, FLIP_START, FLIP_END)

class PositionWiseFFN(tf.keras.Model):
    def __init__(self, config, parameters,index):
        super(PositionWiseFFN, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.config = config 
        self.parameters = parameters 
        self.index = index 
        self.dense1 = LinearLayer(config.ffnNumInput, config.ffnNumHiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = LinearLayer(config.ffnNumHiddens, config.ffnNumInput)

    def call(self, X):
        return self.dense2(self.kernel.random_error(self.relu(self.dense1(X)), ERROR_RATE, FLIP_START, FLIP_END))

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])