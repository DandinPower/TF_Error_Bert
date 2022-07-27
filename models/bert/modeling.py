from .encoder import BERTEncoder
from .block import EncoderBlock
from .layer import LinearLayer
import tensorflow as tf
import time 
from ..train.timer import GetTimeByDict
from tensorflow.python.framework import ops
from dotenv import load_dotenv
import os
load_dotenv()

ERROR_RATE = float(os.getenv('ERROR_RATE'))
FLIP_START = int(os.getenv('FLIP_START'))
FLIP_END = int(os.getenv('FLIP_END'))

@ops.RegisterGradient("RandomError")
def _bits_quant_grad(op, grad):
  inputs = op.inputs[0]
  return [grad] 

class BERTModel(tf.keras.Model):
    def __init__(self, config, parameters):
        super(BERTModel, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.parameters = parameters
        self.encoder = BERTEncoder(config,parameters)
        self.block1 = EncoderBlock(config,parameters,0,True)
        self.block2 = EncoderBlock(config,parameters,1,True)
        self.hidden = tf.keras.Sequential()
        tempLinearLayer = LinearLayer(config.numHiddens, config.numHiddens)
        tempLinearLayer.set_weights([parameters["hidden.0.weight"],parameters["hidden.0.bias"]])
        self.hidden.add(tempLinearLayer)
        self.hidden.add(tf.keras.layers.Activation('tanh'))

    def call(self, inputs):
        (tokens, segments, valid_lens) = inputs
        embeddingX = self.encoder((tokens,segments))
        X = self.block1((embeddingX, valid_lens))
        X = self.block2((X, valid_lens))
        X = self.kernel.random_error(self.hidden(X[:, 0, :]), ERROR_RATE, FLIP_START, FLIP_END)
        return X

    def LoadParameters(self):
        self.encoder.LoadParameters()
        self.block1.LoadParameters()
        self.block2.LoadParameters()

class BERTClassifier(tf.keras.Model):
    def __init__(self, config, parameters):
        super(BERTClassifier, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.config = config 
        self.parameters = parameters
        self.bert = BERTModel(config, self.parameters)
        self.classifier = LinearLayer(config.numHiddens, 2)

    def call(self, tokens):
        tempSegments = tokens * 0
        tempValid = self.GetValidLen(tokens)
        inputs = (tokens,tempSegments,tempValid)
        output = self.bert(inputs)
        output = self.kernel.random_error(self.classifier(output), ERROR_RATE, FLIP_START, FLIP_END)
        result = self.kernel.random_error(tf.nn.softmax(output), ERROR_RATE, FLIP_START, FLIP_END)
        return result

    def GetValidLen(self,inputs):
        tokens = inputs
        padding = tf.constant(1)
        temp = tf.math.equal(tokens, padding)
        paddingNums = tf.math.count_nonzero(temp,axis=1,dtype=tf.dtypes.float32)
        paddingNums = self.config.maxLen - paddingNums
        return paddingNums

    def LoadParameters(self):
        self.bert.LoadParameters()


    


    