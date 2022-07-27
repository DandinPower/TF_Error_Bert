import pickle
from ..bert.configs import Config 
from ..bert.modeling import BERTClassifier
from dotenv import load_dotenv
import tensorflow as tf
load_dotenv()
import os 
PARAMETER_PATH = os.getenv('PARAMETER_PATH')

def load_variable(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def WriteTfLite(modelPath,savePath):
    tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model(modelPath)
    tf_lite_converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = tf_lite_converter.convert()
    open(savePath,"wb").write(tflite_model)

def WriteInt8TFLite(modelPath,savePath):
    tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model(modelPath)
    tf_lite_converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = tf_lite_converter.convert()
    open(savePath,"wb").write(tflite_model)

def LoadModel(savePath):
    newModel = tf.saved_model.load(savePath)
    #model = joblib.load(f'{savePath}fintune.pkl')
    return newModel 

def SaveModel(model,savePath):
    #model.save(savePath, save_format='tf')
    tf.saved_model.save(model,savePath)

class Parameters():
    def __init__(self,data):
        self.data = data

    def __getitem__(self,key):
        return self.data[key]

