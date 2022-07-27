from progressbar import ProgressBar
from ..bert.modeling import BERTClassifier
from ..bert.configs import Config
from ..preprocess.load import load_variable,Parameters
from dotenv import load_dotenv
import os
load_dotenv()
import tensorflow as tf
import time

PARAMETER_PATH = os.getenv('PARAMETER_PATH')

def Inference(model,dataset):
    print('Inferencing...')
    metrics=tf.metrics.SparseCategoricalAccuracy()
    startTime = time.time()
    j = 0
    total = len(dataset)
    pBar = ProgressBar().start()
    for data in dataset:
        X, y = data
        y_pred = model(X)
        metrics.update_state(y, y_pred)
        pBar.update(int((j / (total - 1)) * 100))
        j += 1
    pBar.finish()
    print(f'cost time: {round(time.time() - startTime,3)} sec')
    print(f'test accuracy:{metrics.result().numpy()}')

def Train(model,dataset,lr,num_epochs,savePath):
    print('Training...')
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)  
    metrics=tf.metrics.SparseCategoricalAccuracy()
    for x in range(num_epochs):
        startTime = time.time()
        j = 0
        total = len(dataset)
        pBar = ProgressBar().start()
        for data in dataset:
            X, y = data
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                metrics.update_state(y, y_pred)
                loss = tf.reduce_mean(loss)          
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            pBar.update(int((j / (total - 1)) * 100))
            j += 1
        pBar.finish()
        print(f'cost time: {round(time.time() - startTime,3)} sec')
        print(f'epoch:{x} accuracy:{metrics.result().numpy()}')
        metrics.reset_states()
    return model
