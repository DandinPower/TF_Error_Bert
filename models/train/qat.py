import tensorflow as tf
import tensorflow_model_optimization as tfmot

def QuantizationAwareTraining(model,dataset):
    print(type(model))
    quantiza_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantiza_model(model)
    for inputs in dataset:
        singleData,singleLabels = inputs
        break
    output = q_aware_model(singleData)
    print(q_aware_model.summary())
