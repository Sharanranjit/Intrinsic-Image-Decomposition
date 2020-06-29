import os
import glob
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from losses import albedo_loss_function, shading_loss_function
from utils import evaluate
import tensorflow as tf

# Argument Parser
parser = argparse.ArgumentParser(description='Learning Inverse Rendering using a single image')
parser.add_argument('--model', type=str, help='Trained Keras model file.')
parser.add_argument('--test_data', type=str, help='MPI-Sintel Test split.')
args = parser.parse_args()

#initialize the inputs for loss functions with zero tensors(can by any 3 rank tensor)
inputs = tf.zeros((1, 320, 320, 3))
alb = tf.zeros((1, 320, 320, 3))
shad = tf.zeros((1, 320, 320, 3))

# Custom object needed for inference and training
custom_objects = {  'BilinearUpSampling2D': BilinearUpSampling2D, 
					'albedo_loss_function': albedo_loss_function(inputs,shad), 
					'shading_loss_function' : shading_loss_function(inputs,alb)}

# Load model into GPU / CPU
print('Loading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)


start = time.time()
print('Testing...')

e_alb, e_shad = evaluate(model, batch_size=6, verbose=True, data_zip_file = args.test_data)

end = time.time()
print('\nTest time: ', end-start, 's')