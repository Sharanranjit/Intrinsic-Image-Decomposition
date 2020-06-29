import os
import glob
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from losses import albedo_loss_function, shading_loss_function
from utils import predict,load_images
import tensorflow as tf
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser(description='Learning Inverse Rendering using a single image')
parser.add_argument('--input', type=str, help='Any input image')
parser.add_argument('--model', type=str, help='Trained Keras model file.')
parser.add_argument('--output_dir', type=str, help='Directory to save output images.')
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
#print('Testing...')
inputs = load_images(args.input)
(albedo,shading) = predict(model, inputs)
end = time.time()

#Save images
albedo = (255.0 / albedo.max() * (albedo - albedo.min())).astype(np.uint8)
shading = (255.0 / shading.max() * (shading - shading.min())).astype(np.uint8)
albedo = Image.fromarray(albedo)
shading = Image.fromarray(shading)

albedo.save(args.output_dir+"albedo.png")
shading.save(args.output_dir+"shading.png")


print('\nTime taken: ', end-start, 's')