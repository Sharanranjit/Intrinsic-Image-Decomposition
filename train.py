import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Keras / TensorFlow
from losses import albedo_loss_function, shading_loss_function
from utils import predict, save_images, load_images, display_images, evaluate
from model import create_model
from data import get_mpi_train_test_data
from callbacks import get_mpi_callbacks

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser(description='Learning Inverse Rendering using a single image')
parser.add_argument('--data', default='mpi', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--name', type=str, default='dense_IRN', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1: 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# Create the model
model = create_model( existing=args.checkpoint )

input_batch = model.layers[0].output
albedo_pred = model.layers[-2].output
shading_pred = model.layers[-1].output
# Data loaders
if args.data == 'mpi': 
    train_generator, test_generator = get_mpi_train_test_data( args.bs )
    
# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

#Extra Info about training
if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

# Multi-gpu setup:
basemodel = model
if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

# Optimizer & Losses
optimizer = Adam(lr=args.lr, amsgrad=True)
loss = {"am_conv3" : albedo_loss_function(input_batch, shading_pred), "sm_conv3" : shading_loss_function(input_batch, albedo_pred)}
# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'mpi': callbacks = get_mpi_callbacks(model, basemodel, train_generator, test_generator, () if args.full else None , runPath)

# Start training
history = model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

print(' Training done!')