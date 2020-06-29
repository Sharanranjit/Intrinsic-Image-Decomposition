import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import predict, evaluate

import tensorflow as tf

def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def get_mpi_callbacks(model, basemodel, train_generator, test_generator, test_set, runPath):
    callbacks = []

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir):
            super().__init__(log_dir=log_dir)

            self.num_samples = 6
            self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
            self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)

        def on_epoch_end(self, epoch, logs=None):            
            if not test_set == None:
                # Samples using current model
                import matplotlib.pyplot as plt
                from skimage.transform import resize
                #plasma = plt.get_cmap('plasma')

                #minDepth, maxDepth = 10, 1000

                train_samples = []
                test_samples = []

                for i in range(self.num_samples):
                    x_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
                    x_test, y_test = test_generator[self.test_idx[i]]

                    x_train, y_train_alb, y_train_shad = x_train[0], (y_train['am_conv3'])[0], (y_train['sm_conv3'])[0]
                    x_test, y_test_alb, y_test_shad = x_test[0], (y_test['am_conv3'])[0], (y_test['sm_conv3'])[0]

                    h, w = y_train_alb.shape[0], y_train_alb.shape[1]

                    rgb_train = resize(x_train, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
                    rgb_test = resize(x_test, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)

                    gt_train_alb = y_train_alb[:,:,:3]
                    gt_test_alb = y_test_alb[:,:,:3]
                    gt_train_shad = y_train_shad[:,:,:3]
                    gt_test_shad = y_test_shad[:,:,:3]

                    predict_train_alb, predict_train_shad = predict(model, x_train)
                    predict_train_alb = predict_train_alb[0,:,:,:]
                    predict_train_shad = predict_train_shad[0,:,:,:]

                    predict_test_alb, predict_test_shad = predict(model, x_test)
                    predict_test_alb = predict_test_alb[0,:,:,:]
                    predict_test_shad = predict_test_shad[0,:,:,:]


                    train_samples.append(np.hstack([rgb_train, gt_train_alb, gt_test_shad, predict_train_alb, predict_train_shad]))
                    test_samples.append(np.hstack([rgb_test, gt_test_alb, gt_test_shad, predict_test_alb, predict_test_shad]))

                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.vstack(train_samples)))]), epoch)
                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.vstack(test_samples)))]), epoch)
                
                # Metrics
                e = evaluate(model, batch_size=6, verbose=True)
                logs.update({'rel': e[3]})
                logs.update({'rms': e[4]})
                logs.update({'log10': e[5]})

            super().on_epoch_end(epoch, logs)
    callbacks.append( LRTensorBoard(log_dir=runPath) )

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2)
    callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: save checkpoints
    #callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
        #verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=5))

    return callbacks