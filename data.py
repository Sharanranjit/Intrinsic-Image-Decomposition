import numpy as np
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def get_data(batch_size, data_zipfile='MPI.zip'):
    data = extract_zip(data_zipfile)

    mpi_train = list((row.split(',') for row in (data['MPI/data_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    mpi_test = list((row.split(',') for row in (data['MPI/data_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    #Shapes of input data for training
    shape_rgb_train = (batch_size, 320, 320, 3)
    shape_albedo_train = (batch_size, 320, 320, 3)
    shape_shading_train = (batch_size, 320, 320, 3)
    shape_output_train= (shape_albedo_train, shape_shading_train)

    #Shapes of input data for testing
    shape_rgb_test = (batch_size, 448, 1024, 3)
    shape_albedo_test = (batch_size, 448, 1024, 3)
    shape_shading_test = (batch_size, 448, 1024, 3)
    shape_output_test= (shape_albedo_test, shape_shading_test)
    
    # Helpful for testing...
    if False:
        mpi_train = mpi_train[:10]
        mpi_test = mpi_test[:10]

    return data, mpi_train, mpi_test, shape_rgb_train, shape_output_train, shape_rgb_test, shape_output_test

def get_mpi_train_test_data(batch_size):
    data, mpi_train, mpi_test, shape_rgb_train, shape_output_train, shape_rgb_test, shape_output_test = get_data(batch_size)

    train_generator = MPI_BasicAugmentRGBSequence(data, mpi_train, batch_size=batch_size, shape_rgb=shape_rgb_train, shape_output=shape_output_train)
    test_generator = MPI_BasicRGBSequence(data, mpi_test, batch_size=batch_size, shape_rgb=shape_rgb_test, shape_output=shape_output_test)

    return train_generator, test_generator

class MPI_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_output, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_output = shape_output
        
        #Shuffling dataset as continous frames can lead to overfitting
        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_inp, batch_alb, batch_shad = np.zeros(self.shape_rgb), np.zeros(self.shape_output[0]), np.zeros( self.shape_output[1])

        # Augmentation of RGB images
        for i in range(batch_inp.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open(BytesIO(self.data[sample[0]])))/255,0,1)
            y1 = np.clip(np.asarray(Image.open(BytesIO(self.data[sample[1]])))/255,0,1)
            y2 = np.clip(np.asarray(Image.open(BytesIO(self.data[sample[2]])))/255,0,1)

            batch_inp[i] = x      #input
            batch_alb[i] = y1   #albedo
            batch_shad[i] = y2   #shading

            if is_apply_policy: batch_inp[i], batch_alb[i], batch_shad[i] = self.policy(batch_inp[i], batch_alb[i], batch_shad[i])

            # DEBUG:
            #self.policy.debug_img(batch_inp[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_inp, {'am_conv3' : batch_alb, 'sm_conv3' : batch_shad}

class MPI_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_output):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_output = shape_output
        #self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_inp, batch_alb, batch_shad = np.zeros(self.shape_rgb), np.zeros(self.shape_output[0]), np.zeros( self.shape_output[1])
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray((Image.open(BytesIO(self.data[sample[0]]))).resize((1024,448)))/255,0,1)
            y1 = np.clip(np.asarray((Image.open(BytesIO(self.data[sample[1]]))).resize((1024,448)))/255,0,1)
            y2 = np.clip(np.asarray((Image.open(BytesIO(self.data[sample[2]]))).resize((1024,448)))/255,0,1)

            batch_inp[i] = x      #input
            batch_alb[i] = y1   #albedo
            batch_shad[i] = y2   #shading

            # DEBUG:
            #self.policy.debug_img(batch_inp[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_inp, {'am_conv3' : batch_alb, 'sm_conv3' : batch_shad}