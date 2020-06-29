from keras import applications
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from keras import Model
from layers import BilinearUpSampling2D
from losses import albedo_loss_function, shading_loss_function

def create_model(existing=''):
    
    if len(existing)== 0 :
        #print('Loading Encoder(DenseNet)')
        encoder=applications.DenseNet121(input_shape=(None,None,3), include_top=False)
        #encoder=applications.DenseNet169(input_shape=(None,None,3), include_top=False)

        encoder_output_shape=encoder.layers[-1].output.shape
        
        for layer in encoder.layers : 
            layer.trainable = True
        
        decode_filters = int(int(encoder_output_shape[-1])/2)
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, encoder.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers for Albedo Map
        
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=encoder_output_shape, name='am_conv2')(encoder.output)

        decoder = upproject(decoder, int(decode_filters/2), 'am_up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'am_up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'am_up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'am_up4', concat_with='conv1/relu')
        decoder = upproject(decoder, int(decode_filters/32), 'am_up5', concat_with='input_1')

        # Extract albedo (final layer)
        am_conv3 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='am_conv3')(decoder)
        
        # Decoder Layers for Shading Map

        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=encoder_output_shape, name='sm_conv2')(encoder.output)

        decoder = upproject(decoder, int(decode_filters/2), 'sm_up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'sm_up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'sm_up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'sm_up4', concat_with='conv1/relu')
        decoder = upproject(decoder, int(decode_filters/32), 'sm_up5', concat_with='input_1')

        # Extract shading (final layer)
        sm_conv3 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='sm_conv3')(decoder)

        # Create the model
        model = Model(inputs=encoder.input, outputs=[am_conv3,sm_conv3])
        

    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'albedo_loss_function': albedo_loss_function, 'shading_loss_function': shading_loss_function }
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')

    return model
    