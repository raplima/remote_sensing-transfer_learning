# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:59:00 2019

@author: Rafael Pires de Lima
check filters output for models to understand why they are predicting a single class
"""
import os
import random
from keras import applications
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import set_random_seed

import numpy as np

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

# global variable
BATCH_SIZE = 32

def model_preprocess(model_name):
    """Loads the appropriate CNN preprocess
      Args:
        arch: String key for model to be loaded.
      Returns:
        The specified Keras preprocessing.
      """
    # function that loads the appropriate model
    if model_name == 'Xception':
        return applications.xception.preprocess_input
    elif model_name == 'VGG16':
        return applications.vgg16.preprocess_input
    elif model_name == 'VGG19':
        return applications.vgg19.preprocess_input
    elif model_name == 'ResNet50':
        return applications.resnet50.preprocess_input
    elif model_name == 'InceptionV3':
        return applications.inception_v3.preprocess_input
    elif model_name == 'InceptionResNetV2':
        return applications.inception_resnet_v2.preprocess_input
    elif model_name == 'MobileNet':
        return applications.mobilenet.preprocess_input
    elif model_name == 'DenseNet121':
        return applications.densenet.preprocess_input
    elif model_name == 'NASNetLarge':
        return applications.nasnet.preprocess_input
    elif model_name == 'MobileNetV2':
        return applications.mobilenet_v2.preprocess_input
    else:
        print('Invalid model selected')
        return False


if __name__ == "__main__":
    # set the seed for repetition purposes 
    random.seed(0)
    np.random.seed(0)
    set_random_seed(0)
    
    data_dir = '../data'
    image_dir = '../images'
    model_dir = '../models'
    
    models = [os.path.join(model_dir, mod) for mod in ['VGG19_shallow_UCMerced_rand_init.hdf5', 
              'VGG19_shallow_UCMerced_frozen.hdf5']]
    model_names = ['VGG19', 
                   'VGG19']
    
    test_dir = os.path.join(data_dir, 'out_check')
    
    fig, ax = plt.subplots(ncols=len(model_names), figsize=(26,16), sharey=True)

    for axi, (model_file, model_name) in enumerate(zip(models, model_names)):
       
        model = load_model(model_file)
        
        # set the generator
        datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))

        generator = datagen.flow_from_directory(
                            test_dir,
                            target_size=model.input_shape[1:3],
                            batch_size=1,
                            class_mode='categorical',
                            shuffle=False)

        for ii, layer_name in enumerate([
                                         'block1_conv1', 
                                         'block1_conv2'
                                         ]):
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict_generator(generator, 
                                        verbose=0, 
                                        steps=len(generator)
    									)
            x = np.arange(1, intermediate_output.shape[-1]+1)
            for ii, test_image in enumerate(generator.filenames):
                ax[axi].bar(x+ii/10, np.median(intermediate_output[ii], axis=(0,1)), label=f'{layer_name}-{test_image}')
            ax[axi].legend()
            ax[axi].set_title(f'{os.path.basename(model_file)}')

        print(np.argmax(model.predict_generator(generator, verbose=0, steps=len(generator)), axis=1))
    
    fig.savefig(os.path.join(image_dir, 'out_check-layer_output.pdf'))

    # plot layer weights        
    nrows = 0
    for layer in model.layers:
        if 'conv' in layer.name:
            nrows+=1
    
    fig, ax = plt.subplots(nrows=nrows, figsize=(16,10))
    fig2, ax2 = plt.subplots(nrows=nrows, figsize=(16,10))

    for model_file in models:
        model = load_model(model_file)

        for ii, layer in enumerate(model.layers[1:]):
            if 'conv' in layer.name:
                print(layer.name)
                weights, biases = layer.get_weights()
                x = np.arange(1, len(biases)+1)
                #ax[ii].bar(x+ii/10, biases, label=f'{os.path.basename(model_file)}-{layer_name}')
                ax[ii].plot(biases, label=f'{os.path.basename(model_file)}-{layer.name}')

                ax[ii].set_title(f'{layer.name}')
                
                ax[ii].legend()

                #w eights
                ax2[ii].plot(np.median(weights, axis=(0,1,2)), 
                               label=f'{os.path.basename(model_file)}-{layer.name}')

                ax2[ii].set_title(f'{layer.name}')
                
                ax2[ii].legend()

    fig.savefig(os.path.join(image_dir, 'out_check-layer_biases.pdf'))
    fig2.savefig(os.path.join(image_dir, 'out_check-layer_weights.pdf'))

