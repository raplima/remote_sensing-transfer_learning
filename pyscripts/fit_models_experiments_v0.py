# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:39:36 2019

@author: Rafael Pires de Lima
functions to test transfer learning and fine tuning of remote sensing datasets
This script contains class generator and function to save all features instead of 
using keras generators. 
"""
import os
from shutil import rmtree
import random
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from keras import backend as K 

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from helper import get_model_memory_usage, reset_keras
from plots import plot_history, cf_matrix

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

class feats_generator(Sequence):
    """
    A generator modified from 
    https://keras.io/utils/#sequence
    and 
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    to read the npy files 
    """
    def __init__(self, data_dir, n_classes, dim, n_channels, batch_size, shuffle=True):
        self.npy_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.n_classes = n_classes
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle == True:
            np.random.shuffle(self.npy_filenames)

    def __len__(self):
        return np.int(np.ceil(len(self.npy_filenames) / float(self.batch_size)))

    def on_epoch_end(self):
      # shuffle file list
      if self.shuffle == True:
          np.random.shuffle(self.npy_filenames)

    def __getitem__(self, idx):
        batch_files = self.npy_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=np.int)

        # Generate data
        for i, npy_file in enumerate(batch_files):
            # Store sample
            X[i,] = np.load(npy_file)

            # Store class
            y_temp = os.path.splitext(
                    os.path.basename(npy_file)
                    )[0].split('label_')[-1]
            y[i] = np.int(y_temp)
            print(type(y[i]), y[i])
            assert type(y[i]) == np.int32
            print(f'{y[i]:3}  {os.path.basename(npy_file)}')

        return X, to_categorical(y, num_classes=self.n_classes)

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

def load_part_model(model_name, depth_level, weights='imagenet'):
    """
    function to load the model according to the depth provided
    :param model_name: name of the cnn model to be loaded (currently only 
                       supports InceptionV3 and VGG19)
    :param depth_level: one of [shallow, intermediate, deep]
    :weights: one of 'imagenet' or None
    :return: the model to the desired depth level with imagenet or random weights
    """
    
    #define a dictionary of dictionaries to help load the appropriate model:
    model_dict = {'InceptionV3' : {'shallow'      : 'mixed2',
                                   'intermediate' : 'mixed7',
                                   'deep'         : 'mixed10',
                                    },
                  'VGG19'       : {'shallow'      : 'block1_pool',
                                   'intermediate' : 'block4_pool',
                                   'deep'         : 'block5_pool',
                                    },
    
                 }
    
    if model_name=='InceptionV3':
        # load the entire model:
        base_model = applications.InceptionV3(weights=weights)
    if model_name=='VGG19':
        base_model = applications.VGG19(weights=weights)
    
    model = Model(inputs=base_model.input, 
                 outputs=base_model
                 .get_layer(model_dict[model_name][depth_level]).output)
    
    return model
    
def feature_extract_fit(model_name, depth_level, weights, 
                        data_in,
                        train_dir, valid_dir, test_dir, 
                        image_dir):
    """
    function that uses a cnn model to extract features and train a small
    classification NN on top of the extracted features. 
    :param model_name: name of the cnn model to be loaded (currently only 
                       supports InceptionV3 and VGG19)
    :param depth_level: one of [shallow, intermediate, deep]
    :weights: one of ['imagenet', None]
    :data_in: a tag to keep track of input data used
    :train_dir: training folder
    :valid_dir: validation folder
    :test_dir:  test folder
    :image_dir: image output location
    :return: 
        the entire trained model
        the test set accuracy
        the test set confusion matrix
    also saves a plot of the training loss/accuracy
    """
    ########
    # common trainig parameters:
    ########
    batch_size = 8
    epochs = 8
    lrate = 1e-4
    loss = 'categorical_crossentropy'
    opt = SGD(lr=lrate, momentum=0.0, clipvalue=5.)
    
    train_val_dict = {'train': train_dir,
                      'validation' : valid_dir}
    
    # load the base model:
    base_model = load_part_model(model_name, depth_level, weights)
    
    # save the number of classes:
    num_classes = len(os.listdir(train_dir))
    
    #################################
    # Features can be too large to fit in memory. 
    # Save the features as npy files (x, y in the same npy file)
    #################################
    # set the generator
    datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))
    
    # dictionary that will hold the folder names - to be populated inside loop
    feat_train_val_dict = {}
    # do the same thing for both training and validation:    
    for dset in tqdm(train_val_dict):
        # extract training features, shuffling now as (x,y) will be combined in the 
        # same file. Each file contains a single image feature (X) and its 
        # label (y) will be part of the name
        generator = datagen.flow_from_directory(
            train_val_dict[dset],
            target_size=base_model.input_shape[1:3],
            batch_size=1,
            class_mode='sparse',
            shuffle=False) 
    
        if num_classes != generator.num_classes:
            print('Warning! Different number of classes in training and validation')
        
        # create output folder
        folder_out = os.path.join(os.path.dirname(train_val_dict[dset]),
                                  f"feat_{dset}")
        rmtree(folder_out, ignore_errors=True)
        os.mkdir(folder_out)
        # save location into dictionary:
        feat_train_val_dict[dset] = folder_out
        
        # loop through all the samples, saving them in the appropriate folder
        sample = 0
        for X_set, y_set in generator: 
            X_feat = base_model.predict(X_set)
            
            np.save(os.path.join(folder_out, 
                                 f'sample_{sample}_label_{np.int(y_set[0])}.npy'), 
                    X_feat)
            sample+=1

            if sample == len(generator):
                break;
    
    # generator parameters
    params = {'dim': base_model.output_shape[1:3],
              'batch_size': batch_size,
              'n_classes': num_classes,
              'n_channels': base_model.output_shape[3]}
        
    # Generators
    train_generator = feats_generator(data_dir=feat_train_val_dict['train'], 
                                      shuffle=True, 
                                      **params)
    validation_generator = feats_generator(data_dir=feat_train_val_dict['validation'],
                                           shuffle=False, 
                                           **params)

    #################################
    # Features saved, generators set
    #################################
        
    # create the top model:
    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:4]))
    #top_model.add(Conv2D(512))
    #top_model.add(GlobalAveragePooling2D())
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(rate=0.5))  
    # the last layer depends on the number of classes
    top_model.add(Dense(num_classes, activation='softmax'))
           
    # train model with parameters specified at the top of the function
    top_model.compile(optimizer=opt,
                                 loss=loss,
                                 metrics=['accuracy'])

    history = top_model.fit_generator(generator=train_generator,
                                      validation_data=validation_generator,
                                      epochs=epochs)
    
    # plot and save the training history:
    plot_history(history, model_name, depth_level, data_in, 'feat_extr', image_dir)
    
    # set the entire model:
    # build the network
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    model.compile(optimizer=opt,
                  loss=loss, 
                  metrics=['accuracy'])
    
    # predict test values accuracy
    generator = datagen.flow_from_directory(
                        test_dir,
                        target_size=base_model.input_shape[1:3],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)

    pred = model.predict_generator(generator, 
                                    verbose=0, 
                                    steps=len(generator))
    
     # save results as dataframe
    df = pd.DataFrame(pred, columns=generator.class_indices.keys())
    df['file'] = generator.filenames
    df['true_label'] = df['file'].apply(os.path.dirname).apply(str.lower)
    df['pred_idx'] = np.argmax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    # save as the label (dictionary comprehension because generator.class_indices has the
    # key,values inverted to what we want
    df['pred_label'] = df['pred_idx'].map({value: key for key, value in generator.class_indices.items()}).apply(
        str.lower)
    # save the maximum probability for easier reference:
    df['max_prob'] = np.amax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    
    y_true = df['true_label']
    y_pred = df['pred_label']
    
    # compute accuracy:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    # compute confusion matrix:
    cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    # clear Keras/backend session (freeing memory and preventing slowdown)
    K.clear_session()
    
    return top_model, acc, cfm

def train_frozen(model_name, depth_level, weights, 
                        data_in,
                        train_dir, valid_dir, test_dir, 
                        image_dir, 
                        model_dir,
                        epochs):
    """
    function that freezes a cnn model to extract features and train a small
    classification NN on top of the extracted features. 
    :param model_name: name of the cnn model to be loaded (currently only 
                       supports InceptionV3 and VGG19)
    :param depth_level: one of [shallow, intermediate, deep]
    :weights: one of ['imagenet', None]
    :data_in: a tag to keep track of input data used
    :train_dir: training folder
    :valid_dir: validation folder
    :test_dir:  test folder
    :image_dir: image output location
    :model_dir: model output location
    :epochs: number of epochs to train
    :return: 
        the test set accuracy
        the test set confusion matrix
    also saves a plot of the training loss/accuracy
    saves the model 
    saves a plot of the confusion matrix
    """
    ########
    # common trainig parameters:
    ########
    batch_size = 32
    lrate = 1e-3
    loss = 'categorical_crossentropy'
    opt = SGD(lr=lrate, momentum=0.0, clipvalue=5.)
    
    # load the base model:
    base_model = load_part_model(model_name, depth_level, weights)
    # freeze layers (layers will not be updated during the first training process)
    for layer in base_model.layers:
    	layer.trainable = False
    
    # save the number of classes:
    num_classes = len(os.listdir(train_dir))
    
    # set the generator
    datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))
    
    # do the same thing for both training and validation:    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=base_model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True) 
    
    valid_generator = datagen.flow_from_directory(
        valid_dir,
        target_size=base_model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) 
    
    if valid_generator.num_classes != train_generator.num_classes:
        print('Warning! Different number of classes in training and validation')

    #################################
    # generators set
    #################################
        
    # create the top model:
    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:4]))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(rate=0.5))  
    # the last layer depends on the number of classes
    top_model.add(Dense(num_classes, activation='softmax'))
           
    # set the entire model:
    # build the network
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(optimizer=opt,
              loss=loss, 
              metrics=['accuracy'])
    #print(model.summary())
    print(f'model needs {get_model_memory_usage(batch_size, model)} Gb')

    history = model.fit_generator(generator=train_generator,
                                  validation_data=valid_generator,
                                  shuffle=True,
                                  epochs=epochs)
    
    # save model:
    model.save(os.path.join(model_dir, f"{model_name}_{depth_level}_{data_in}_frozen.hdf5"))
    
    # plot and save the training history:
    plot_history(history, model_name, depth_level, data_in, 'feat_extr', image_dir)
        
    # predict test values accuracy
    generator = datagen.flow_from_directory(
                        test_dir,
                        target_size=base_model.input_shape[1:3],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)

    pred = model.predict_generator(generator, 
                                    verbose=0, 
                                    steps=len(generator))
    
     # save results as dataframe
    df = pd.DataFrame(pred, columns=generator.class_indices.keys())
    df['file'] = generator.filenames
    df['true_label'] = df['file'].apply(os.path.dirname).apply(str.lower)
    df['pred_idx'] = np.argmax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    # save as the label (dictionary comprehension because generator.class_indices has the
    # key,values inverted to what we want
    df['pred_label'] = df['pred_idx'].map({value: key for key, value in generator.class_indices.items()}).apply(
        str.lower)
    # save the maximum probability for easier reference:
    df['max_prob'] = np.amax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    
    y_true = df['true_label']
    y_pred = df['pred_label']
    
    # compute accuracy:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    # compute confusion matrix:
    cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    # plot confusion matrix:
    cf_matrix(y_true, y_pred, image_dir, 
              plot_name=f"conf_matrix_{data_in}_{model_name}_{depth_level}_feat_extr.png", 
              dpi=200)
    
    # clear memory
    reset_keras()
    
    return acc, cfm

def fine_tune(model_filename, model_name, 
              data_in,
              train_dir, valid_dir, test_dir, 
              image_dir, 
              model_dir,
              epochs):
    """
    function that fine tunes a pre-trained cnn model using a small learning rate
    :param model_filename: filename of hdf5 file (Keras model) to be loaded
    :param model_name: name of the original cnn model - necessary for preprocessing
        (currently only supports InceptionV3 and VGG19)
    :data_in: a tag to keep track of input data used
    :train_dir: training folder
    :valid_dir: validation folder
    :test_dir:  test folder
    :image_dir: image output location
    :model_dir: model output location
    :epochs: number of epochs to train
    :return: 
        the entire trained model
        the test set accuracy
        the test set confusion matrix
    also saves a plot of the training loss/accuracy
    """
    ########
    # common trainig parameters:
    ########
    batch_size = 32
    lrate = 5*1e-5
    loss = 'categorical_crossentropy'
    opt = SGD(lr=lrate, momentum=0.0, clipvalue=5.)
    
    # load the model:
    model = load_model(os.path.join(model_dir, model_filename))
    
    # make sure all layers are trainable
    for layer in model.layers:
    	layer.trainable = True
    
    # set the generator
    datagen = ImageDataGenerator(preprocessing_function=model_preprocess(model_name))
    
    # do the same thing for both training and validation:    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True) 
    
    valid_generator = datagen.flow_from_directory(
        valid_dir,
        target_size=model.input_shape[1:3],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) 
    
    if valid_generator.num_classes != train_generator.num_classes:
        print('Warning! Different number of classes in training and validation')

    #################################
    # generators set
    #################################
        
    model.compile(optimizer=opt,
              loss=loss, 
              metrics=['accuracy'])
    print(f'model needs {get_model_memory_usage(batch_size, model)} Gb')

    history = model.fit_generator(generator=train_generator,
                                  validation_data=valid_generator,
                                  shuffle=True,
                                  epochs=epochs)
    
    # plot and save the training history:
    plot_history(history, model_name, depth_level, data_in, 'fine_tune', image_dir)
        
    # predict test values accuracy
    generator = datagen.flow_from_directory(
                        test_dir,
                        target_size=model.input_shape[1:3],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)

    pred = model.predict_generator(generator, 
                                    verbose=0, 
                                    steps=len(generator))
    
     # save results as dataframe
    df = pd.DataFrame(pred, columns=generator.class_indices.keys())
    df['file'] = generator.filenames
    df['true_label'] = df['file'].apply(os.path.dirname).apply(str.lower)
    df['pred_idx'] = np.argmax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    # save as the label (dictionary comprehension because generator.class_indices has the
    # key,values inverted to what we want
    df['pred_label'] = df['pred_idx'].map({value: key for key, value in generator.class_indices.items()}).apply(
        str.lower)
    # save the maximum probability for easier reference:
    df['max_prob'] = np.amax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    
    y_true = df['true_label']
    y_pred = df['pred_label']
    
    # compute accuracy:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    # compute confusion matrix:
    cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    cf_matrix(y_true, y_pred, image_dir, 
          plot_name=f"conf_matrix_{data_in}_{model_name}_{depth_level}_fine_tune.png", 
          dpi=200)

    # clear memory
    reset_keras()

    return acc, cfm


if __name__ == "__main__":
    # set the seed for repetition purposes 
    random.seed(0)
    
    data_dir = '../data'
    image_dir = '../images'
    model_dir = '../models'
    datasets_dir = [os.path.join(data_dir, 'dogs'),
                    os.path.join(data_dir, 'AID'),
                    os.path.join(data_dir, 'PatternNet'), 
                    os.path.join(data_dir, 'UCMerced_LandUse')]
    
    models = ['VGG19', 'InceptionV3']
    depth_levels = ['shallow', 
                    'intermediate', 
                    'deep']
    epochs = 3
    
    # test different models:
    acc_dict={}
    dict_counter = 0
    weights = 'imagenet'
    for dataset in datasets_dir:
        for model in models:
            for depth_level in depth_levels:
                print(f'{model:20} {depth_level:20} {dataset}')
                data_in   = os.path.basename(dataset)
                train_dir = os.path.join(dataset, 'images_train')
                valid_dir = os.path.join(dataset, 'images_validation')
                test_dir  = os.path.join(dataset, 'images_test')
                pre_mod, acc, _ = train_frozen(model, 
                                         depth_level, 
                                         weights, 
                                         data_in,
                                         train_dir,
                                         valid_dir, 
                                         test_dir, 
                                         image_dir, 
                                         model_dir,
                                         epochs)
                
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"    :model,
                                         "depth"   :depth_level,
                                         "dataset" :dataset,
                                         "mode"    :'transfer learning',
                                         "accuracy":acc}
                dict_counter+=1
                
                # fine tune:
                model_filename = f"{model}_{depth_level}_{data_in}_frozen.hdf5"
                pre_mod, acc, _ = fine_tune(model_filename, model, 
                                             data_in,
                                             train_dir,
                                             valid_dir, 
                                             test_dir, 
                                             image_dir, 
                                             model_dir,
                                             epochs)
                
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"   :model,
                                         "depth"   :depth_level,
                                         "dataset" :dataset,
                                         "mode"    :'fine tune',
                                         "accuracy":acc}
                dict_counter+=1
                
                reset_keras()


    # print accuracy results
    for k in acc_dict:
        print(acc_dict[k])