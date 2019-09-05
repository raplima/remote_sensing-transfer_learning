# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:39:36 2019

@author: Rafael Pires de Lima
functions to test transfer learning and fine tuning of remote sensing datasets
"""
import os
import random
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import SGD
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import set_random_seed

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from helper import get_model_memory_usage, reset_keras
from plots import plot_history, cf_matrix

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

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
    
def train_frozen(model_name, depth_level, weights, 
                        data_in,
                        train_dir, valid_dir, test_dir, 
                        image_dir, 
                        model_dir,
                        epochs, 
                        opt):
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
    :opt:  string that maps to keras optmizer to be used for training
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
    batch_size = 16

    loss = 'categorical_crossentropy'
    
    opts = {'SGD (1e-2)'          : optimizers.SGD(lr=0.01, momentum=0.0, clipvalue=5.), 
        'SGD (1e-2) momentum 0.5' : optimizers.SGD(lr=0.01, momentum=0.5, clipvalue=5.), 
        'SGD (1e-2) momentum 0.9' : optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=5.), 
        'SGD (1e-3)'              : optimizers.SGD(lr=0.001, momentum=0.0, clipvalue=5.), 
        'SGD (1e-3) momentum 0.5' : optimizers.SGD(lr=0.001, momentum=0.5, clipvalue=5.), 
        'SGD (1e-3) momentum 0.9' : optimizers.SGD(lr=0.001, momentum=0.9, clipvalue=5.), 
        'RMSprop (1e-3) '         : optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        'Adam (1e-2)'             : optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        'Adamax (2e-3)'           : optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        }
    opt = opts[opt]

    
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
                                  steps_per_epoch=len(train_generator),
                                  validation_steps=len(valid_generator),
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
                                    steps=len(generator)
									)
    
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
              dpi=1000)
    
    # clear memory
    reset_keras()
    
    return acc, cfm, history

def fine_tune(model_filename, model_name, depth_level,
              data_in,
              train_dir, valid_dir, test_dir, 
              image_dir, 
              model_dir,
              epochs, 
              opt):
    """
    function that fine tunes a pre-trained cnn model using a small learning rate
    or trains a model from scracth
    :param model_filename: filename of hdf5 file (Keras model) to be loaded
    :param model_name: name of the original cnn model - necessary for preprocessing
        (currently only supports InceptionV3 and VGG19)
    :param depth_level: one of [shallow, intermediate, deep]
    :data_in: a tag to keep track of input data used
    :train_dir: training folder
    :valid_dir: validation folder
    :test_dir:  test folder
    :image_dir: image output location
    :model_dir: model output location
    :epochs: number of epochs to train
    :opt: string that maps to  keras optmizer to be used for training
    :return: 
        the entire trained model
        the test set accuracy
        the test set confusion matrix
    also saves a plot of the training loss/accuracy
    """
    ########
    # common trainig parameters:
    ########
    
    batch_size = 16
    loss = 'categorical_crossentropy'
    
    # save the number of classes:
    num_classes = len(os.listdir(train_dir))
    
    opts = {'SGD (1e-2)'          : optimizers.SGD(lr=0.01, momentum=0.0, clipvalue=5.), 
        'SGD (1e-2) momentum 0.5' : optimizers.SGD(lr=0.01, momentum=0.5, clipvalue=5.), 
        'SGD (1e-2) momentum 0.9' : optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=5.), 
        'SGD (1e-3)'              : optimizers.SGD(lr=0.001, momentum=0.0, clipvalue=5.), 
        'SGD (1e-3) momentum 0.5' : optimizers.SGD(lr=0.001, momentum=0.5, clipvalue=5.), 
        'SGD (1e-3) momentum 0.9' : optimizers.SGD(lr=0.001, momentum=0.9, clipvalue=5.), 
        'RMSprop (1e-3) '         : optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        'Adam (1e-2)'             : optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        'Adamax (2e-3)'           : optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        }
    opt = opts[opt]
    
    if model_filename:
        # user provided a filename of a model saved with weights
        tag = 'fine_tune'
        # load the model:
        model = load_model(os.path.join(model_dir, model_filename))
    else:
        print('starting new model with random weights')

        tag = 'randomly_initialized_weights'
        # we start a model from scratch
        base_model = load_part_model(model_name, depth_level, None)
   
        top_model = Sequential()
        top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:4]))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(Dropout(rate=0.5))  
        # the last layer depends on the number of classes
        top_model.add(Dense(num_classes, activation='softmax'))
           
        # set the entire model:
        # build the network
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        
        # delete top model to release memory
        del top_model

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
                                  steps_per_epoch=len(train_generator),
                                  validation_steps=len(valid_generator),
                                  epochs=epochs)
    
    # plot and save the training history:
    plot_history(history, model_name, depth_level, data_in, tag, image_dir)
        
    # predict test values accuracy
    generator = datagen.flow_from_directory(
                        test_dir,
                        target_size=model.input_shape[1:3],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)

    pred = model.predict_generator(generator, 
                                    verbose=0, 
                                    steps=len(generator)
									)

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
          plot_name=f"conf_matrix_{data_in}_{model_name}_{depth_level}_{tag}.png", 
          dpi=1000)

    # clear memory
    reset_keras()

    return acc, cfm, history


if __name__ == "__main__":
    # set the seed for repetition purposes 
    random.seed(0)
    np.random.seed(0)
    set_random_seed(0)
    
    data_dir = '../data'
    image_dir = '../images'
    model_dir = '../models'
    
    datasets_dir = [
                    os.path.join(data_dir, 'PatternNet'), 
                    os.path.join(data_dir, 'AID'),
                    os.path.join(data_dir, 'UCMerced'),
                    ]
    
    # helpers dictionary to create an easier to plot dataframe later
    depths_dict = {'shallow' : 1, 
                   'intermediate' : 2, 
                   'deep' : 3, 
                    }
    data_mod_dict = {'PatternNet'   : +0.1, 
                     'AID'          : +0.0, 
                     'UCMerced'     : -0.1, 
                    }


    
    models = ['VGG19', 'InceptionV3']
    depth_levels = [
                    #'shallow', 
                    #'intermediate', 
                    'deep'
                    ]
    opt = 'SGD (1e-3) momentum 0.9'
    epochs = 2
    
    # test different models:
    acc_dict={}
    dict_counter = 0
    weights = 'imagenet'
    for dataset in tqdm(datasets_dir):
        for model in models:
            for depth_level in depth_levels:
                print(f'{model:20} {depth_level:20} {dataset}')
                data_in   = os.path.basename(dataset)
                train_dir = os.path.join(dataset, 'images_train')
                valid_dir = os.path.join(dataset, 'images_validation')
                test_dir  = os.path.join(dataset, 'images_test')

                acc, _, _ = train_frozen(model, 
                                         depth_level, 
                                         weights, 
                                         data_in,
                                         train_dir,
                                         valid_dir, 
                                         test_dir, 
                                         image_dir, 
                                         model_dir,
                                         epochs, 
                                         opt)
                
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"    :model,
                                         "depth"   :depth_level,
                                         "depth_sw":depths_dict[depth_level] + data_mod_dict[data_in],
                                         "dataset" :data_in,
                                         "mode"    :'feature extraction',
                                         "accuracy":acc}
                dict_counter+=1
                reset_keras()
                ###############################################################
                # fine tune:
                # to mantain the same overall epochs used in training, 
                # first fine tune using half of the total epochs (save the history):
                _, _, hist = train_frozen(model, 
                                         depth_level, 
                                         weights, 
                                         data_in,
                                         train_dir,
                                         valid_dir, 
                                         test_dir, 
                                         image_dir, 
                                         model_dir,
                                         epochs//2, 
                                         opt)
                
                # now, fine tune the model for the "second half" of the 
                # epochs:
                model_filename = f"{model}_{depth_level}_{data_in}_frozen.hdf5"
                
                acc, _, h2 = fine_tune(model_filename, model, depth_level,
                                             data_in,
                                             train_dir,
                                             valid_dir, 
                                             test_dir, 
                                             image_dir, 
                                             model_dir,
                                             epochs//2, 
                                             opt)
                
                for metr in h2.history:
                    print(metr)
                    for jj in h2.history[metr]:
                        hist.history[metr].append(jj)

                # plot the history:
                plot_history(hist, model, depth_level, data_in, 'fine_tune', image_dir)
                
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"   :model,
                                         "depth"   :depth_level,
                                         "depth_sw":depths_dict[depth_level] + data_mod_dict[data_in],                                         
                                         "dataset" :data_in,
                                         "mode"    :'fine tune',
                                         "accuracy":acc}
                dict_counter+=1
                reset_keras()
                ###############################################################
                # random initialization:
                model_filename = None

                acc, _, _ = fine_tune(model_filename, model, depth_level,
                                             data_in,
                                             train_dir,
                                             valid_dir, 
                                             test_dir, 
                                             image_dir, 
                                             model_dir,
                                             epochs, 
                                             opt)
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"   :model,
                                         "depth"   :depth_level,
                                         "depth_sw":depths_dict[depth_level] + data_mod_dict[data_in],                                         
                                         "dataset" :data_in,
                                         "mode"    :'randomly initialized weights',
                                         "accuracy":acc}
                dict_counter+=1
                reset_keras()
                
                # convert accuracy to dataframe    
                df_acc = pd.DataFrame.from_dict(acc_dict, orient='index',)
                # save dataframe:
                df_acc.to_csv(os.path.join(data_dir, 'df_accuracy.csv'))
                
    # print accuracy results
    for k in acc_dict:
        print(acc_dict[k])
    
    # convert accuracy to dataframe    
    df_acc = pd.DataFrame.from_dict(acc_dict, orient='index',)
    # create a column to be used for ploting with slightly different
    #"depth" values:
    # Create the dictionary 
    depths_dict = {'shallow' : 1, 
                   'intermediate' : 2, 
                   'deep' : 3, 
                    }
    
    df_acc['depth_n'] = df_acc['depth'].map(depths_dict) 
    df_acc['depth_sw'] = df_acc['depth_n']
    
    # plot accuracy:
    plt.style.use('fivethirtyeight')
    #sns.set_style("whitegrid")
    sns.set_context("paper")
    
    g = sns.relplot(x="depth_sw", y="accuracy", hue="mode", style="dataset", 
                    col="model", 
                    #kind="swarm",
                    s = 10,  
                    legend='full', alpha=.75,
                    data=df_acc, 
                    height=3, aspect=3/3)
    
    for c in range(g.axes.shape[1]):
        g.axes[0,c].set_title(g.axes[0,c].get_title().split('model = ')[1]) 
        g.axes[0,c].set_xlabel('depth') 
        g.axes[0,c].set_xticks([val for val in depths_dict.values()])
        g.axes[0,c].set(xticklabels=[k for k in depths_dict], 
                        xlim=(min([val for val in depths_dict.values()])-.3, 
                              max([val for val in depths_dict.values()])+.3))

    plt.savefig(os.path.join(image_dir, 'accuracy_all'), 
                facecolor="white",
                dpi=1000)
    plt.close('all')
    
    # save the dataframes:
    df_acc.to_csv(os.path.join(data_dir, 
                               f'df_acc_{"_".join(df_acc.depth.unique().tolist())}.csv'))