# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:42:21 2019

@author: pire3708
Use the smaller UCMerced to test different optmizers and 
"""
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from keras import optimizers
from tensorflow import set_random_seed

from matplotlib import pyplot as plt
import seaborn as sns

from fit_models_experiments import fine_tune
from helper import reset_keras


if __name__ == "__main__":

    # set the seed for repetition purposes 
    random.seed(0)
    np.random.seed(0)
    set_random_seed(0)


    data_dir = '../data'
    image_dir = '../images'
    model_dir = '../models'
    
    dataset = os.path.join(data_dir, 'UCMerced')
    
    models = [
            'VGG19', 
            'InceptionV3'
              ]
    depth_levels = ['shallow', 
                    'intermediate', 
                    'deep'
                    ]
    opts = {'SGD (1e-2)'              : optimizers.SGD(lr=0.01, momentum=0.0, clipvalue=5.), 
            #'SGD (1e-2) momentum 0.5' : optimizers.SGD(lr=0.01, momentum=0.5, clipvalue=5.), 
            'SGD (1e-2) momentum 0.9' : optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=5.), 
            'SGD (1e-3)'              : optimizers.SGD(lr=0.001, momentum=0.0, clipvalue=5.), 
            #'SGD (1e-3) momentum 0.5' : optimizers.SGD(lr=0.001, momentum=0.5, clipvalue=5.), 
            'SGD (1e-3) momentum 0.9' : optimizers.SGD(lr=0.001, momentum=0.9, clipvalue=5.), 
            #'RMSprop (1e-3) '         : optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
            'Adam (1e-2)'             : optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
            'Adamax (2e-3)'           : optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            }
    epochs_random_init = 100
    
    # test different models:
    acc_dict={}
    hist_dict={}
    dict_counter = 0
    weights = 'imagenet'
    for model in models:
        for depth_level in tqdm(depth_levels):
            print(f'\n\n\n{model:20} {depth_level:20} {dataset}\n\n\n')
            data_in   = os.path.basename(dataset)
            train_dir = os.path.join(dataset, 'images_train')
            valid_dir = os.path.join(dataset, 'images_validation')
            test_dir  = os.path.join(dataset, 'images_test')
            ###############################################################
            # random initialization:
            model_filename = None
            for opt in opts:
                
                acc, _, history = fine_tune(model_filename, model, depth_level,
                                             data_in,
                                             train_dir,
                                             valid_dir, 
                                             test_dir, 
                                             image_dir, 
                                             model_dir,
                                             epochs_random_init, 
                                             opt)
                print(f'{acc:.2f}')
                acc_dict[dict_counter]={"model"   :model,
                                        "depth"   :depth_level,
                                        "opt"     :opt,
                                        "opt_mode":opt.split(' ')[0],
                                        "mode"    :'randomly initialized weights',
                                        "accuracy":acc}
                hist_dict[f'{model}_{depth_level}_{opt}']={'train_acc' : history.history['acc'], 
                                                   'val_acc' : history.history['val_acc'], 
                                                   'train_loss': history.history['loss'], 
                                                   'val_loss': history.history['val_loss'],
                                                   'epoch' : np.arange(1, len(history.history['acc'])+1),
                                                   'model' : model, 
                                                   'depth': depth_level, 
                                                   'optmizer' : opt}
                         
                print(f'{model}_{depth_level}_{opt} okay')
                dict_counter+=1
                reset_keras()
                
    # print accuracy results
    for k in acc_dict:
        print(acc_dict[k])
    
    # convert accuracy to dataframe    
    df_acc = pd.DataFrame.from_dict(acc_dict, orient='index',)
    # change column names to facilitate legends:
    df_acc = df_acc.rename(columns={"opt": "optmizer parameters", 
                           "opt_mode": "optmizer"}
                    )
    
    # plot accuracy:
    plt.style.use('seaborn-paper')
    plt.figure(figsize=(4, 4), )
    g = sns.catplot(x="depth", y="accuracy", hue="optmizer parameters",
                    col="model",
                    s = 10,  kind="swarm",
                    legend='full', alpha=.75,
                    data=df_acc)
    """
    g=sns.relplot(x="depth", y="accuracy", hue="optmizer parameters", style="optmizer", 
                    col="model",
                    s = 200,
                    legend='full', alpha=.75,
                    data=df_acc)
    """
    for c in range(g.axes.shape[1]):
        g.axes[0,c].set_title(g.axes[0,c].get_title().split(' = ')[1]) 
    plt.savefig(os.path.join(image_dir, 'accuracy_opt.png'), dpi=1000)
    plt.close('all')
    
    # convert history dataframes:
    hist_df = pd.DataFrame()
    for h in hist_dict:
        hist_df = hist_df.append(pd.DataFrame.from_dict(hist_dict[h], orient='columns'))
    
    melt_df=pd.melt(hist_df, id_vars = ['model', 'depth', 'optmizer', 'epoch'])
    melt_df[['set', 'metric']] = melt_df['variable'].str.split('_', expand=True)
    melt_df['set'].replace('val', 'validation', inplace=True)
    melt_df['metric'].replace('acc', 'accuracy', inplace=True)
    melt_df = melt_df.rename(columns={"optmizer": "optmizer parameters"}
                )
    melt_df = melt_df[melt_df['metric']=='accuracy'].copy()
    # plot losses:
    plt.style.use('fivethirtyeight')
    #sns.set_style("whitegrid")
    sns.set_context("paper")

    g = sns.relplot(x="epoch", y="value", hue="optmizer parameters", style="set", 
                 col='model', row="depth", kind="line", ci=None, 
                 legend='brief', alpha=.75, 
                 lw=0.75, 
                 dashes=[(),(3, 0.75)], 
                 data=melt_df, 
                 facet_kws=dict(sharey=True),
                 height=3, aspect=3/3
                 )
    for c in range(g.axes.shape[1]):
        g.axes[0,c].set_title(g.axes[0,c].get_title().split('model = ')[1]) 
        for r in range(1, g.axes.shape[0]):
            g.axes[r,c].set_title('') 
    for pos, met in enumerate(melt_df['depth'].unique()):
        g.axes[pos,0].set_ylabel(f'accuracy - {met}')
    g.savefig(os.path.join(image_dir, 'loss.png'), dpi=1000)
    plt.close('all')

    # save the dataframes:
    df_acc.to_csv(os.path.join(data_dir, 'opt_acc.csv'))
    hist_df.to_csv(os.path.join(data_dir, 'hist_df.csv'))