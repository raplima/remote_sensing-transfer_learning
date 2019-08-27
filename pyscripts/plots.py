# -*- coding: utf-8 -*-
"""
Plotting scripts 
"""
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

def plot_history(history, model_name, depth_level, data_in, tag, image_dir, dpi=200):
    """
    plots and saves a figure with trainig and validation loss from the 
    """
    plt.style.use('fivethirtyeight')
    # plotting the metrics
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['acc'])
    ax[0].plot(range(1, len(history.history['acc']) + 1), history.history['val_acc'])
    #ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='lower right')

    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['loss'])
    ax[1].plot(range(1, len(history.history['acc']) + 1), history.history['val_loss'])
    #ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper right')

    # set up figure
    fig.set_size_inches(w=5, h=7)

    plt.savefig(os.path.join(image_dir, f"{data_in}_{model_name}_{depth_level}_{tag}.png"), dpi=dpi)
    plt.clf()
    plt.close()

def cf_matrix(y_true, y_pred, image_dir, plot_name='confusion_matrix', dpi=200):
    """
    Plot the confusion matrix
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :param image_dir: the image output folder
    :param plot_name: name for the figure to be saved
    :param dpi: dots per inch
    :return: saves image file
    """

    # get the unique labels
    labels = sorted(set(np.concatenate([y_true, y_pred])))

    # compute confusion matrix with sklearn
    cfm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cfm, index=[i.replace('_', ' ') for i in labels],
                         columns=[i.replace('_', ' ') for i in labels])
    df_cm_annot = df_cm.copy()
    df_cm_annot[df_cm_annot<=0.0]=''
    
    plt.figure(figsize=(10, 9), )
    sns.set(font_scale=1.0)  # for label size
    sns.heatmap(df_cm, annot=df_cm_annot, fmt='', 
                cmap="Greens",
                linewidths=.5,
                cbar_kws={"ticks": [0, np.int(np.amax(cfm) / 2), np.int(np.amax(cfm))]},
                annot_kws={"size": 8.5})  # font size

    plt.text(-1, len(labels) + 4, f"Accuracy: {accuracy_score(y_true, y_pred):.2f}",
             size=10,
             ha="right", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0., 0., 0.),
                       fc=(1., 1., 1.)
                       )
             )
             
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"{plot_name}.png"), dpi=dpi)
    plt.clf()
    plt.close()
