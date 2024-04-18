# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:33:46 2024

@author: LoïcMARCADET
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

from scores_utils import *

def kendall_loss(y_true, y_pred):
    """
    Kendall Tau loss function.
    y_true and y_pred should have the same shape.
    """
    print(y_true)
    print(y_pred)
    
    # Flatten the inputs
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    print(y_true)
    print(y_pred)

    # Calculate Kendall Tau correlation
    kendall_tau, _ = tf.py_function(kendalltau, [y_true_flat, y_pred_flat], [tf.float32, tf.float32])

    loss = 1.0 - kendall_tau

    return loss


#%% Retrieve dataframes

MS, SU, SP, RE = get_scores()
MSH, SUH, SPH, REH = homogen_df(MS, SU, SP, RE)
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)
scores = get_score_df()
scores_valid, valid_indices = keep_valid()
std_scores = standardise_df()

# All the agencies
# dict_agencies = {'MS':  MSS['Score'], 'SP': SPS['Score'],
#    'RE': RES['Score'], 'SU': SUS['Score']}

# All the agencies except MSCI
df = scores_valid.drop(columns= ['MS'])

#%% Split datasets into combinations

pred_SU, pred_SP, pred_RE = df.copy(), df.copy(), df.copy()
labels_SU, labels_SP, labels_RE = pred_SU.pop('SU'), pred_SP.pop('SP'), pred_RE.pop('RE')

norm_SU, norm_SP, norm_RE = tf.keras.layers.Normalization(axis=-1), tf.keras.layers.Normalization(axis=-1), tf.keras.layers.Normalization(axis=-1)

norm_SU.adapt(np.array(pred_SU))
norm_SP.adapt(np.array(pred_SP))
norm_RE.adapt(np.array(pred_RE))

#%% Neural Network

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mae',
                optimizer=tf.keras.optimizers.Adam(0.001))
  #model.compile(loss=kendall_loss, optimizer=tf.keras.optimizers.Adam(0.001))
  return model

model_SU, model_SP, model_RE = build_and_compile_model(norm_SU), build_and_compile_model(norm_SP), build_and_compile_model(norm_RE)

#model_SU.summary()

#%% Train

history_SU = model_SU.fit(pred_SU, df['SU'], validation_split=0.2, verbose=0, epochs=100)
history_SP = model_SP.fit(pred_SP, df['SP'], validation_split=0.2, verbose=0, epochs=100)
history_RE = model_RE.fit(pred_RE, df['RE'], validation_split=0.2, verbose=0, epochs=100)

#%% Losses

fig, ax = plt.subplots(1,3, constrained_layout=True)

def plot_loss(ax, history, agency):
  ax.plot(history.history['loss'], label='loss')
  ax.plot(history.history['val_loss'], label='val_loss')
  ax.set_ylim([0, 10])
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Error')
  ax.grid(True)
  ax.set_title('Training losses for ' + agency)
  ax.legend()

plot_loss(ax[0], history_SU, 'Sustainalytics')

plot_loss(ax[1], history_SP, 'S&P')

plot_loss(ax[2], history_RE, 'Refinitiv')
