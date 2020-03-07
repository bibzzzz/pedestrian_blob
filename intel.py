import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import math
import glob
import os

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle



def load_data(filepath='./sim_data/0.2_expl_rate_20_move_limit_4_coord_range_sim_data.csv',
              val_split=0.2, test_split=0.2, version_lookback=math.inf, version_upper_cutoff=math.inf, obs_limit=100, order_strategy='random'):

    dataframe = pd.read_csv(filepath)
    dataframe = dataframe[dataframe.decision_stage=='pre']
    intel_version = max(dataframe.intel_version) + 1
    version_lower_cutoff = intel_version - 1 - version_lookback

    dataframe = dataframe[(dataframe.intel_version>=version_lower_cutoff) & (dataframe.intel_version<=version_upper_cutoff)]


    if order_strategy == 'random':
        if obs_limit >= len(dataframe):
            sample_fraction = 1
        else:
            sample_fraction = obs_limit / len(dataframe)
            print('sampling %s percent of training data' %(sample_fraction * 100))

        dataframe = dataframe.sample(frac=sample_fraction)
    else:
        dataframe = dataframe[-obs_limit:]

    train, test = train_test_split(dataframe, test_size=test_split)
    train, val = train_test_split(train, test_size=val_split)

    return train, val, test, intel_version

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32, input_cols=['blob_x', 'blob_y'],
                  response_cols=['sim_result'], coord_range=4, classification=True):

  dataframe = dataframe.copy()
  if classification == True:
      output_data = dataframe[response_cols]
      output_array = np.array(output_data)
  else:
      output_data = dataframe[response_cols]
      output_data['label'] = output_data.sim_move_total - output_data.n_moves
      output_array = np.array(output_data.label, dtype=np.float32)
      #print(output_data.label.head(5))
      #print(output_array[:5])

  # standardize input data
  input_data = dataframe[input_cols] / coord_range

  #print(input_data.head(5))
  #print(output_data.head(5))

  #ds = tf.data.Dataset.from_tensor_slices((dict(input_data), dict(output_data)))
  ds = tf.data.Dataset.from_tensor_slices((dict(input_data), output_array))

  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def model_update(expl_rate=0.2, move_limit=20, coord_range=4, batch_size=32, n_epochs=5, test_split=0.2, val_split=0.2, order_strategy='random', response_cols=['sim_result'], input_cols=['blob_x', 'blob_y', 'target_x', 'target_y'], classification=True, version_lookback=math.inf):

    sim_data_dir = './sim_data/'
    #input_cols = ['blob_x', 'blob_y', 'target_x', 'target_y']
    #response_cols = ['sim_result']

    file_pattern = '%s_expl_rate_%s_move_limit_%s_coord_range' %(expl_rate, move_limit, coord_range)

    # load existing best intel result
    if os.path.exists('./intel/%s.best.pickle' %(file_pattern)):
        with open('./intel/%s.best.pickle' %(file_pattern), 'rb') as f:
            persist_best_score = pickle.load(f)
            #chkpt_cb.best = best
            print('existing intel found with best loss score of %s...' %(persist_best_score))
    else:
        print('no existing intel found...')
        persist_best_score = math.inf

    train, val, test, intel_version = load_data(filepath=sim_data_dir+file_pattern+'_sim_data.csv',
                                                order_strategy=order_strategy, obs_limit=math.inf,
                                                test_split=test_split, val_split=val_split,
                                                version_lookback=version_lookback)

    train_ds = df_to_dataset(train, batch_size=batch_size, input_cols=input_cols,
                             response_cols=response_cols, coord_range=coord_range, classification=classification)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size, input_cols=input_cols,
                           response_cols=response_cols, coord_range=coord_range, classification=classification)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size, input_cols=input_cols,
                            response_cols=response_cols, coord_range=coord_range, classification=classification)

    feature_columns = []
    for header in input_cols:
        feature_columns.append(feature_column.numeric_column(header))

    feature_layer = layers.DenseFeatures(feature_columns)

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('./intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model_weights.h5' %(expl_rate, move_limit, coord_range, intel_version), save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')

    #model = tf.keras.Sequential([
    #  feature_layer,
    #  layers.Dense(128, activation='relu'),
    #  layers.Dense(128, activation='relu'),
    #  layers.Dense(len(response_cols))
    #])

    model = tf.keras.Sequential([
            #tf.keras.layers.Dense(len(input_cols), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, use_bias=True),
            feature_layer,
            tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='relu'),
            tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.00001), activation='relu', use_bias=True),
            #tf.keras.layers.Dense(1, activation='sigmoid')
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
    ])

    #model.compile(tf.keras.optimizers.Adam(lr=1e-3),
    model.compile(tf.keras.optimizers.RMSprop(lr=1e-3),
              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              loss=tf.keras.losses.MeanSquaredError(),
              #metrics=['accuracy'])
              metrics=['mae', 'mse'])

    #model.fit(train_ds,
    #      validation_data=val_ds,
    #      epochs=n_epochs)

    print('commencing version %s update...' %(intel_version))
    model.fit(train_ds, validation_data=val_ds, epochs=n_epochs, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    #loss, accuracy = model.evaluate(test_ds)
    loss, mae, mse = model.evaluate(test_ds)
    #print("Accuracy", accuracy)
    print("MAE, MSE:", mae, mse)

    persist_best_score = math.inf #TODO: control always save mechanism
    if mcp_save.best <= persist_best_score:
        # store new best score
        with open('./intel/%s.best.pickle' %(file_pattern), 'wb') as f:
            pickle.dump(mcp_save.best, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save model object
        tf.keras.models.save_model(model, './intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model' %(expl_rate, move_limit, coord_range, intel_version))



if __name__ == '__main__':

    model_update(expl_rate=0.2, move_limit=math.inf, coord_range=4, batch_size=32, n_epochs=150,
                 test_split=0.2, val_split=0.2, order_strategy='random',
                 response_cols=['sim_move_total', 'n_moves'], classification=False,
                 version_lookback=0)
