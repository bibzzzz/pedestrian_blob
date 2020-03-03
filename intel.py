import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import math
import glob

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def load_data(filepath='./sim_data/0.2_expl_rate_20_move_limit_4_coord_range_sim_data.csv', val_split=0.2, test_split=0.2, version_lower_cutoff=0, version_upper_cutoff=1, obs_limit=100, order_strategy='random'):

    dataframe = pd.read_csv(filepath)
    dataframe = dataframe[(dataframe.intel_version>=version_lower_cutoff) & (dataframe.intel_version<=version_upper_cutoff)]

    if order_strategy == 'random':
        if obs_limit >= len(dataframe):
            sample_fraction = 1
        else:
            sample_fraction = obs_limit / len(dataframe)

        dataframe = dataframe.sample(frac=sample_fraction)
    else:
        dataframe = dataframe[-obs_limit:]

    print(dataframe.shape)
    print(test_split)

    train, test = train_test_split(dataframe, test_size=test_split)
    train, val = train_test_split(train, test_size=val_split)

    print(train.shape)
    print(val.shape)
    print(test.shape)

    return train, val, test

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32, input_cols=['blob_x', 'blob_y'], response_cols=['sim_result']):

  dataframe = dataframe.copy()
  output_data = dataframe[response_cols]
  input_data = dataframe[input_cols]

  print(input_data.head(5), output_data.head(5))
  ds = tf.data.Dataset.from_tensor_slices((dict(input_data), dict(output_data)))

  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

if __name__ == '__main__':

    sim_data_dir = './sim_data/'
    input_cols = ['blob_x', 'blob_y', 'target_x', 'target_y']

    expl_rate = 0.2
    move_limit = 20
    #move_limit = math.inf
    coord_range = 4

    file_pattern = '%s_expl_rate_%s_move_limit_%s_coord_range_sim_data.csv' %(expl_rate, move_limit, coord_range)
    intel_filelist = glob.glob('/intel/%s_expl_rate_%s_move_limit_%s_coord_range_*.json' %(expl_rate, move_limit, coord_range))
    intel_version = len(intel_filelist)

    response_cols = ['sim_result']
    batch_size = 32
    n_epochs = 5

    train, val, test = load_data(filepath=sim_data_dir+file_pattern, order_strategy='random',
                                 obs_limit=math.inf, test_split=0.2, val_split=0.2)
    train_ds = df_to_dataset(train, batch_size=batch_size, input_cols=input_cols, response_cols=response_cols)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size, input_cols=input_cols, response_cols=response_cols)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size, input_cols=input_cols, response_cols=response_cols)

    feature_columns = []
    for header in input_cols:
        feature_columns.append(feature_column.numeric_column(header))

    feature_layer = layers.DenseFeatures(feature_columns)

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('/intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model_weights.h5' %(expl_rate, move_limit, coord_range, intel_version), save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')

    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(len(response_cols))
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #model.fit(train_ds,
    #      validation_data=val_ds,
    #      epochs=n_epochs)

    model.fit(train_ds, validation_data=val_ds, epochs=n_epochs, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(
        os.path.split(__file__)[0],
        '/intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model.json' %(expl_rate, move_limit, coord_range, intel_version)
        ), "w") as json_file:
            json_file.write(model_json)

