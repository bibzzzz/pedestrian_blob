import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import math

from intel import df_to_dataset
from keras.models import model_from_json, load_model
import keras


def execute_decision(x, y, move_x, move_y):
    x_new = x + move_x
    y_new = y + move_y

    return x_new, y_new

def calc_decision(x, y, target_x, target_y, x_open, coord_range, move_type='manhattan',
                  traffic='off', use_intel=0, intel_version=0, expl_rate=0, move_limit=10,
                  model='none'):

    if x_open==1:
        #x_moves = [-1, 0, 1]
        x_moves = [-1, 1]
        y_moves = [0]
    else:
        x_moves = [0]
        #y_moves = [-1, 0, 1]
        y_moves = [-1, 1]

    x_moves = [i for i in x_moves if abs(i + x) <= coord_range]
    y_moves = [i for i in y_moves if abs(i + y) <= coord_range]
    x_proj = [i + x for i in x_moves]
    y_proj = [i + y for i in y_moves]
    #print(x_moves)
    #print(y_moves)

    xy_moves = list(itertools.product(x_moves, y_moves))
    xy_proj = list(itertools.product(x_proj, y_proj))
    #print(xy_moves)
    #print(xy_proj)

    if use_intel==0:
        #pick random move from move list
        proj_vec = np.ones(len(xy_moves))
        #decision_index = np.random.randint(0, len(xy_moves), 1)[0]
    else:
        proj_dataframe = pd.DataFrame(xy_proj)
        proj_dataframe.columns = ['blob_x', 'blob_y']
        proj_dataframe['target_x'] = target_x
        proj_dataframe['target_y'] = target_y

        proj_dataframe = proj_dataframe / coord_range
        #print(proj_dataframe.head(5))
       # proj_input = df_to_dataset(proj_dataframe)
        #proj_input = tf.data.Dataset.from_tensor_slices(dict(proj_dataframe))
        #proj_input = proj_input.batch(len(xy_proj))

        proj_input = np.array(proj_dataframe)
        #print(proj_input)

        #proj_input = np.array(proj_dataframe)
        #for features_tensor in proj_input:
        #    print(f'features:{features_tensor}')

        #model = tf.keras.models.load_model('./intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model' %(expl_rate, move_limit, coord_range, intel_version))

        #proj_vec = model.predict(proj_input)
        #proj_vec = np.array(model.predict_on_batch(proj_input), dtype=np.float32).flatten()
        proj_vec = np.array(model.predict(proj_input), dtype=np.float32).flatten()
        #print(proj_vec)

    rand_adj_vec = np.random.uniform(0, 1, len(xy_moves))
    #print(rand_adj_vec)
    result_vec = (1-expl_rate)*proj_vec + (expl_rate)*proj_vec*rand_adj_vec
    #print(result_vec)
    #decision_index = np.argmax(result_vec)
    decision_index = np.argmin(result_vec)
    #print(decision_index)
    (x_move, y_move) = xy_moves[decision_index]
    #print((x_move, y_move))

    return (x_move, y_move)


#if __name__ == '__main__':
#
#    xy_move = calc_decision(x=0, y=0, target_x=2, target_y=2, x_open=0, coord_range=1, move_type='manhattan', traffic='off', use_intel=0, intel_version=0)
#
#    print(xy_move[0])
#    print(xy_move[1])
