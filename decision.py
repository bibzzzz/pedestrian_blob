import numpy as np
import pandas as pd
import itertools
import math

def execute_decision(x, y, move_x, move_y):
    x_new = x + move_x
    y_new = y + move_y

    return x_new, y_new

def calc_decision(x, y, target_x, target_y, x_open, coord_range, move_type='manhattan',
                  traffic='off', use_intel=0, intel_version=0, expl_rate=0):

    if x_open==1:
        x_moves = [-1, 0, 1]
        y_moves = [0]
    else:
        x_moves = [0]
        y_moves = [-1, 0, 1]

    x_moves = [i for i in x_moves if abs(i + x) <= coord_range]
    y_moves = [i for i in y_moves if abs(i + y) <= coord_range]
    #print(x_moves)
    #print(y_moves)

    xy_moves = list(itertools.product(x_moves, y_moves))
    #print(xy_moves)

    if use_intel==0:
        #pick random move from move list
        proj_vec = np.zeros(len(xy_moves))
        #decision_index = np.random.randint(0, len(xy_moves), 1)[0]
    else:
        proj_vec = predict(sdjlaksldja)

    rand_adj_vec = np.random.uniform(0, 1, len(xy_moves))
    #print(rand_adj_vec)
    result_vec = proj_vec + (expl_rate)*rand_adj_vec
    #print(result_vec)
    decision_index = np.argmax(result_vec)
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
