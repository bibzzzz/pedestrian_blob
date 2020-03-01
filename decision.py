import numpy as np
import pandas as pd

def execute_decision(x, y, move_x, move_y):
    x_new = x + move_x
    y_new = y + move_y

    return x_new, y_new

def calc_decision(x, y, target_x, target_y, x_open, y_open, coord_range, move_type='manhattan', traffic='off', use_intel=0):

    if x_open==1:
        x_moves = [-1, 0, 1]
        y_moves [0]
    else:
        x_moves [0]
        y_moves = [-1, 0, 1]

    x_proj = x_moves + x
    y_proj = y_moves + y

    print(x_moves)
    x_moves = x_moves[abs(x_proj) > coord_range]
    y_moves = y_moves[abs(y_proj) > coord_range]
    print(x_moves)

    xy_moves = list(itertools.combinations(x_moves, y_moves))
    print(xy_moves)

    if use_intel==0:
        #pick random move from move list
        decision_index = np.random.randint(0, len(xy_moves), 1)[0]
        (x_move, y_move) = xy_moves[decision_index]
    else:
        predict_outcome = predict(sdjlaksldja)

    print(xy_moves)

    return (x_move, y_move)


if __name__ == '__main__':

    xy_move = calc_decision(x=0, y=0, target_x=2, target_y=2, coord_range=1, move_type='manhattan', traffic='off', use_intel=0)

    print(xy_move[0])
    print(xy_move[1])
