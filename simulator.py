import numpy as np
import pandas as pd
import itertools
import math
import uuid
import six.moves.cPickle as pickle
import os
from datetime import datetime, date, timedelta
from simulator_input import get_sim_data
from mlp import predict_mlp_given_x


class BlobSimulation():

    '''
        class object that simulates a complete pedestrian blob attempt at reaching target coords and records the results
    '''

    def __init__(self, x_start, y_start, x_target, y_target, coord_range, move_limit,
                 data_dir, use_intel=0, move_types='manhattan', traffic='none'):

        self.simID = str(uuid.uuid4())
        self.data_dir = data_dir

        self.use_intel = use_intel

        self.x = x_start
        self.y = y_start
        self.x_target = x_target
        self.y_target = y_target

        self.move_types = move_types
        self.traffic = traffic
        self.move_limit = move_limit

        # initialize sim outcome vars
        self.n_moves = 0
        self.sim_result = 0

        # determine intel version
        if self.use_intel == 0:
            self.intel_version = 0
        else:
            # TODO: count number of matching files in intel dir
            self.intel_version = 1

    def simulate(self):

        pre_decisionList = []
        post_decisionList = []
        while (self.n_moves < self.move_limit) or (self.x == self.target_x and self.y == self.target_y):

            pre_decisionList = pre_decisionList.append([self.simID, self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            print('executing move %s...' %(self.n_moves))
            x_open = np.random.randint(0,2,1)[0]

            (x_move, y_move) = calc_decision(x=self.x, y=self.y, target_x=self.target_x,
                                             target_y=self.target_y, x_open=x_open,
                          coord_range=self.coord_range, move_type=self.move_type,
                          traffic=self.traffic, use_intel=self.use_intel,
                          intel_version=self.intel_version)

            print('executing move %s...' %(self.n_moves))
            self.x, self.y = execute_decision(x=self.x, y=self.y, move_x=x_move, move_y=y_move)

            self.n_moves += 1
            post_decisionList = post_decisionList.append([self.simID, self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            if (self.x == self.target_x and self.y == self.target_y):
                self.sim_result += 1


