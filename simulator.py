import numpy as np
import pandas as pd
import itertools
import math
import uuid
import six.moves.cPickle as pickle
import os
import csv
from multiprocessing.dummy import Pool as ThreadPool
from decision import calc_decision, execute_decision


class BlobSimulation():

    '''
        class object that simulates a complete pedestrian blob attempt at reaching target coords and records the results
    '''

    def __init__(self, coord_range, move_limit, data_dir, use_intel=0, expl_rate=0, move_type='manhattan', traffic='none'):

        self.simID = str(uuid.uuid4())
        self.data_dir = data_dir

        self.use_intel = use_intel
        self.expl_rate = expl_rate

        self.x = np.random.randint(-coord_range, coord_range, 1)[0]
        self.y = np.random.randint(-coord_range, coord_range, 1)[0]
        self.target_x = np.random.randint(-coord_range, coord_range, 1)[0]
        self.target_y = np.random.randint(-coord_range, coord_range, 1)[0]

        self.coord_range = coord_range

        self.move_type = move_type
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

        self.pre_decisionList = []
        self.post_decisionList = []

        print('initializing simID %s: starting at (%s, %s) with target (%s, %s) and %s moves' %(self.simID, self.x, self.y, self.target_x, self.target_y, self.move_limit))

        while (self.n_moves < self.move_limit) and (self.sim_result == 0):

            self.pre_decisionList.append([self.simID, 'pre', self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            #print('calculating move %s...' %(self.n_moves))
            x_open = np.random.randint(0,2,1)[0]

            (x_move, y_move) = calc_decision(x=self.x, y=self.y, target_x=self.target_x, target_y=self.target_y, x_open=x_open, coord_range=self.coord_range, move_type=self.move_type, traffic=self.traffic, use_intel=self.use_intel, intel_version=self.intel_version, expl_rate=self.expl_rate)

            #print('executing move %s... x:%s, y:%s' %(self.n_moves, x_move, y_move))
            self.x, self.y = execute_decision(x=self.x, y=self.y, move_x=x_move, move_y=y_move)

            self.n_moves += 1
            self.post_decisionList.append([self.simID, 'post',  self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            #print(self.x, self.y)

            if (self.x == self.target_x and self.y == self.target_y):
                print('%s TARGET REACHED in %s MOVES!' %(self.simID, self.n_moves))
                self.sim_result += 1

            if self.n_moves == self.move_limit:
                print('%s MOVE LIMIT EXCEEDED!' %(self.simID))


if __name__ == '__main__':

    # processing settings
    n_simulations = 10
    n_workers = 4

    # sim settings
    expl_rate = 0.2
    move_limit = math.inf
    coord_range = 8

    # define simulation function (for parallelerization)
    def simulate(simulation):
        simulation.simulate()

    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/sim_data/'
    data_filepath = os.path.dirname(os.path.realpath(__file__)) + '/sim_data/%s_expl_rate_%s_move_limit_%s_coord_range' %(expl_rate, move_limit, coord_range)

    # create a fleet of simulations, and store them in a list
    sims = [BlobSimulation(coord_range=coord_range, move_limit=move_limit, data_dir=data_dir, use_intel=0, expl_rate=expl_rate, move_type='manhattan', traffic='none') for x in range(0,n_simulations)]

    # make the Pool of workers
    pool = ThreadPool(n_workers)

    # threaded simulation
    results = pool.map(simulate, sims)

    #close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # store results
    # specify filepaths

    for sim in sims:

        # if output file doesnt exist create it
        if not os.path.exists(data_filepath):
            with open(data_filepath,'w') as f:
                sim_output = csv.writer(f)
                rowEntry = ['simID', 'decision_stage', 'n_moves', 'blob_x', 'blob_y', 'target_x', 'target_y', 'coord_range', 'sim_move_limit', 'sim_move_total', 'sim_result']
                sim_output.writerow(rowEntry)
        # open files for appending
        with open(data_filepath,'a') as f:
            sim_output = csv.writer(f)
            for entry in sim.pre_decisionList:
                rowEntry = entry
                rowEntry.extend([sim.move_limit, sim.n_moves, sim.sim_result])
                sim_output.writerow(rowEntry)
            f.close()


