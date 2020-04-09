import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
import math
import uuid
import six.moves.cPickle as pickle
import os
import csv
import glob
from multiprocessing.dummy import Pool as ThreadPool
from decision import calc_decision, execute_decision
from intel import model_update


class BlobSimulation():

    '''
        class object that simulates a complete pedestrian blob attempt at reaching target coords and records the results
    '''

    def __init__(self, batchID, coord_range, move_limit, data_dir, use_intel=0, expl_rate=0,
                 move_type='manhattan', traffic='none', print_freq=200, intel_move_limit=math.inf,
                 intel_version=None):

        #self.batchID = batchID
        self.simID = '%s-%s' %(str(uuid.uuid4())[0:8], batchID)
        self.data_dir = data_dir
        self.print_freq = print_freq

        self.use_intel = use_intel
        self.intel_move_limit = intel_move_limit
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
            if intel_version==None:
                # TODO: count number of matching files in intel dir
                intel_filelist = glob.glob('intel/%s_expl_rate_%s_move_limit_%s_coord_range_*model' %(self.expl_rate, self.move_limit, self.coord_range))
                #print(intel_filelist)
                self.intel_version = len(intel_filelist)
            else:
                self.intel_version = intel_version
            if self.intel_version == 0:
                self.use_intel = 0
                self.model = 'none'
            else:
                self.model = tf.keras.models.load_model('./intel/%s_expl_rate_%s_move_limit_%s_coord_range_%s_version_model' %(self.expl_rate, self.move_limit, self.coord_range, self.intel_version))

    def simulate(self):

        self.pre_decisionList = []
        self.post_decisionList = []
        self.decision_history = []

        print('initializing simID %s: starting at (%s, %s) with target (%s, %s) and %s moves, using intel version %s' %(self.simID, self.x, self.y, self.target_x, self.target_y, self.move_limit, self.intel_version))

        while (self.n_moves < self.move_limit) and (self.sim_result == 0):

            #print(self.decision_history[-10:])
            #print(set(self.decision_history[-10:]))

            if (self.use_intel == 1) and (self.n_moves > self.intel_move_limit):
            #if (len(self.decision_history) >= self.intel_move_limit) and (len(set(self.decision_history[-self.intel_move_limit:])) == 1):
                self.use_intel = 0
                print('simID %s shutting off intel...' %(self.simID))

            self.pre_decisionList.append([self.simID, self.intel_version, 'pre', self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            #print('calculating move %s...' %(self.n_moves))
            self.x_open = np.random.randint(0,2,1)[0]

            (self.x_move, self.y_move) = calc_decision(x=self.x, y=self.y, target_x=self.target_x,
                                             target_y=self.target_y, x_open=self.x_open,
                                             coord_range=self.coord_range, move_type=self.move_type,
                                             traffic=self.traffic, use_intel=self.use_intel,
                                             intel_version=self.intel_version,
                                             expl_rate=self.expl_rate, move_limit=self.move_limit,
                                             model=self.model)

            self.decision_history.append((self.x_move, self.y_move))

            #print('executing move %s... x:%s, y:%s' %(self.n_moves, x_move, y_move))
            self.x, self.y = execute_decision(x=self.x, y=self.y, move_x=self.x_move, move_y=self.y_move)

            self.n_moves += 1
            self.post_decisionList.append([self.simID, self.intel_version, 'post',  self.n_moves, self.x, self.y, self.target_x, self.target_y, self.coord_range])

            if self.n_moves % self.print_freq == 0:
                print('simID %s: after %s moves blob is at %s, %s - target at %s, %s' %(self.simID, self.n_moves, self.x, self.y, self.target_x, self.target_y))

            if (self.x == self.target_x and self.y == self.target_y):
                print('%s TARGET REACHED in %s MOVES!' %(self.simID, self.n_moves))
                self.sim_result += 1

            if self.n_moves == self.move_limit:
                print('%s MOVE LIMIT EXCEEDED!' %(self.simID))


if __name__ == '__main__':

    # processing settings
    n_simulations = 40
    n_workers = 4

    target_n_obs = 30000

    # sim settings
    expl_rate = 0.1
    #move_limit = 20
    move_limit = math.inf
    coord_range = 5
    intel_version = None

    # define simulation function (for parallelerization)
    def simulate(simulation):
        simulation.simulate()

    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/sim_data/'
    data_filepath = os.path.dirname(os.path.realpath(__file__)) + '/sim_data/%s_expl_rate_%s_move_limit_%s_coord_range_sim_data.csv' %(expl_rate, move_limit, coord_range)

    # loop simulate
    version_n_obs = 0
    sim_counter = 0
    sim_multiple = n_simulations
    while version_n_obs <= target_n_obs:
        # create a fleet of simulations, and store them in a list
        sims = [BlobSimulation(batchID=x, coord_range=coord_range, move_limit=move_limit,
                               data_dir=data_dir, use_intel=1, expl_rate=expl_rate,
                               move_type='manhattan', traffic='none', intel_move_limit = 100,
                               intel_version=intel_version) for x in range(0, n_simulations)]

        # make the Pool of workers
        pool = ThreadPool(n_workers)

        # threaded simulation
        results = pool.map(simulate, sims)

        #close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # store results
        for sim in sims:

            # if output file doesnt exist create it
            if not os.path.exists(data_filepath):
                with open(data_filepath,'w') as f:
                    sim_output = csv.writer(f)
                    rowEntry = ['simID', 'intel_version', 'decision_stage', 'n_moves', 'blob_x', 'blob_y', 'target_x', 'target_y', 'coord_range', 'sim_move_limit', 'sim_move_total', 'sim_result']
                    sim_output.writerow(rowEntry)
            # open files for appending
            with open(data_filepath,'a') as f:
                sim_output = csv.writer(f)
                for entry in sim.pre_decisionList:
                    rowEntry = entry
                    rowEntry.extend([sim.move_limit, sim.n_moves, sim.sim_result])
                    sim_output.writerow(rowEntry)
                for entry in sim.post_decisionList:
                    rowEntry = entry
                    rowEntry.extend([sim.move_limit, sim.n_moves, sim.sim_result])
                    sim_output.writerow(rowEntry)

                f.close()

        # extra simulations
        sim_data = pd.read_csv(data_filepath)
        sim_data = sim_data[sim_data.decision_stage=='pre']
        current_version = max(sim_data.intel_version)
        version_n_obs = len(sim_data[sim_data.intel_version==current_version])
        n_obs_gap = target_n_obs - version_n_obs

        sim_counter += n_simulations

        n_simulations = min(np.int(n_obs_gap / sim_counter), sim_multiple)
        print('running %s extra simulations for intel version %s' %(n_simulations, current_version))

    # train new model on new experience
    print('commencing model update procecedure...')
    model_update(expl_rate=expl_rate, move_limit=move_limit, coord_range=coord_range, batch_size=32,
             n_epochs=500, test_split=0.2, val_split=0.2, order_strategy='random',
             response_cols=['sim_move_total', 'n_moves'], classification=False,
             version_lookback=math.inf)
