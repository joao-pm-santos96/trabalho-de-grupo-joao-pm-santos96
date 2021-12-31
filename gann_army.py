#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
from logging import log
import os
import time
import pygad
from pygad import nn
import pickle
import sys
import json
import numpy as np
from itertools import combinations
from js_logger import logger
import select
import argparse

from numpy.core.numeric import moveaxis
from actions import *
from utils import *
from utils import _PRINT

DEBUG_FILE = "client_debug.txt"
"""
METADATA
"""
__author__ = 'Joao Santos'
__copyright__ = 'Copyright December2021'
__credits__ = ['Joao Santos']
__version__ = '1.0.0'
__maintainer__ = 'Joao Santos'
__email__ = 'joao.pm.santos96@gmail.com'
__status__ = 'Production'
# __license__ = 'GPL'

"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class Environment:
    def __init__(self, difficulty, base_cost, base_prod, neural_net=None):
        self.difficulty = difficulty
        self.resources = 0
        self.building_level = 0
        self.base_cost = base_cost
        self.base_prod = base_prod
        self.board = [[None]*WIDTH for h in range(HEIGHT)]

        """
        JS STUFF
        """
        self.neural_net = neural_net
        # self.nn_actions = np.array([[0,0], 
        #                             [1,0], 
        #                             [-1,0], 
        #                             [0,1], 
        #                             [0,-1],
        #                             [1/2,0], 
        #                             [-1/2,0], 
        #                             [0,1/2], 
        #                             [0,-1/2]])

        self.motions={'U': [0,-1], 
                    'D': [0,1],
                    'L': [-1,0],
                    'R':[1,0]}


        base_moves = [None, 'U', 'D', 'L', 'R']
        self.cenas = base_moves
        self.cenas.extend(list(combinations(base_moves,2)))
        self.cenas.append('upgrade')
        self.cenas.append('recruit_melee')
        self.cenas.append('recruit_ranged')
        # TODO outputs for position and amount? same turn recruit both types?

        playActions([])

    @property
    def upgrade_cost(self):
        return int(self.base_cost*(1.4**self.building_level))

    @property
    def production(self):
        return int(self.base_prod*(1.2**self.building_level))

    def readEnvironment(self):
        # state = input()
        i, o, e = select.select( [sys.stdin], [], [], 0.500 )

        if i:
            state = sys.stdin.readline().strip()
        else:
            state = 'ERROR'
            logger.error('Input read error (timeout)')
        
        if state in ["END", "ERROR"]:
            return state
        level, resources, board = state.split()
        level = int(level)
        resources = int(resources)
        # debug(f"Building Level: {level}, Current resources: {resources}")
        self.building_level = level
        self.resources = resources

        self.board = np.swapaxes(np.array(json.loads(board)),0,1)

    def play(self): # agent move, call playActions only ONCE

        actions = []
        # print("Current production per turn is:", self.production)
        # print("Current building cost is:", self.upgrade_cost)

        # SOLDIERS DECISION MAKING

        soldiers = self.board[:,:,0]
        troops = np.argwhere((soldiers==ALLIED_SOLDIER_RANGED) | (soldiers==ALLIED_SOLDIER_MELEE))

        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        enemies = [tuple(x) for x in enemies]

        soldiers_data = []
        for x,y in troops:
            enemy = Environment.findEnemy((x,y), enemies)

            data = [self.difficulty,
                self.board[x,y,0], 
                self.board[x,y,1], 
                self.resources,
                self.upgrade_cost,
                int((self.board[x,y-1,0] if y > 0 else WALL) or 0),
                int((self.board[x,y+1,0] if y < HEIGHT-1 else WALL) or 0),
                int((self.board[x+1,y,0] if x < WIDTH-1 else WALL) or 0),
                int((self.board[x-1,y,0] if x > 0 else WALL) or 0),
                x,
                y,
                int(enemy[0] if enemy is not None else -1),
                int(enemy[1] if enemy is not None else -1)]

            soldiers_data.append(data)

        data = np.array(soldiers_data)

        predictions = pygad.nn.predict(last_layer=self.neural_net, data_inputs=data)

        for idx, pos in enumerate(troops):
            x = pos[0]
            y = pos[1]

            move = self.cenas[predictions[idx]] 

            if move is not None:
                if move == 'upgrade':
                    actions.append(upgradeBase())
                    self.resources -= self.upgrade_cost

                elif 'recruit' in move:
                    cost = SOLDIER_MELEE_COST if 'melee' in move else SOLDIER_RANGED_COST
                    type = ALLIED_SOLDIER_MELEE if 'melee' in move else ALLIED_SOLDIER_RANGED
                    amount = self.resources // cost

                    recruitSoldiers(type, amount)
                    self.resources -= amount * cost

                else:
                    for dir in move:
                        if dir is not None:
                            dest = np.add([x,y], self.motions[dir]).astype(int)
                            amount = int(self.board[x,y,1] // len(move))
                            actions.append(moveSoldiers((x,y), dest, amount))
            
            # if not np.array_equal(move, [0, 0]):
                
            #     ratio = np.linalg.norm(move)
            #     move = np.divide(move, ratio)
            #     amount = int(self.board[x,y,1] // (1/ratio))
            #     dest = np.add([x,y], move)

            #     actions.append(moveSoldiers((x,y), dest.astype(int), amount))
     
        playActions(actions)

    """
    JS METHODS
    """
    @staticmethod
    def findEnemy(pos, enemies, distance = 3):

        open_nodes = [pos]
        visited = []
        actions = [(1,0), (-1,0), (0,1), (0,-1)]

        while open_nodes != []:
            node = open_nodes.pop(0)

            if node in enemies:
                return node

            visited.append(node)
            new_nodes = []

            for action in actions:
                new_node = (node[0]+action[0], node[1]+action[1])
                dist = sum([abs(new_node[0]-pos[0]), abs(new_node[1]-pos[1])])
                in_board = (0 <= new_node[0] <= WIDTH) and (0 <= new_node[1] <= HEIGHT)
                if new_node not in visited and dist <= distance and in_board:
                    new_nodes.append(new_node)

            # breath strategy
            open_nodes.extend(new_nodes)

        return None

"""
FUNCTIONS DEFINITIONS
"""
def debug(*args):
    _PRINT(*args, file=sys.stderr, flush=True)
    with open(DEBUG_FILE, 'a') as f:
        stdout, sys.stdout = sys.stdout, f # Save a reference to the original standard output
        _PRINT(*args)
        sys.stdout = stdout

print = debug # dont erase this line, otherwise you cannot use the print function for debug

def upgradeBase():
    return f"{UPGRADE_BUILDING}"

def recruitSoldiers(type, amount, location=(1,VCENTER)):
    return f"{RECRUIT_SOLDIERS}|{type}|{amount}|{location[0]}|{location[1]}".replace(" ","")

def moveSoldiers(pos, to, amount):
    return f"{MOVE_SOLDIERS}|{pos[0]}|{pos[1]}|{to[0]}|{to[1]}|{amount}".replace(" ","")

def playActions(actions):
    _PRINT(';'.join(map(str,actions)), flush=True)

def create_network(num_neurons_input, 
                   num_neurons_output, 
                   num_neurons_hidden_layers=[], 
                   output_activation="softmax", 
                   hidden_activations="relu"):

    # Creating the input layer as an instance of the nn.InputLayer class.
    input_layer = nn.InputLayer(num_neurons_input)

    if type(hidden_activations) not in [list,tuple]:
        hidden_activations = [hidden_activations]*len(num_neurons_hidden_layers)

    if len(num_neurons_hidden_layers) > 0:
        # If there are hidden layers, then the first hidden layer is connected to the input layer.
        hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                     previous_layer=input_layer, 
                                     activation_function=hidden_activations.pop(0))
        # For the other hidden layers, each hidden layer is connected to its preceding hidden layer.
        for hidden_layer_idx in range(len(num_neurons_hidden_layers)):
            hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                         previous_layer=hidden_layer, 
                                         activation_function=hidden_activations.pop(0))

        # The last hidden layer is connected to the output layer.
        # The output layer is created as an instance of the nn.DenseLayer class.
        output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                     previous_layer=hidden_layer, 
                                     activation_function=output_activation)

    # If there are no hidden layers, then the output layer is connected directly to the input layer.
    elif len(num_neurons_hidden_layers) == 0:
        # The output layer is created as an instance of the nn.DenseLayer class.
        output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                     previous_layer=input_layer,
                                     activation_function=output_activation)

    # Returning the reference to the last layer in the network architecture which is the output layer. Based on such reference, all network layer can be fetched.
    return output_layer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', action='store_true')
    parser.add_argument('-w', '--weights', type=str)
    args = vars(parser.parse_args())

    if args['run']: # Run mode
        neural_net = create_network(13,18,[15,15])
        weights = np.load(args['weights'])['arr_0']
        weights_matrix = nn.layers_weights_as_matrix(neural_net, weights)
        
        nn.update_layers_trained_weights(last_layer=neural_net,
                                        final_weights=weights_matrix)
        
    else: # Train mode
        # open nn pipe
        pid = os.getpid()
        name = '.pipes/' + str(pid) + '_nn'
        logger.debug('Reading neural network')
        with open(name, 'rb') as pipe:
            neural_net = pickle.load(pipe)
            logger.debug('Neural network loaded')
    
    open(DEBUG_FILE, 'w').close()
    difficulty, base_cost, base_prod = map(int,input().split())

    env = Environment(difficulty, base_cost, base_prod, neural_net=neural_net)
    while 1:
        signal = env.readEnvironment()
        if signal=="END":
            # debug("GAME OVER")
            logger.debug('GAME OVER')
            sys.exit(0)
        elif signal=="ERROR":
            # debug("ERROR")
            logger.debug('ERROR')
            sys.exit(1)

        env.play()
"""
MAIN
"""
if __name__ == '__main__':
    main()

    # tree = DecisionTree()
    # tree.buildSplitTree()

    # a = tree.selectMove([None, 1, None, None], tree.root)
    # print(a)