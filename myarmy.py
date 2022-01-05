#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import sys
import json
import numpy as np
import itertools

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
class Question:

    def __init__(self, field, optional=None):
        self.field = field
        self.optional = optional

    def check(self, data):
        if self.optional is not None:
            return self.question(data[self.field], self.optional)
        else:
            return self.question(data[self.field])

class RangedQuestion(Question):

    def __init__(self, field, optional=None):
        super().__init__(field, optional=optional)

        if field == 'amount':
            self.question = lambda amount, value : amount > value

        elif field in ['up', 'down', 'left', 'right']:
            self.question = lambda soldier : soldier in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]

        elif field == 'pos_x':
            self.question = lambda x : x < 10 # TODO 10 elsewhere

        elif field == 'pos_y':
            self.question = lambda y : y % 2 == 0

        elif field == 'enemy_d':
            self.question = lambda dist : dist <= 3 if dist is not None else False

class BaseQuestion(Question):

    def __init__(self, field, optional=None):
        super().__init__(field, optional=optional)
        
        if field == 'enemy_d':
            self.question = lambda dist, limit : dist < limit if dist is not None else False

        elif field == 'resources':
            self.question = lambda resources, cost : resources >= cost

        elif field == 'upgrade':
            self.question = lambda can_upgrade : bool(can_upgrade)

        elif field in ['up', 'down', 'left', 'right']:
            self.question = lambda soldier : soldier in [EMPTY_CELL] # TODO check also allied type

class LeafNode:

    def __init__(self, outcome):
        self.outcome = outcome

class DecisionTree:

    def __init__(self):
        pass

    @staticmethod
    def selectMove(data, node):

        if isinstance(node, LeafNode):
            return node.outcome

        if node.question.check(data):
            return DecisionTree.selectMove(data, node.true_branch)
        else:
            return DecisionTree.selectMove(data, node.false_branch)

    @staticmethod
    def buildRangedTree():

        # TODO different behaviors if close or further from base
        soldiers_up_down = np.array([(0,-1),(0,1)])
        soldiers_up = np.array([(0,-1)])
        soldiers_down = np.array([(0,1)])
        soldiers_back = np.array([(-1,0)])
        soldiers_forward = np.array([(1,0)])
        soldiers_nowhere = np.array([()])

        leaf_nothing = LeafNode(soldiers_nowhere)

        split_vert = DecisionNode(RangedQuestion('down'), LeafNode(soldiers_up_down), LeafNode(soldiers_up))
        move_back = DecisionNode(RangedQuestion('left'), LeafNode(soldiers_back), leaf_nothing)
        can_split = DecisionNode(RangedQuestion('amount', 1), split_vert, LeafNode(soldiers_up)) # Prioritize up

        move_down = DecisionNode(RangedQuestion('down'), LeafNode(soldiers_down), leaf_nothing)
        split_formattion = DecisionNode(RangedQuestion('up'), can_split, move_down)

        move_forward = DecisionNode(RangedQuestion('right'), LeafNode(soldiers_forward), leaf_nothing)

        move_formattion = DecisionNode(RangedQuestion('pos_x'), move_forward,leaf_nothing)

        formattion = DecisionNode(RangedQuestion('pos_y'), split_formattion, move_formattion)

        split_amount = DecisionNode(RangedQuestion('up'), can_split, move_back)
        move = DecisionNode(RangedQuestion('enemy_d'), leaf_nothing, formattion)

        root = DecisionNode(RangedQuestion('amount', 50), split_amount, move)

        return root

    @staticmethod
    def buildMeleeTree():
        pass

    @staticmethod
    def buildBaseTree():

        no_action = LeafNode(None)
        
        recruit = LeafNode([ALLIED_SOLDIER_MELEE] * 3)

        in_emergency = DecisionNode(BaseQuestion('enemy_d', 5), LeafNode(None), LeafNode(None))
        
        can_buy = DecisionNode(BaseQuestion('resources', min([SOLDIER_MELEE_COST, SOLDIER_RANGED_COST])), in_emergency, no_action)
        upgrade = DecisionNode(BaseQuestion('upgrade'), LeafNode(upgradeBase), no_action)

        root = DecisionNode(BaseQuestion('enemy_d', 15), can_buy, upgrade)

        return root

class DecisionNode:

    def __init__(self, question, true_branch, false_branch):
        self.question = question # Question the node makes
        self.true_branch = true_branch # Node if question is true
        self.false_branch = false_branch # Node if question is false

class Environment:
    def __init__(self, difficulty, base_cost, base_prod):
        self.difficulty = difficulty
        self.resources = 0
        self.building_level = 0
        self.base_cost = base_cost
        self.base_prod = base_prod
        self.board = [[None]*WIDTH for h in range(HEIGHT)]

        """
        JS STUFF
        """
        self.ranged_root = DecisionTree.buildRangedTree()
        self.base_root = DecisionTree.buildBaseTree()
        self.soldier_data = {}
        self.base_data = {}

        playActions([])

    @property
    def upgrade_cost(self):
        return int(self.base_cost*(1.4**self.building_level))

    @property
    def production(self):
        return int(self.base_prod*(1.2**self.building_level))

    def readEnvironment(self):
        state = input()

        if state in ["END", "ERROR"]:
            return state
        level, resources, board = state.split()
        level = int(level)
        resources = int(resources)
        debug(f"Building Level: {level}, Current resources: {resources}")
        self.building_level = level
        self.resources = resources

        self.board = np.swapaxes(np.array(json.loads(board)),0,1)

    def play(self): # agent move, call playActions only ONCE

        actions = []
        print("Current production per turn is:", self.production)
        print("Current building cost is:", self.upgrade_cost)

        # SOLDIERS DECISION MAKING
        soldiers = self.board[:,:,0]

        troops = np.argwhere((soldiers==ALLIED_SOLDIER_RANGED) | (soldiers==ALLIED_SOLDIER_MELEE))
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        enemies = [tuple(x) for x in enemies]

        for x,y in troops:

            soldier = self.board[x,y,0]

            enemy_pos = Environment.findEnemy((x,y), enemies)
            dist = sum([abs(x-enemy_pos[0]), abs(y-enemy_pos[1])]) if enemy_pos is not None else None
            
            self.soldier_data['amount'] = self.board[x,y,1]
            self.soldier_data['resources'] = self.resources
            self.soldier_data['up'] = self.board[x,y-1,0] if y > 0 else WALL
            self.soldier_data['down'] = self.board[x,y+1,0] if y < HEIGHT-1 else WALL
            self.soldier_data['right'] = self.board[x+1,y,0] if x < WIDTH-1 else WALL
            self.soldier_data['left'] = self.board[x-1,y,0] if x > 0 else WALL
            self.soldier_data['pos_x'] = x
            self.soldier_data['pos_y'] = y
            self.soldier_data['enemy_d'] = dist

            moves = []
            if soldier == ALLIED_SOLDIER_RANGED:
                moves = DecisionTree.selectMove(self.soldier_data, self.ranged_root)

            elif soldier == ALLIED_SOLDIER_MELEE:
                pass

            for move in moves:
                if len(move) == 2:
                    action = moveSoldiers((x,y), np.add((x,y), move), self.board[x,y,1]//len(moves))
                    actions.append(action)

        # BASE DECISION MAKING
        self.base_data['resources'] = self.resources
        self.base_data['upgrade'] = (self.resources >= self.upgrade_cost)
        self.base_data['up'] = self.board[0,VCENTER-1,0] if y > 0 else WALL
        self.base_data['down'] = self.board[0,VCENTER+1,0] if y < HEIGHT-1 else WALL
        self.base_data['right'] = self.board[1,VCENTER,0] if x < WIDTH-1 else WALL
        self.base_data['enemy_d'] = min([pos[0] for pos in enemies]) if len(enemies) > 0 else None

        decision = DecisionTree.selectMove(self.base_data, self.base_root)

        if callable(decision):
            actions.append(decision())
        

        
                
        

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

def main():

    open(DEBUG_FILE, 'w').close()
    difficulty, base_cost, base_prod = map(int,input().split())

    env = Environment(difficulty, base_cost, base_prod)
    while 1:
        signal = env.readEnvironment()
        if signal=="END":
            debug("GAME OVER")
            sys.exit(0)
        elif signal=="ERROR":
            debug("ERROR")
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