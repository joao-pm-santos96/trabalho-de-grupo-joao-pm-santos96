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

    def __init__(self, type, col, optional=None):
        self.type = type
        self.col = col
        self.optional = optional

        if col == 1:
            self.question = lambda amount, value : amount > value

        elif col in [3, 4, 5, 6]:
            self.question = lambda can : bool(can) == True

        elif col == 7:
            self.question = lambda x : x < 20

        elif col == 8:
            self.question = lambda y : y % 2 == 0

        elif col == 9:
            self.question = lambda dist : dist <= 3 if dist is not None else False

    def check(self, data):
        if self.optional is not None:
            return self.question(data[self.col], self.optional)
        else:
            return self.question(data[self.col])

class LeafNode:

    def __init__(self, outcome):
        self.outcome = outcome

class DecisionTree:

    def __init__(self):
        self.root = None

    # def buildSplitTree(self):

    #     split_up_down = LeafNode([(0,1), (0,-1)])
    #     split_up = LeafNode([(0,1)])
    #     split_down = LeafNode([(0,-1)])
    #     split_back = LeafNode([(-1,0), (0,0)])
    #     no_split = LeafNode([(0,0)])

    #     can_move_back = DecisionNode(Question(3), split_back, no_split)
    #     can_move_down_1 = DecisionNode(Question(1), split_down, can_move_back)
    #     can_move_down_2 = DecisionNode(Question(1), split_up_down, split_up)

    #     return DecisionNode(Question(0), can_move_down_2, can_move_down_1)

    def buildRangedTree(self):

        soldier = ALLIED_SOLDIER_RANGED

        soldiers_up_down = np.array([(0,1),(0,-1)])
        soldiers_up = np.array([(0,-1)])
        soldiers_down = np.array([(0,1)])
        soldiers_back = np.array([(-1,0)])
        soldiers_forward = np.array([(1,0)])
        soldiers_nowhere = np.array([()])

        leaf_nothing = LeafNode(soldiers_nowhere)

        split_vert = DecisionNode(Question(soldier, 4), LeafNode(soldiers_up_down), LeafNode(soldiers_up))
        move_back = DecisionNode(Question(soldier, 6), LeafNode(soldiers_back), leaf_nothing)
        can_split = DecisionNode(Question(soldier, 1, 1), split_vert, LeafNode(soldiers_up)) # Prioritize up

        move_down = DecisionNode(Question(soldier, 4), LeafNode(soldiers_down), leaf_nothing)
        split_formattion = DecisionNode(Question(soldier, 3), can_split, move_down)

        move_forward = DecisionNode(Question(soldier, 5), LeafNode(soldiers_forward), leaf_nothing)

        move_formattion = DecisionNode(Question(soldier, 7), move_forward,leaf_nothing)

        formattion = DecisionNode(Question(soldier, 8), split_formattion, move_formattion)

        split_amount = DecisionNode(Question(soldier, 3), can_split, move_back)
        move = DecisionNode(Question(soldier, 9), leaf_nothing, formattion)

        root = DecisionNode(Question(soldier, 1, 50), split_amount, move)

        return root

    @staticmethod
    def selectMove(data, node):

        if isinstance(node, LeafNode):
            return node.outcome

        if node.question.check(data):
            return DecisionTree.selectMove(data, node.true_branch)
        else:
            return DecisionTree.selectMove(data, node.false_branch)


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
        self.decision = DecisionTree()
        self.root = self.decision.buildRangedTree()

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

        # # Split Up/Down
        # col_filled = np.sum(np.in1d(self.board[1,:,0], [ALLIED_SOLDIER_RANGED, ALLIED_SOLDIER_MELEE])) == HEIGHT
        # if not col_filled:
        #     for y in range(1, HEIGHT-1):

        #         amount = self.board[1, y, 1]
        #         orig = (1,y)

        #         if y < VCENTER and amount > 1 and self.board[1, y-1, 0] in [EMPTY_CELL, self.board[1, y, 0]]:
        #             actions.append(moveSoldiers(orig, np.add(orig, (0,-1)), amount//2))

        #         elif y > VCENTER and amount > 1 and self.board[1, y+1, 0] in [EMPTY_CELL, self.board[1, y, 0]]:
        #             actions.append(moveSoldiers(orig, np.add(orig, (0,1)), amount//2))

        #         elif y == VCENTER and amount > 1:
        #             if self.board[1, y+1, 0] in [EMPTY_CELL, self.board[1, y, 0]]:
        #                 actions.append(moveSoldiers(orig, np.add(orig, (0,1)), self.board[1, VCENTER, 1]//2))
        #             if self.board[1, y-1, 0] in [EMPTY_CELL, self.board[1, y, 0]]:
        #                 actions.append(moveSoldiers(orig, np.add(orig, (0,-1)), self.board[1, VCENTER, 1]//2))
        # else:
        #     for y in range(HEIGHT):
        #         if self.board[1, y, 0] in [ALLIED_SOLDIER_RANGED, ALLIED_SOLDIER_MELEE]:
        #             actions.append(moveSoldiers((1,y), (2,y), self.board[1, y, 1]))


        # # Move column forward
        # for x in range(2, WIDTH-1):
        #     my_soldiers = self.board[x, :, 0]
        #     next_cells = self.board[x+1, :, 0]

        #     for y in range(HEIGHT):
        #         if next_cells[y] in [EMPTY_CELL, my_soldiers[y]] and my_soldiers[y] in [ALLIED_SOLDIER_MELEE, ALLIED_SOLDIER_RANGED]:
        #             actions.append(moveSoldiers((x,y), (x+1,y), self.board[x, y, 1]))







        # # # Recruit ranged
        # # buy_amount = self.resources//SOLDIER_RANGED_COST
        # # if self.resources >= SOLDIER_RANGED_COST * buy_amount:
        # #     actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, buy_amount))
        # #     self.resources -= buy_amount*SOLDIER_RANGED_COST

        # # Recruit melee
        # buy_amount = 21
        # if self.resources >= SOLDIER_MELEE_COST * buy_amount and self.board[1,VCENTER,0] in [ALLIED_SOLDIER_MELEE, EMPTY_CELL]:
        #     actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, buy_amount))
        #     self.resources -= buy_amount*SOLDIER_MELEE_COST








        # if self.resources >= self.upgrade_cost: # upgrade building
        #     actions.append(upgradeBase())
        #     self.resources -= self.upgrade_cost




        soldiers = self.board[:,:,0]
        # troops = np.argwhere((soldiers == ALLIED_MAIN_BUILDING) | (soldiers==ALLIED_SOLDIER_RANGED) | (soldiers==ALLIED_SOLDIER_MELEE))

        troops = np.argwhere(soldiers==ALLIED_SOLDIER_RANGED)
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))

        for x,y in troops:

            soldier = self.board[x,y,0]

            enemy_pos = Environment.findEnemy((x,y), [tuple(x) for x in enemies])
            dist = sum([abs(x-enemy_pos[0]), abs(y-enemy_pos[1])]) if enemy_pos is not None else None
            
            data = [soldier,
                self.board[x,y,1],
                self.resources,
                self.board[x,y-1,0] in [EMPTY_CELL, soldier] if y > 0 else False,
                self.board[x,y+1,0] in [EMPTY_CELL, soldier] if y < HEIGHT-1 else False,
                self.board[x+1,y,0] in [EMPTY_CELL, soldier] if x < WIDTH-1 else False,
                self.board[x-1,y,0] in [EMPTY_CELL, soldier] if x > 0 else False,
                x,
                y,
                dist]                

            moves = self.decision.selectMove(data, self.root)

            for move in moves:
                if len(move) == 2:
                    action = moveSoldiers((x,y), np.add((x,y), move), self.board[x,y,1]//len(moves))
                    actions.append(action)

        playActions(actions)




    """
    JS METHODS
    """
    @staticmethod
    def findEnemy(pos, enemies, limit = 3):

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
                if new_node not in visited and dist <= limit:
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