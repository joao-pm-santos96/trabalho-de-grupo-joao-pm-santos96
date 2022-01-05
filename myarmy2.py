import sys


from actions import *
from utils import *
from utils import _PRINT

import json
import numpy as np


DEBUG_FILE = "client_debug.txt"

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



# ENVIRONMENT
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
        self.turn = 0
        self.initial_cols = [4,5]
        
        
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
        # self.board = json.loads(board)

        # uncomment next lines to use numpy array instead of array of array of array (3D array)
        # IT IS RECOMMENDED for simplicity
        # arrays to numpy converstion:  self.board[y][x][idx] => self.board[x,y,idx] 
        #
        self.board = np.swapaxes(np.array(json.loads(board)),0,1)
        #debug(self.board.shape)
        

    def play(self): # agent move, call playActions only ONCE

        actions = []
        print("Current production per turn is:", self.production)
        print("Current building cost is:", self.upgrade_cost)

        soldiers = self.board[:,:,0]
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        enemies = [tuple(x) for x in enemies]


        for x in range(WIDTH):
            for y in [VCENTER-1, VCENTER, VCENTER+1]:

                # Move Melee
                if self.board[x,y,0] == ALLIED_SOLDIER_MELEE \
                    and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE, ENEMY_SOLDIER_MELEE, ENEMY_SOLDIER_RANGED] \
                    and self.board[x,y,1] > 0:

                    actions.append(moveSoldiers((x,y),(x+1,y),self.board[x,y,1]))

                # Move Ranged
                if self.board[x,y,0] == ALLIED_SOLDIER_RANGED \
                    and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                    and self.board[x,y,1] > 0 \
                    and Environment.findEnemy((x,y), enemies) is None:

                    actions.append(moveSoldiers((x,y),(x+1,y),self.board[x,y,1]))





                    


        amount = self.resources // SOLDIER_MELEE_COST
        if amount > 2 and self.board[1,VCENTER,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:

            actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount // 3, (1, VCENTER)))
            actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount // 3, (0, VCENTER-1)))
            actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount // 3, (0, VCENTER+1)))


            self.resources -= amount * 3 * SOLDIER_MELEE_COST








        # amount = 16
        # soldier_type = ALLIED_SOLDIER_RANGED
        # cost = SOLDIER_RANGED_COST

        # if self.resources >= amount * cost:
        #     if self.board[0, VCENTER-1, 0] in [EMPTY_CELL, soldier_type]:
        #         actions.append(recruitSoldiers(soldier_type, amount//2,(0,VCENTER-1)))
        #         self.resources -= amount//2 * cost

        #     if self.board[0, VCENTER+1, 0] in [EMPTY_CELL, soldier_type]:
        #         actions.append(recruitSoldiers(soldier_type, amount//2,(0,VCENTER+1)))
        #         self.resources -= amount//2 * cost

        # for pos in [(0,VCENTER-1), (0,VCENTER+1)]:

        #     if self.board[pos[0], pos[1], 1] > 0 \
        #         and self.board[pos[0]+1, pos[1], 0] in [EMPTY_CELL, self.board[pos[0], pos[1], 0]] \
        #         and self.board[pos[0], pos[1], 0] in [ALLIED_SOLDIER_RANGED, ALLIED_SOLDIER_MELEE]:

        #         actions.append(moveSoldiers(pos, (pos[0]+1, pos[1]), self.board[pos[0], pos[1], 1]))

        # amount = 20
        # soldier_type = ALLIED_SOLDIER_MELEE
        # cost = SOLDIER_MELEE_COST
        # if self.resources >= amount * cost:
        #     if self.board[0, VCENTER-1, 0] in [EMPTY_CELL, soldier_type]:
        #         actions.append(recruitSoldiers(soldier_type, amount))
        #         self.resources -= amount * cost

        # if self.board[0, VCENTER, 1] > 3:

        #     amount = self.resources // SOLDIER_MELEE_COST
        #     actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount, (0, VCENTER+1)))
        #     self.resources -= SOLDIER_MELEE_COST * amount

        
        self.turn += 1
        playActions(actions)

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
        

if __name__ == "__main__":
    main()

