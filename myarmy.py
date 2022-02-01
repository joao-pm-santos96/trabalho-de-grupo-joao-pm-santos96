#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import sys
import random

from actions import *
from utils import *
from utils import _PRINT

import json
import numpy as np
import time

"""
GLOBALS
"""
DEBUG_FILE = "client_debug.txt"

"""
METADATA
"""
__author__ = 'Joao Santos'
__copyright__ = 'Copyright January2022'
__credits__ = ['Joao Santos']
__version__ = '1.0.0'
__maintainer__ = 'Joao Santos'


"""
TODO
- emergency mode when formattion is broken beyond repair
"""

"""
CLASS DEFINITIONS
"""
class Environment:
    def __init__(self, difficulty, base_cost, base_prod):
        self.difficulty = difficulty
        self.resources = 0
        self.building_level = 0
        self.base_cost = base_cost
        self.base_prod = base_prod
        self.board = [[None]*WIDTH for h in range(HEIGHT)]

        self.enemies = None
        self.turn = 0
        self.initial_cols = [4,5]        
        
        self.max_melee = 20
        self.max_ranged = 50

        if self.difficulty == 0:
            self.formation_col = 20
            # self.level_steps = np.array([7, 14, 22])
            self.level_steps = np.array([6, 18, 23])
            self.formation_rows = [4,6] 
        else:
            self.formation_col = 15
            # self.level_steps = np.array([5, 10, 14])
            self.level_steps = np.array([5, 15, 18])
            self.formation_rows = list(range(1, HEIGHT-1))

        self.extra_upgrd_cond = np.zeros(self.level_steps.shape, dtype=bool)
        self.in_panic = False

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

        tic = time.time()

        actions = []
        print("Current production per turn is:", self.production)
        print("Current building cost is:", self.upgrade_cost)

        soldiers = self.board[:,:,0]
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        self.enemies = [tuple(x) for x in enemies]

        if self.board[0, VCENTER, 1] >= self.level_steps[1]:
            self.max_ranged = 250
            # if self.difficulty == 0:
            self.formation_col = WIDTH-5 # TODO 26?

        if self.board[0, VCENTER, 1] >= self.level_steps[2]:
            self.max_ranged = 500

        # if np.any(enemies[:,0] < self.formation_col):
        #     self.in_panic = True

        self.in_panic = np.any(enemies[:,0] < self.formation_col)

        # if self.difficulty == 1 and self.board[0,VCENTER,1] >= self.level_steps[-1]:
        #     self.formation_rows = list(range(0, HEIGHT))

        for x in range(WIDTH):
            for y in range(HEIGHT):

                soldier_type = self.board[x,y,0]
                soldier_amount = self.board[x,y,1]
                y_dir = [1,-1][y<VCENTER]

                if soldier_type == ALLIED_MAIN_BUILDING:
                    actions.extend(self.buyStrategy())

                elif soldier_type == ALLIED_SOLDIER_MELEE:
                    if self.difficulty == 0:
                        actions.extend(self.meleeStrategy0(x,y,soldier_amount))
                    else:
                        actions.extend(self.meleeStrategy1(x,y,soldier_amount))

                elif soldier_type == ALLIED_SOLDIER_RANGED:
                    if self.difficulty == 0:
                        actions.extend(self.rangedStrategy0(x,y,soldier_amount))
                    else:
                        actions.extend(self.rangedStrategy1(x,y,soldier_amount))

        
        actions = Environment.filterActions(actions)
        
        self.turn += 1  
        playActions(actions)

        print(f'Play time: {(time.time() - tic)*1000:.3f} ms')

    def buyStrategy(self):

        actions = []
        cols_ok =  []

        if self.difficulty == 0:
            
            rows = np.arange(0, HEIGHT) # ignore top and bottom rows
            rows = rows[rows != VCENTER] # ignore center row

            for i in range(2):
                col = self.formation_col - i
                cols_ok.append(np.all((self.board[col, rows,0] == ALLIED_SOLDIER_RANGED) & (self.board[col, rows,1] >= self.max_ranged)))

            self.extra_upgrd_cond[0] = True
            self.extra_upgrd_cond[1] = cols_ok[0] and cols_ok[1]
            self.extra_upgrd_cond[2] = cols_ok[0] and cols_ok[1] # and cols_ok[2] and cols_ok[3]

        else:

            for i in range(3):
                col = self.formation_col - i
                cols_ok.append(np.all((self.board[col, self.formation_rows,0] == ALLIED_SOLDIER_RANGED) & (self.board[col, self.formation_rows,1] >= self.max_ranged)))

            self.extra_upgrd_cond[0] = True
            self.extra_upgrd_cond[1] = cols_ok[0] and cols_ok[1]
            self.extra_upgrd_cond[2] = cols_ok[0] and cols_ok[1]# and cols_ok[2]
            
        curr_lvl = self.board[0,VCENTER,1]
        next_lvl_idx = np.argwhere(self.level_steps > curr_lvl)[0,0] if np.any(self.level_steps > curr_lvl) else None
        
        upgr_condition = self.extra_upgrd_cond[next_lvl_idx] if next_lvl_idx is not None else False

        if upgr_condition:
            if self.resources >= self.upgrade_cost:
                actions.append(upgradeBase())
                self.resources -= self.upgrade_cost

        else:

            # set amounts
            # melee_amount = 20 if self.difficulty == 0 else 40
            if self.difficulty == 0:
                melee_amount = 20
            else: 
                melee_amount = 40 if (not self.in_panic) else 20

            ranged_amount = int((self.resources - melee_amount * SOLDIER_MELEE_COST) // SOLDIER_RANGED_COST )

            # recruit ranged
            ranged_min_amount = 2 if self.difficulty == 0 else 1
            ranged_cond = self.resources >= ranged_amount * SOLDIER_RANGED_COST \
                and ranged_amount >= ranged_min_amount

            if self.difficulty == 0:
                recruit_pos = [0] * 2
                recruit_pos[0] = (self.board[0,VCENTER-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]) * -1 # up
                recruit_pos[1] = self.board[0,VCENTER+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] # down

                for pos in recruit_pos:
                    if pos and ranged_cond:
                        actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, ranged_amount//np.count_nonzero(recruit_pos), (0,VCENTER+pos)))
                        self.resources -= ranged_amount//np.count_nonzero(recruit_pos) * SOLDIER_RANGED_COST

            else:
                recruit_pos = self.board[1,VCENTER,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] # front

                if recruit_pos and ranged_cond:
                    actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, ranged_amount))
                    self.resources -= ranged_amount * SOLDIER_RANGED_COST

            # recruit melee
            if melee_amount > 0:
                melee_amount = melee_amount if (melee_amount <= self.resources // SOLDIER_MELEE_COST) else self.resources // SOLDIER_MELEE_COST

            melee_pos = [[1,VCENTER]] if self.difficulty == 0 else [[0,VCENTER-1],[0,VCENTER+1]]

            melee_cond = melee_amount > 0 \
                and self.resources >= melee_amount * SOLDIER_MELEE_COST 

            for pos in melee_pos:
                if melee_cond and self.board[pos[0], pos[1], 0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                    actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, melee_amount//len(melee_pos), pos))
                    self.resources -= melee_amount * SOLDIER_MELEE_COST

        return actions

    def meleeStrategy0(self, x, y, amount):

        actions = []
        y_dir = [1,-1][y<VCENTER]
        
        if x > self.formation_col \
            and y not in self.formation_rows \
            and amount > self.max_melee \
            and y not in [0,10]: # try to move to formattion row and/or forward avoiding enemies (sacrifice if only option is back)
            
            dest = (0,0)

            if y == VCENTER:
                y_dir = random.choice([1,-1])

            for d in [y_dir, -1 * y_dir]:
                if (y + d) >= 0 and (y + d) < HEIGHT:
                    if self.board[x,y+d,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                        dest=(x,y+d)
                        break

            if dest != (0,0):
                actions.append(moveSoldiers((x,y), dest, self.board[x,y,1]))

        else: # try to move forward avoiding enemies
            
            # try to go forward
            if self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                and (self.board[x+2,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]) if (x+2) < WIDTH - 1 else True:

                actions.append(moveSoldiers((x,y), (x+1,y), amount))

            elif y not in [0,10]:

                # try to go up or down
                dest = (0,0)
                dirs = [1,-1]
                random.shuffle(dirs)

                for d in dirs:
                    if (y + d) >= 0 and (y + d) < HEIGHT:
                        if self.board[x,y+d,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                            and self.board[x+1,y+d,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                            dest=(x,y+d)
                            break

                if dest == (0,0) and x-1 > -1:
                    if self.board[x-1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                        actions.append(moveSoldiers((x,y), (x-1,y), amount))

                if dest != (0,0):
                    actions.append(moveSoldiers((x,y), dest, self.board[x,y,1]))

        return actions

    def meleeStrategy1(self, x, y, amount):

        actions = []
        y_dir = [1,-1][y<VCENTER]

        # Move forward in the upper row (when invisible)
        if amount <= self.max_melee:
            
            if y in [0, HEIGHT-1] and self.board[x+1,y,0] not in [ALLIED_SOLDIER_RANGED]: # move forward
                actions.append(moveSoldiers((x,y), (x+1,y), self.board[x,y,1]))

            elif 0 < y <= VCENTER and self.board[x,y-1,0] not in [ALLIED_SOLDIER_RANGED]: # move upward
                actions.append(moveSoldiers((x,y), (x,y-1), self.board[x,y,1]))

            elif VCENTER < y < HEIGHT - 1 and self.board[x,y+1,0] not in [ALLIED_SOLDIER_RANGED]: # move downward
                actions.append(moveSoldiers((x,y), (x,y+1), self.board[x,y,1]))

        else:
            pass # TODO

        return actions

    def rangedStrategy0(self, x, y, amount):

        actions = []
        y_dir = [1,-1][y<VCENTER]

        if amount > self.max_ranged and self.board[x+1,y,0] == ALLIED_SOLDIER_RANGED:

            delta = amount - self.max_ranged

            if self.board[x-1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] if (x - 1) > 0 else False:
                actions.append(moveSoldiers((x,y),(x-1,y), delta))

        elif Environment.findEnemy((x,y), self.enemies) is None: # enemy not in range

            if y not in self.formation_rows and x < self.initial_cols[1]: # split in formation

                # find closest row
                min_idx = np.argmin([abs(y-self.formation_rows[0]), abs(y-self.formation_rows[1])])

                # move towards that row
                y_dir = np.sign(self.formation_rows[min_idx] - y)

                if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                    actions.append(moveSoldiers((x,y),(x,y+y_dir), self.board[x,y,1]))

            else:

                if not np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, self.max_ranged]) \
                    and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:

                    amount_frwd = min([self.board[x,y,1], self.max_ranged - self.board[x+1,y,1]]) 

                    if amount_frwd > 0 and x < self.formation_col: # move forward until max reached or in formation column
                        actions.append(moveSoldiers((x,y), (x+1,y), amount_frwd))

                elif (y+y_dir) >= 0 and (y+y_dir) <= HEIGHT-1:
                    
                    if np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, self.max_ranged]) \
                        and self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: 

                        amount_y_dir = min([self.board[x,y,1], self.max_ranged - self.board[x,y+y_dir,1]]) 

                        if amount_y_dir > 0: # move up/down until column reached
                            actions.append(moveSoldiers((x,y), (x,y+y_dir), amount_y_dir))

        return actions

    def rangedStrategy1(self, x, y, amount):

        actions = []
        y_dir = [1,-1][y<VCENTER]

        if y not in self.formation_rows: # Leave top and bottom rows free
                        
            if y == 0 and self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # down 
                actions.append(moveSoldiers((x,y),(x,y+1), self.board[x,y,1]))
            
            elif y == HEIGHT - 1 and self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # up
                actions.append(moveSoldiers((x,y),(x,y-1), self.board[x,y,1]))                    


        else: # Move to battle front

            if Environment.findEnemy((x,y), self.enemies) is None:
                
                if self.board[x+1,y,1] < self.max_ranged \
                    and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # forward motion

                    amount_frwd = self.max_ranged - self.board[x+1,y,1]
                    amount_frwd = min([amount_frwd, self.board[x,y,1]])

                    if x < self.formation_col and amount_frwd > 0:
                        actions.append(moveSoldiers((x,y),(x+1,y), amount_frwd))

                
                elif self.board[x+1,y,1] >= self.max_ranged \
                    or x >= self.formation_col: # split

                    if y == VCENTER:
                        
                        dirs = [1,-1]
                        random.shuffle(dirs)

                        amount0 = amount // 2
                        amount0 = min([amount0, self.max_ranged - self.board[x,y+dirs[0],1]])

                        amount1 = amount - amount0
                        amount1 = min([amount1, self.max_ranged - self.board[x,y+dirs[1],1]])

                        if amount0 > 0 and self.board[x,y+dirs[0],0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                            actions.append(moveSoldiers((x,y),(x,y+dirs[0]), amount0))

                        if amount1 > 0 and self.board[x,y+dirs[1],0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                            actions.append(moveSoldiers((x,y),(x,y+dirs[1]), amount1))

                    elif y in self.formation_rows[1:-1]:

                        if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                            and self.board[x,y+y_dir,1] < self.board[x,y,1]:

                            delta = self.board[x,y,1] - self.board[x,y+y_dir,1]

                            amount = min([delta, self.max_ranged - self.board[x,y+y_dir,1]])

                            if amount > 0:
                                actions.append(moveSoldiers((x,y),(x,y+y_dir), amount))
                
        return actions

    @staticmethod
    def filterActions(actions):
        # filter moves to same cell              
        seen = []
        uniq = []

        for action in actions:
            if '|' in action:
                data = action.split('|')
                if data[0] == str(MOVE_SOLDIERS):
                    tmp = '|' + data[3] + '|' + data[4] + '|'
                    if tmp not in seen:
                        seen.append(tmp)
                        uniq.append(action)
                else:
                    uniq.append(action)
            else:
                uniq.append(action)

        return uniq

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