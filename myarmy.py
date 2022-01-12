import sys
import random

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

        if self.difficulty == 0:
            actions = self.dif0()
        elif self.difficulty == 1:
            actions = self.dif1()

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
        actions = uniq 
        
        self.turn += 1  
        playActions(actions)


    def dif0(self):
        actions = []

        soldiers = self.board[:,:,0]
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        enemies = [tuple(x) for x in enemies]

        formation_rows = [4,6] 
        formation_col = 14
        initial_col = 5
        initial_desired_level = 7
        max_desired_lvl = 14 
        
        for x in range(WIDTH):
            for y in range(HEIGHT):

                soldier_type = self.board[x,y,0]
                soldier_amount = self.board[x,y,1]

                y_dir = [1,-1][y<VCENTER]

                if soldier_type == ALLIED_MAIN_BUILDING:

                    formation1 = np.all(self.board[formation_col-1:formation_col+1,0:formation_rows[0]+1,1] == 50)
                    formation2 = np.all(self.board[formation_col-1:formation_col+1,formation_rows[1]:HEIGHT,1] == 50)

                    buy_condition = formation1 and formation2 and self.board[0,VCENTER,1] < max_desired_lvl

                    if self.board[0,VCENTER,1] < initial_desired_level \
                        or buy_condition:

                        if self.resources >= self.upgrade_cost:
                            actions.append(upgradeBase())
                            self.resources -= self.upgrade_cost                        

                    else: 

                        # set amounts
                        melee_amount = 20 
                        ranged_amount = int((self.resources - melee_amount * SOLDIER_MELEE_COST) // SOLDIER_RANGED_COST )

                        # recruit ranges
                        ranged_condition = ranged_amount > 1 \
                            and self.resources >= ranged_amount * SOLDIER_RANGED_COST 

                        recruit_pos = [0] * 2
                        recruit_pos[0] = (self.board[0,VCENTER-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]) * -1 # up
                        recruit_pos[1] = self.board[0,VCENTER+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] # down

                        for pos in recruit_pos:
                            if pos and ranged_condition:
                                actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, ranged_amount//np.count_nonzero(recruit_pos), (0,VCENTER+pos)))
                                self.resources -= ranged_amount//np.count_nonzero(recruit_pos) * SOLDIER_RANGED_COST

                        # recruit melee
                        melee_amount = melee_amount if (melee_amount <= self.resources // SOLDIER_MELEE_COST) else self.resources // SOLDIER_MELEE_COST
                        melee_condition = melee_amount > 0 \
                            and self.resources >= melee_amount * SOLDIER_MELEE_COST \
                            and self.turn % 1 == 0

                        if self.board[1,VCENTER,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                            and melee_condition:

                            actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, melee_amount))
                            self.resources -= melee_amount * SOLDIER_MELEE_COST

                elif soldier_type == ALLIED_SOLDIER_MELEE:
                    
                    # closest_enemy = Environment.findEnemy((x,y),enemies)
                    max_soldiers = 20

                    if x > formation_col \
                        and y not in formation_rows \
                        and soldier_amount > max_soldiers \
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

                            actions.append(moveSoldiers((x,y), (x+1,y), soldier_amount))

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
                                    actions.append(moveSoldiers((x,y), (x-1,y), soldier_amount))

                            if dest != (0,0):
                                actions.append(moveSoldiers((x,y), dest, self.board[x,y,1]))

                elif soldier_type == ALLIED_SOLDIER_RANGED:

                    max_soldiers = 50

                    if soldier_amount > max_soldiers and self.board[x+1,y,0] == ALLIED_SOLDIER_RANGED:

                        delta = soldier_amount - max_soldiers

                        if self.board[x-1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] if (x - 1) > 0 else False:
                            actions.append(moveSoldiers((x,y),(x-1,y), delta))

                    elif Environment.findEnemy((x,y), enemies) is None: # enemy not in range

                        if y not in formation_rows and x < initial_col: # split in formation

                            # find closest row
                            min_idx = np.argmin([abs(y-formation_rows[0]), abs(y-formation_rows[1])])

                            # move towards that row
                            y_dir = np.sign(formation_rows[min_idx] - y)

                            if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                                actions.append(moveSoldiers((x,y),(x,y+y_dir), self.board[x,y,1]))

                        else:

                            if not np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]) \
                                and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:

                                amount_frwd = min([self.board[x,y,1], max_soldiers-self.board[x+1,y,1]]) 

                                if amount_frwd > 0 and x < formation_col: # move forward until max reached or in formation column
                                    actions.append(moveSoldiers((x,y),(x+1,y), amount_frwd))

                            elif (y+y_dir) >= 0 and (y+y_dir) <= HEIGHT-1:
                                
                                if np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]) \
                                    and self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: 

                                    amount_y_dir = min([self.board[x,y,1], max_soldiers-self.board[x,y+y_dir,1]]) 

                                    if amount_y_dir > 0: # move up/down until column reached
                                        actions.append(moveSoldiers((x,y),(x,y+y_dir), amount_y_dir))

        return actions

    def dif1(self):
        actions = []

        soldiers = self.board[:,:,0]
        enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        enemies = [tuple(x) for x in enemies]

        formation_col = 14
        initial_desired_level = 5

        for x in range(WIDTH):
            for y in range(HEIGHT):

                soldier_type = self.board[x,y,0]
                soldier_amount = self.board[x,y,1]

                y_dir = [1,-1][y<VCENTER]

                if soldier_type == ALLIED_MAIN_BUILDING:

                    if self.board[0,VCENTER,1] < initial_desired_level:

                        if self.resources >= self.upgrade_cost:
                            actions.append(upgradeBase())
                            self.resources -= self.upgrade_cost

                    else:
                    
                        # # Recruit melee
                        # melee_pos = (0,VCENTER-1)
                        # amount = self.resources // SOLDIER_MELEE_COST

                        # if self.resources >= SOLDIER_MELEE_COST \
                        #     and self.board[melee_pos[0], melee_pos[1], 0] == EMPTY_CELL \
                        #     and amount >= 20:

                        #     amount = 20
                        #     actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount, melee_pos))
                        #     self.resources -= amount * SOLDIER_MELEE_COST
                        
                        # # Recruit ranged
                        # amount = self.resources // SOLDIER_RANGED_COST
                        # if amount > 2:
                        #     actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, amount))
                        #     self.resources -= amount * SOLDIER_RANGED_COST

                        # set amounts
                        melee_amount = 20 
                        ranged_amount = int((self.resources - melee_amount * SOLDIER_MELEE_COST) // SOLDIER_RANGED_COST )

                        # recruit ranges
                        ranged_condition = ranged_amount > 0 \
                            and self.resources >= ranged_amount * SOLDIER_RANGED_COST \
                            and self.board[1,VCENTER,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]

                        if ranged_condition:
                            actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, ranged_amount))
                            self.resources -= ranged_amount * SOLDIER_RANGED_COST

                        # recruit melee
                        melee_amount = melee_amount if (melee_amount <= self.resources // SOLDIER_MELEE_COST) else self.resources // SOLDIER_MELEE_COST
                        melee_condition = melee_amount > 0 \
                            and self.board[0,VCENTER-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]

                        if melee_condition:
                            actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, melee_amount, (0,VCENTER-1)))
                            self.resources -= melee_amount * SOLDIER_MELEE_COST

                    



                elif soldier_type == ALLIED_SOLDIER_MELEE:
                    
                    max_soldiers = 20

                    # Move forward in the upper row (when invisible), as less enemies
                    if soldier_amount <= max_soldiers:
                        if y == 0 and self.board[x+1,y,0] not in [ALLIED_SOLDIER_RANGED]:
                            actions.append(moveSoldiers((x,y), (x+1,y), self.board[x,y,1]))

                        elif 0 < y < HEIGHT and self.board[x,y-1,0] not in [ALLIED_SOLDIER_RANGED]:
                            actions.append(moveSoldiers((x,y), (x,y-1), self.board[x,y,1]))




                
                elif soldier_type == ALLIED_SOLDIER_RANGED:
                    
                    max_soldiers = 50
                    formation_rows = list(range(1, HEIGHT-1))

                    if y not in formation_rows: # Leave top and bottom rows free
                        
                        if y == 0 and self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # down 
                            actions.append(moveSoldiers((x,y),(x,y+1), self.board[x,y,1]))
                        
                        elif y == HEIGHT - 1 and self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # up
                            actions.append(moveSoldiers((x,y),(x,y-1), self.board[x,y,1]))                    


                    else: # Move to battle front

                        if Environment.findEnemy((x,y), enemies) is None:
                            
                            if self.board[x+1,y,1] < max_soldiers \
                                and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # forward motion

                                amount_frwd = max_soldiers - self.board[x+1,y,1]
                                amount_frwd = min([amount_frwd, self.board[x,y,1]])

                                if x < formation_col and amount_frwd > 0:
                                    actions.append(moveSoldiers((x,y),(x+1,y), amount_frwd))

                            
                            elif self.board[x+1,y,1] >= max_soldiers \
                                or x >= formation_col: # split

                                if y == VCENTER:
                                    
                                    dirs = [1,-1]
                                    random.shuffle(dirs)

                                    amount0 = soldier_amount // 2
                                    amount0 = min([amount0, max_soldiers - self.board[x,y+dirs[0],1]])

                                    amount1 = soldier_amount - amount0
                                    amount1 = min([amount1, max_soldiers - self.board[x,y+dirs[1],1]])

                                    if amount0 > 0 and self.board[x,y+dirs[0],0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                                        actions.append(moveSoldiers((x,y),(x,y+dirs[0]), amount0))

                                    if amount1 > 0 and self.board[x,y+dirs[1],0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                                        actions.append(moveSoldiers((x,y),(x,y+dirs[1]), amount1))

                                elif y in formation_rows[1:-1]:

                                    if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                                        and self.board[x,y+y_dir,1] < self.board[x,y,1]:

                                        delta = self.board[x,y,1] - self.board[x,y+y_dir,1]

                                        amount = min([delta, max_soldiers-self.board[x,y+y_dir,1]])

                                        if amount > 0:
                                            actions.append(moveSoldiers((x,y),(x,y+y_dir), amount))

                                






                                    







                            
                            # if x < formation_col \
                            #     and not np.array_equal(self.board[x+1,y],[ALLIED_SOLDIER_RANGED, max_soldiers]):

                            #     if self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                            #         actions.append(moveSoldiers((x,y),(x+1,y), self.board[x,y,1]))

                            # else: # split them

                            #     # if y == VCENTER:
                            #     #     amount_up = soldier_amount//2
                            #     #     amount_down = soldier_amount - amount_up

                            #     #     if amount_up > 0 \
                            #     #         and self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                            #     #         and not np.array_equal(self.board[x,y-1],[ALLIED_SOLDIER_RANGED, max_soldiers]):

                            #     #         actions.append(moveSoldiers((x,y),(x,y-1), amount_up))

                            #     #     if amount_down > 0 \
                            #     #         and self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                            #     #         and not np.array_equal(self.board[x,y+1],[ALLIED_SOLDIER_RANGED, max_soldiers]):

                            #     #         actions.append(moveSoldiers((x,y),(x,y+1), amount_down))

                            #     if y not in [1, HEIGHT-2]:

                            #         amount = soldier_amount // 2
                            #         delta = max_soldiers - self.board[x, y+y_dir, 1]

                            #         if delta < amount:
                            #             amount = delta

                            #         if amount > 0:
                            #             actions.append(moveSoldiers((x,y), (x,y+y_dir), amount))

                            
                            













        return actions

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


