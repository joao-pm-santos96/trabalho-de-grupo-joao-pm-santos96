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
            for y in range(HEIGHT):

                soldier_type = self.board[x,y,0]
                soldier_amount = self.board[x,y,1]

                if soldier_type == ALLIED_MAIN_BUILDING:
                    
                    # recruit ranged
                    amount = 20 if self.board[0,VCENTER,1] < 5 else 40
                    if self.board[0,VCENTER-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                        and self.board[0,VCENTER+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                        and self.resources >= amount * SOLDIER_RANGED_COST:

                        actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, amount//2, (0,VCENTER-1)))
                        actions.append(recruitSoldiers(ALLIED_SOLDIER_RANGED, amount//2, (0,VCENTER+1)))
                        self.resources -= amount * SOLDIER_RANGED_COST

                    amount = 20 if self.turn % 5 != 0 else 30
                    if self.board[1,VCENTER,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                        and self.resources >= amount * SOLDIER_RANGED_COST \
                        and amount > 0:

                        actions.append(recruitSoldiers(ALLIED_SOLDIER_MELEE, amount))
                        self.resources -= amount * SOLDIER_MELEE_COST

                    if self.resources >= self.upgrade_cost:
                        actions.append(upgradeBase())
                        self.resources -= self.upgrade_cost

                elif soldier_type == ALLIED_SOLDIER_MELEE:
                    
                    closest_enemy = Environment.findEnemy((x,y),enemies)
                    if soldier_amount <= 50: 
                        
                        # try to go forward
                        if self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                            actions.append(moveSoldiers((x,y), (x+1,y), soldier_amount))

                        elif (y + 1) < HEIGHT - 1: # try to go down
                            if self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                                and self.board[x+1,y+1,0] not in [ENEMY_SOLDIER_MELEE, ENEMY_SOLDIER_RANGED] \
                                and self.board[x-1,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]: 

                                actions.append(moveSoldiers((x,y), (x,y+1), soldier_amount))

                        elif (y - 1) > 0: # try to go up
                            if self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE] \
                                and self.board[x+1,y-1,0] not in [ENEMY_SOLDIER_MELEE, ENEMY_SOLDIER_RANGED] \
                                and self.board[x-1,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:

                                actions.append(moveSoldiers((x,y), (x,y-1), soldier_amount))

                    elif closest_enemy is not None: 
                        delta = [closest_enemy[0]-x,closest_enemy[1]-y]

                        direction = np.argmax(np.abs(delta))
                        towards = np.sign(delta[direction])


                        go_to = [0]*2
                        go_to[direction] = towards 

                        print((x,y))
                        print(delta)
                        print(direction)
                        print(towards)
                        print(go_to)

                        dest = np.add((x,y), go_to).astype(int)

                        if self.board[dest[0], dest[1],0] in [EMPTY_CELL, ALLIED_SOLDIER_MELEE]:
                            actions.append(moveSoldiers((x,y),dest,soldier_amount))


                elif soldier_type == ALLIED_SOLDIER_RANGED:

                    formation_rows = [3, 7]
                    formation_col = 6
                    max_soldiers = 50
                    initial_col = 5

                    if Environment.findEnemy((x,y), enemies) is None: # enemy not in range

                        if y not in formation_rows and x < initial_col: # split in formation

                            # find closest row
                            min_idx = np.argmin([abs(y-formation_rows[0]), abs(y-formation_rows[1])])

                            # move towards that row
                            y_dir = np.sign(formation_rows[min_idx] - y)

                            if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                                actions.append(moveSoldiers((x,y),(x,y+y_dir), self.board[x,y,1]))

                        else:

                            y_dir = [1,-1][y<VCENTER]

                            if not np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]) \
                                and self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:

                                amount_frwd = min([self.board[x,y,1], max_soldiers-self.board[x+1,y,1]]) 

                                if amount_frwd > 0 and x < formation_col: # move forward until max reached or in formation collumn
                                    actions.append(moveSoldiers((x,y),(x+1,y), amount_frwd))

                            elif (y+y_dir) >= 0 and (y+y_dir) < HEIGHT:
                                
                                if np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]) \
                                    and self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: 

                                    amount_y_dir = min([self.board[x,y,1], max_soldiers-self.board[x,y+y_dir,1]]) 

                                    if amount_y_dir > 0: # move up/down until column reached
                                        actions.append(moveSoldiers((x,y),(x,y+y_dir), amount_y_dir))


                            




                                




                        # elif y in formation_rows and x < formation_col \
                        #     and not np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]): # go forward
                            
                        #     if self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                        #         and ((self.board[x,y,1] or 0) + (self.board[x+1,y,1] or 0))  <= max_soldiers: # only if sum will not get higher than threshold

                        #         actions.append(moveSoldiers((x,y),(x+1,y), self.board[x,y,1]))

                        #     elif self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] \
                        #         and ((self.board[x,y,1] or 0) + (self.board[x+1,y,1] or 0))  > max_soldiers: # split soldiers to allway be 50 at max

                        #         amount_frwd = min([self.board[x,y,1], max_soldiers-self.board[x+1,y,1]])
                        #         if amount_frwd > 0:
                        #             actions.append(moveSoldiers((x,y),(x+1,y), amount_frwd))

                        # elif y in formation_rows and x < formation_col \
                        #     and np.array_equal(self.board[x+1,y], [ALLIED_SOLDIER_RANGED, max_soldiers]): #  go up/down

                        #     y_dir = [1,-1][y<VCENTER]

                        #     if self.board[x,y+y_dir,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                        #         actions.append(moveSoldiers((x,y),(x,y+y_dir), self.board[x,y,1]))



                                















                        # if y in [3, 7, 5] and x < 20:

                        #     if self.board[x+1,y,0] == EMPTY_CELL:
                        #         actions.append(moveSoldiers((x,y),(x+1,y),soldier_amount))

                        # # split them in formation 
                        # elif y not in [3, 7] and x > 2 and x < 20:

                        #     if y in [0,1,2,6] and self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # move towards 3
                        #         actions.append(moveSoldiers((x,y), (x,y+1), soldier_amount))

                        #     elif y in [10,9,8,4] and self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]: # move towards 7
                        #         actions.append(moveSoldiers((x,y), (x,y-1), soldier_amount))

                        #     elif y == 5 and self.board[x,y+1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] and soldier_amount > 1:
                        #         actions.append(moveSoldiers((x,y), (x,y+1), soldier_amount//2))

                        #     elif y == 5 and self.board[x,y-1,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED] and soldier_amount > 1:
                        #         actions.append(moveSoldiers((x,y), (x,y-1), soldier_amount - soldier_amount//2))

                        # elif x == 20 and y not in [0,10]:
                        #     # try go forward
                        #     if self.board[x+1,y,0] == ALLIED_SOLDIER_RANGED and self.board[x+1,y,1] < 50 or self.board[x+1,y,0] == EMPTY_CELL:
                        #         cenas = min([self.board[x,y,1], 50-self.board[x+1,y,1]])
                        #         actions.append(moveSoldiers((x,y),(x+1,y),cenas))

                        #     # try move up
                        #     elif self.board[x,y-1,0] == EMPTY_CELL and self.board[x+1,y,1] >= 50 and y < VCENTER:
                        #         actions.append(moveSoldiers((x,y),(x,y-1),self.board[x,y,1]))

                        #     # try move down
                        #     elif self.board[x,y+1,0] == EMPTY_CELL and self.board[x+1,y,1] >= 50 and y > VCENTER:
                        #         actions.append(moveSoldiers((x,y),(x,y+1),self.board[x,y,1]))

                        # elif x <= 2:

                        #     if self.board[x+1,y,0] in [EMPTY_CELL, ALLIED_SOLDIER_RANGED]:
                        #         actions.append(moveSoldiers((x,y),(x+1,y),soldier_amount))







                            



        

        
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


