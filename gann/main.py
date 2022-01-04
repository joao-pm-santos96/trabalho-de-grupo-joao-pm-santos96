#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""

from datetime import datetime
import os
from multiprocessing import Process, Queue
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import random
import time
import pygad
import pygad.nn
import pygad.gann
import pickle
import sys
import json
import numpy as np
from itertools import combinations
import select
import signal
import traceback

from numpy.core.numeric import moveaxis
from actions import *
from utils import *
from utils import _PRINT
from viewer import Viewer

from js_logger import logger

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
"""

"""
CONSTANTS
"""
MAX_RETARD = 20000
ACTION_MAP = [UpgradeBase, RecruitSoldiers, MoveSoldiers]
ACTION_LABEL = ["Upgrade building", "Recruit soldiers", "Move soldiers"]
CELL_PIXELS = 48

"""
CLASS DEFINITIONS
"""
class Army:
    def __init__(self, difficulty, base_cost, base_prod, neural_net=None, in_q=None, out_q=None):
        self.difficulty = difficulty
        self.resources = 0
        self.building_level = 0
        self.base_cost = base_cost
        self.base_prod = base_prod
        self.board = [[None]*WIDTH for h in range(HEIGHT)]

        """
        JS STUFF
        """
        self.debug_file = "client_debug.txt"
        self.neural_net = neural_net
        self.in_q = in_q
        self.out_q = out_q

        self.motions={'U': [0,-1], 
                    'D': [0,1],
                    'L': [-1,0],
                    'R':[1,0]}

        base_moves = ['U', 'D', 'L', 'R']
        self.outputs = base_moves
        base_moves = [None, 'U', 'D', 'L', 'R']
        self.outputs.extend(list(combinations(base_moves,2)))
        self.outputs.append('upgrade')
        self.outputs.append('recruit_melee')
        self.outputs.append('recruit_ranged')
        # TODO outputs for position and amount? same turn recruit both types?

        self.playActions([])

    @property
    def upgrade_cost(self):
        return int(self.base_cost*(1.4**self.building_level))

    @property
    def production(self):
        return int(self.base_prod*(1.2**self.building_level))

    def readEnvironment(self):
        # state = input()
        state = self.in_q.get()
        
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
        
        soldiers = self.board[:,:,0]
        troops = np.argwhere((soldiers==ALLIED_SOLDIER_RANGED) | (soldiers==ALLIED_SOLDIER_MELEE) | (soldiers==ALLIED_MAIN_BUILDING))

        # enemies = np.argwhere((soldiers == ENEMY_SOLDIER_MELEE) | (soldiers == ENEMY_SOLDIER_RANGED))
        # enemies = [tuple(x) for x in enemies]

        for x,y in troops:
        #     enemy = Environment.findEnemy((x,y), enemies)

            data = [self.difficulty, # difficulty
                    self.resources, # resources
                    self.upgrade_cost, # upgrade cost
                    self.production, # current production
                    x, # soldier x
                    y, # soldier y
            ]

            for dx in range(-3,4):
                for dy in range(-3,4):
                    if abs(dx) + abs(dy) < 4:
                        a = x + dx
                        b = y + dy
                        condition = (0 < a < WIDTH-1 and 0 < b < HEIGHT-1)

                        data.append(int((self.board[a,b,0] if condition else WALL) or -1)) # type in cell (substitute None by -1)
                        data.append(int(self.board[a,b,1] if condition else 0)) # amount in cell 

            prediction = pygad.nn.predict(last_layer=self.neural_net, data_inputs=np.array([data]))

            move = self.outputs[prediction[0]]

            if move is not None:
                if move == 'upgrade':
                    actions.append(upgradeBase())
                    self.resources -= self.upgrade_cost
                    
                elif 'recruit' in move:
                    cost = SOLDIER_MELEE_COST if 'melee' in move else SOLDIER_RANGED_COST
                    type = ALLIED_SOLDIER_MELEE if 'melee' in move else ALLIED_SOLDIER_RANGED

                    amount = self.resources // cost

                    actions.append(recruitSoldiers(type, amount))
                    self.resources -= amount * cost
                    
                else:
                    for dir in move:
                        if dir is not None:
                            dest = np.add([x,y], self.motions[dir]).astype(int)
                            amount = int(self.board[x,y,1] // len(move))

                            actions.append(moveSoldiers((x,y), dest, amount))
     
        self.playActions(actions)

    def playActions(self, actions):
        self.out_q.put(';'.join(map(str,actions)))
        

    def main(self):
        open(self.debug_file, 'w').close()

        # difficulty, base_cost, base_prod = map(int,input().split())

        # env = Environment(difficulty, base_cost, base_prod, neural_net=neural_net)
        return_code = 0
        while 1:
            signal = self.readEnvironment()

            if signal=="END":
                # debug("GAME OVER")
                logger.debug('GAME OVER')
                return_code = 0
                break
                # sys.exit(0)
            elif signal=="ERROR":
                # debug("ERROR")
                logger.debug('ERROR')
                return_code = 1
                break
                # sys.exit(1)
            
            self.play()

        self.out_q.put({'return_code': return_code})

class Server:
    def __init__(self, difficulty, viewer, in_q = None, out_q = None):
        self.difficulty = difficulty
        self.base_cost = BASE_COST[difficulty]
        self.base_prod = BASE_PRODUCTION[difficulty]
        self.viewer = viewer
        self.building_level = 0
        self.retard = 0
        self.actions_taken = 0

        # self.turn = 0
        self.turn = 1

        self.resources = 500
        self.board = [[[None, 0] for w in range(WIDTH)] for h in range(HEIGHT)]

        self.board[HEIGHT//2][0] = [ALLIED_MAIN_BUILDING, self.building_level]

        for i in range(HEIGHT):
            self.board[i][4] = [ALLIED_SOLDIER_RANGED, 5]
            self.board[i][5] = [ALLIED_SOLDIER_MELEE, 10]

        self.debug_file = "server_debug.txt"
        self.in_q = in_q
        self.out_q = out_q

    def end(self):
        return self.board[HEIGHT//2][0][0] != ALLIED_MAIN_BUILDING or self.turn>=MAX_T or self.retard>=MAX_RETARD

    def validatePurchases(self, actions):
        leftactions = []
        for action in actions:
            if isinstance(action, UpgradeBase):
                cost = int(self.base_cost*(1.4**self.building_level))
                if self.resources < cost:
                    return None, INVALID_ACTION(f"Cannot upgrade building, current cost is {cost} and you only have {self.resources} available")
                self.resources -= cost
                self.building_level += 1
                self.board[HEIGHT//2][0] = [ALLIED_MAIN_BUILDING, self.building_level]

            elif isinstance(action, RecruitSoldiers):
                cost = action.getPrice()
                lx,ly = action.location
                if self.resources < cost:
                    return None, INVALID_ACTION(f"Cannot recuit soldiers, current cost is {cost} and you only have {self.resources} available")
                cell = self.board[ly][lx] 
                if cell[0] and cell[0]!=action.type:
                    return None, INVALID_ACTION(f"Cannot recruit soldiers of different type in the same location, move your soldiers first")
                self.resources -= cost
                if cell[0]:
                    self.board[ly][lx][1] += action.amount
                else:
                    self.board[ly][lx] = [action.type, action.amount]
            else:
                leftactions.append(action)
            
        return leftactions, None

    @property
    def upgrade_cost(self):
        return int(self.base_cost*(1.4**self.building_level))


    @property
    def production(self):
        return int(self.base_prod*(1.2**self.building_level))

            
    def applyActions(self, actions):
        actions = sorted(actions)
        for action in actions:
            if isinstance(action, UpgradeBase):
                cost = int(self.base_cost*(1.4**self.building_level))
                if self.resources < cost:
                    return INVALID_ACTION(f"Cannot upgrade building, current cost is {cost} and you only have {self.resources} available")
                self.resources -= cost
                self.building_level += 1
                self.board[HEIGHT//2][0] = [ALLIED_MAIN_BUILDING, self.building_level]
           
        
        self.enemyPlay()

        self.resources += int(self.base_prod*(1.2**self.building_level))
    
    def get_state_dict(self): 
        return {
            'turn': self.turn,
            'retard': self.retard, 
            'resources': self.resources,
            'production': self.production,
            'upgrade_cost': self.upgrade_cost,
            'actions_taken': self.actions_taken
        }
    

    def validateAndApplyMovements(self, actions):
        disabledRanged = [[0]*WIDTH for h in range(HEIGHT)] 
        
        visited = []

        influence = {} 

        alied_soldiers = [[None]*WIDTH for h in range(HEIGHT)]
        
        for action in actions:
            if not isinstance(action, MoveSoldiers):
                return INVALID_ACTION(action.actionID)         

            frx, fry = action.fr

            cell = self.board[fry][frx]
            if cell[0] not in alied_s:
                return INVALID_ACTION(f"Cannot move troops in {action.fr}")

            if cell[1] < action.amount:
                return INVALID_ACTION(f"Cannot move {action.amount} soldiers when you only have {cell[1]} in coord {action.fr}")

            if action.fr not in influence:
                influence[action.fr] = []

            if action.to not in influence:
                influence[action.to] = []
            
            influence[action.fr].append( (cell[0], -action.amount) )
            influence[action.to].append( (cell[0], action.amount) )
            
            if cell[0] == ALLIED_SOLDIER_RANGED:
                disabledRanged[fry][frx] += action.amount

            visited.append((frx, fry))

        for row in range(HEIGHT):
            for col in range(WIDTH):                   
                
                cell = self.board[row][col]
                if (col,row) not in influence: 
                    if cell[0] in alied_s:
                        alied_soldiers[row][col] = self.board[row][col][:]
                    continue #not affected by movement

                moveaway = [entry for entry in influence[(col,row)] if entry[1]<0]
                moveamount = -sum(entry[1] for entry in moveaway)
                currentamount = cell[1] if cell[0] in alied_s else 0
                #debug("moveamount",moveamount)
                if moveamount > currentamount: 
                    # not enought troops
                    return INVALID_ACTION(f"Cannot move {moveamount} soldiers when you only have {currentamount} in coord {(col, row)}")

                remaining = currentamount - moveamount 

                movein = [entry for entry in influence[(col,row)] if  entry[1]>0]
                #debug("movein: ", movein)
                if not movein:
                    alied_soldiers[row][col] = [cell[0], remaining] if remaining else None
                    continue
                else:
                    melees = sum([entry[1] for entry in movein if entry[0]==ALLIED_SOLDIER_MELEE]) 
                    rangeds = sum([entry[1] for entry in movein if entry[0]==ALLIED_SOLDIER_RANGED]) 

                    if remaining and ((melees and cell[0]==ALLIED_SOLDIER_RANGED) or (rangeds and cell[0]==ALLIED_SOLDIER_MELEE)):
                        return INVALID_ACTION(f"Cannot mix ranged soldiers with melee soldiers in same cell {(col, row)}, move then first or swap them in the same turn")

                    # TODO add possibility to suicide/clean melee soldiers and position ranged squad in the same turn
                    if melees and rangeds:
                        return INVALID_ACTION(f"Cannot merge ranged soldiers with melee soldiers in the same cell {(col, row)}")

                    if remaining:
                        alied_soldiers[row][col] = [cell[0], remaining + melees + rangeds]  
                    else:
                        alied_soldiers[row][col] = [ALLIED_SOLDIER_MELEE if melees else ALLIED_SOLDIER_RANGED, melees + rangeds] if melees + rangeds else None

        #first apply damage from range units
        areadamage = [(3,0),(2,1),(1,2),(0,3),(-1,2),(-2,1),(-3,0),(-2,-1),(-1,-2),(0,-3),(1,-2),(2,-1)]
        for row in range(HEIGHT):
            for col in range(WIDTH):
                cell = self.board[row][col]
                if cell[0] == ALLIED_SOLDIER_RANGED and cell[1]>disabledRanged[row][col]:
                    damage = min(500, cell[1]-disabledRanged[row][col]) #max 500 damage
                    if not damage: continue
                    if damage < 0:
                        return INVALID_ACTION(f"Negative damage on ranged units error!!")
                    for area in areadamage:
                        mx, my = area
                        nx = col + mx
                        ny = row + my 
                        if not (0<=nx<WIDTH and 0<=ny<HEIGHT): continue
                        targetcell = self.board[ny][nx]
                        if targetcell[0] in [ENEMY_SOLDIER_MELEE,ENEMY_SOLDIER_RANGED]:
                            targetcell[1] = max(0, targetcell[1]-damage)
                            if targetcell[1]==0: # kill all troops
                                self.board[ny][nx] = [None, 0]


        for row in range(HEIGHT):
            for col in range(WIDTH):
                cell = self.board[row][col]
                if alied_soldiers[row][col]:
                    if cell[0] in enemy_s:
                        self.board[row][col] = duelResult(alied_soldiers[row][col], cell)
                    else:
                        self.board[row][col] = alied_soldiers[row][col][:]
                
                elif cell[0] in alied_s:
                    self.board[row][col] = [None, 0]


        # debug("State after your actions:")
        # debug("\n".join([', '.join([gridstr(cell) for cell in aliedrow]) for aliedrow in self.board]))


        return None


    def enemyEngage(self, troops):
        for row in range(HEIGHT):
            for col in range(WIDTH):
                cell = self.board[row][col]
                if troops[row][col] and cell[0] in [ALLIED_MAIN_BUILDING, ALLIED_SOLDIER_MELEE, ALLIED_SOLDIER_RANGED]:
                    self.board[row][col] = duelResult(cell, [ENEMY_SOLDIER_MELEE, troops[row][col]])
                    troops[row][col] = 0
                

        
    def enemyMovement(self):
        for row in range(HEIGHT):
            cell = self.board[row][WIDTH-1]
            if cell[0] in alied_s:
                self.retard =  min(MAX_RETARD, self.retard+cell[1]/5.0)
                self.resources += cell[1]*SOLDIERS_COST[cell[0]]*2
                self.board[row][WIDTH-1] = [None, 0] 
        
        target = [0]*HEIGHT*WIDTH
        for row in range(HEIGHT):
            for col in range(WIDTH):
                cell = self.board[row][col]
                if cell[0] in [ALLIED_MAIN_BUILDING, ALLIED_SOLDIER_RANGED] or cell[0]==ALLIED_SOLDIER_MELEE and cell[1]>20:
                    target[row*WIDTH+col] = 1


        ntroops = [[0]*WIDTH for h in range(HEIGHT)]
        
        top_mask = (1<<WIDTH)-1
        bot_mask = top_mask<<((HEIGHT-1)*WIDTH)
        left_mask = 1
        right_mask = 1<<(WIDTH-1)
        for row in range(HEIGHT):
            left_mask |= left_mask<<WIDTH
            right_mask |= right_mask<<WIDTH
        

        moves = [1, -WIDTH, -1, WIDTH]
        masks = [right_mask, top_mask, left_mask, bot_mask]

        nodes=[-1]*WIDTH*HEIGHT

        for row in range(HEIGHT):
            for col in range(WIDTH):
                origincell = self.board[row][col] 
                if origincell[0] != ENEMY_SOLDIER_MELEE: continue
                visited = 1<<(row*WIDTH+col) 
                nodes[0] = row*WIDTH+col
                left = 0
                right = 1
                search = 1 
                while search:
                    node = nodes[left]
                    left += 1
                    b = 1<<node
                    for actionidx in range(4):
                        if (b & masks[actionidx]): continue
                        nextnode = node + moves[actionidx]
                        nb = 1 << nextnode
                        if (nb & visited): continue
                        
                        if target[nextnode]:
                            tx,ty = nextnode%WIDTH, nextnode//WIDTH
                            if abs(tx-col)>abs(ty-row): # prioritize y
                                ny = row
                                nx = col + (1 if tx>col else -1)
                            else:
                                nx = col
                                ny = row + (1 if ty>row else -1)
                            search = 0
                            break   
                        nodes[right] = nextnode
                        right+=1
                        visited |= nb

                
                ntroops[ny][nx] += origincell[1]
       
        
        # clear previous enemy troops
        for row in range(HEIGHT):
            for col in range(WIDTH):
                cell = self.board[row][col]
                if cell[0] in [ENEMY_SOLDIER_MELEE]:
                    self.board[row][col] = [None, 0]

        self.enemyEngage(ntroops)
        
        # reposition non-duel troops
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if ntroops[row][col]:
                    if self.board[row][col][0]:
                        # debug("ERROR!!")
                        pass
                    self.board[row][col] = [ENEMY_SOLDIER_MELEE, ntroops[row][col]]

                

    def enemySpawn(self):
  
        spawnsoldiers = 2 + int( ( max(self.turn - self.retard, self.turn/3) **2)/65)
        if self.difficulty==0:
            cell = random.randint(0,HEIGHT-1)
        else:
            cell = self.turn%HEIGHT
        self.board[cell][WIDTH-1] = [ENEMY_SOLDIER_MELEE, spawnsoldiers]

        # debug("State after enemy actions:")

        # debug("\n".join([', '.join([gridstr(cell) for cell in aliedrow]) for aliedrow in self.board]))


    def setSoldier(self, soldiers):
        pass

    def outputState(self):
        
        self.output(f"{self.building_level} {self.resources} {json.dumps(self.board, separators=(',', ':'))}")
        
    # cycle
    def readAndApplyTurnEvents(self):
        repeat = True
        while repeat:
            repeat = False
            signal.signal(signal.SIGALRM, raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, MAX_PROCESS_TIME/1000.0)
            
            try:
                actions, error = self.readActions() # 1
                signal.setitimer(signal.ITIMER_REAL, 0)
                if error: return error
                actions, error = self.validatePurchases(actions) # 2 
                if error: return error
                error = self.validateAndApplyMovements(actions) # 3
                if error: return error
                if self.viewer:
                    self.viewer.drawmap(self.board, self.get_state_dict()) # 5
                    
            except TimeoutError:
                # debug(f"ACTIONS were not read in time ({MAX_PROCESS_TIME}ms)!")
                repeat = True

            # enemy play
            self.enemyMovement() # 6
            self.enemySpawn() # 7
            self.resources += int(self.base_prod*(1.2**self.building_level)) # 8
            if self.viewer:
                self.viewer.drawmap(self.board, self.get_state_dict()) # 9
                
            self.turn += 1

    def readActions(self): # error handling, syntax handling (1)
        try:
            # actions = input()
            # if actions=='':
            #     # debug("No actions were taken!")
            #     return [], None

            actions=self.in_q.get()
            
            if actions=='':
                # debug("No actions were taken!")
                return [], None

            if " " in actions:
                return None, COMMANDS_CANT_CONTAIN_SPACES(actions)
            actions = actions.split(";")
            if len(actions)>MAX_ACTIONS:
                return None, NUM_ACTIONS_EXCEEDED(len(actions), MAX_ACTIONS)

            parsed_actions = []
            
            for action in actions:
                actionId, *rest = action.split('|')           

                # debug("Action received: ", ACTION_LABEL[int(actionId)], ' '.join(rest))
                nextaction = ACTION_MAP[int(actionId)](rest)
                if nextaction.error:
                    return None, nextaction.error
                parsed_actions.append(nextaction)
            self.actions_taken += len(parsed_actions)
            return parsed_actions, None 

        except TimeoutError:
            raise TimeoutError
        except:
            traceback.print_exc(file=sys.stderr)
            return None, INVALID_LINE_ERROR

    def output(self, *line):
        # _PRINT(*line)
        self.out_q.put(*line)
        # sys.stdout.flush()
        # if line=="END":
        #     sys.exit(0)
        # elif line=="ERROR":
        #     sys.exit(1)

    def main(self):
        
        open(self.debug_file, 'w').close()

        random.seed(int(time.time()*1000))

        # debug("Difficulty is", level)
        
        # env = Environment(level, viewer)    
        # Output(f"{self.difficulty} {self.base_cost} {self.base_prod}")
        ok = self.in_q.get()
        score = 0
        return_code = 0
        while 1:
            self.outputState()
            error = self.readAndApplyTurnEvents()

            score = self.turn if self.retard<MAX_RETARD else int(MAX_T*1.5-self.turn/2)

            # debug("SCORE: ", score, ", retard: ", env.retard)
            if self.end():
                # debug("END!")
                self.output("END")
                logger.debug('END')
                return_code = 0
                break
            elif error:
                # debug("ERROR:",error)
                self.output("ERROR")
                logger.debug(f'ERROR: {error}')
                return_code = 1
                break
       
        self.out_q.put({'score': score, 'retard': self.retard, 'return_code': return_code})

"""
FUNCTIONS DEFINITIONS
"""

def raise_timeout(signum, frame):
    raise TimeoutError

# def debug(*args):
#     _PRINT(*args, file=sys.stderr, flush=True)
#     with open(DEBUG_FILE, 'a') as f:
#         stdout, sys.stdout = sys.stdout, f # Save a reference to the original standard output
#         _PRINT(*args)
#         sys.stdout = stdout
# print = debug # dont erase this line, otherwise you cannot use the print function for debug

def upgradeBase():
    return f"{UPGRADE_BUILDING}"

def recruitSoldiers(type, amount, location=(1,VCENTER)):
    return f"{RECRUIT_SOLDIERS}|{type}|{amount}|{location[0]}|{location[1]}".replace(" ","")

def moveSoldiers(pos, to, amount):
    return f"{MOVE_SOLDIERS}|{pos[0]}|{pos[1]}|{to[0]}|{to[1]}|{amount}".replace(" ","")

# def playActions(actions):
#     _PRINT(';'.join(map(str,actions)), flush=True)

def fitness_func(solution, index):
    global gann

    server_in = Queue()
    server_out = Queue()
    nn = gann.population_networks[index]
    
    difficulty = 1
    eval = True

    viewer = None if eval else Viewer()

    server = Server(difficulty, viewer, in_q=server_in, out_q=server_out)
    army = Army(difficulty, BASE_COST[difficulty], BASE_PRODUCTION[difficulty], neural_net=nn, in_q=server_out, out_q=server_in)

    server_t = Process(target=server.main)
    army_t = Process(target=army.main)

    server_t.start()
    army_t.start()

    server_t.join()
    army_t.join()

    server_return = server_out.get()
    army_return = server_in.get()

    penalty = (0.5) if (server_return['return_code'] or army_return['return_code']) else 1
    fitness = server_return['score'] * penalty

    return fitness

def on_generation(ga_instance):
    global ga, gann

    population_matrices = pygad.gann.population_as_matrices(population_networks=gann.population_networks, population_vectors=ga_instance.population)
        
    gann.update_population_trained_weights(population_trained_weights=population_matrices)

    logger.info("Generation: {generation}".format(generation=ga_instance.generations_completed))
    logger.info(f'Mean fitness: {np.mean(ga_instance.last_generation_fitness)}')
    logger.info(f'Best fitness: {np.max(ga_instance.last_generation_fitness)}')

    # Save current best
    solution, _, _ = ga.best_solution()
    np.savez('current_best', solution)  
    logger.info('Saved current_best')      

    if ga_instance.best_solution_generation != -1:
        logger.info("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

def main():
    global ga, gann

    gann = pygad.gann.GANN(num_solutions=100,
                        num_neurons_input=56,
                        num_neurons_output=17,
                        # num_neurons_hidden_layers=[56, 35, 35, 17],
                        # num_neurons_hidden_layers=[45],
                        num_neurons_hidden_layers=[48, 40, 32, 24],
                        hidden_activations="relu",
                        output_activation="softmax")

    logger.info('GANN created')

    population_vectors = pygad.gann.population_as_vectors(population_networks=gann.population_networks)

    initial_population = population_vectors.copy()
    logger.info(f'# Weights = {len(initial_population[0])}')

    ga = pygad.GA(num_generations=10000,
                        num_parents_mating=50,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=on_generation,
                        mutation_percent_genes=5,
                        init_range_low=-5,
                        init_range_high=5,
                        parent_selection_type='sus',
                        crossover_type='single_point',
                        mutation_type='random',
                        allow_duplicate_genes=False,
                        save_best_solutions=False,
                        stop_criteria=["reach_500"])

    logger.info('PooledGA created')
    logger.info('Starting')

    ga.run()

    solution, solution_fitness, solution_idx = ga.best_solution()
    logger.info("Parameters of the best solution : {solution}".format(solution=solution))
    logger.info("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    logger.info("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    name_append =f'{round(solution_fitness, 3)}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'

    ga.save(filename=f'outputs/genetic_{name_append}')

    filename = f'outputs/results_{name_append}' 
    np.savetxt(filename + '.txt', solution, delimiter=',')
    np.savez(filename, solution)

    ga.plot_fitness(save_dir=f'outputs/graph_{name_append}.png')

"""
MAIN
"""
ga = None
gann = None

if __name__ == '__main__':
    main()
