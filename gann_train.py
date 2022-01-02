#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
from functools import partial
from logging import critical
import os
import numpy as np
import pygad
import pygad.nn
import pygad.gann
import pickle
from js_logger import logger
import time
import signal
from datetime import datetime

import sys, subprocess, threading
from multiprocessing import Pool


from subprocess import TimeoutExpired

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
class SubprocessThread(threading.Thread):
  def __init__(self,
               args,
               stdin_pipe=subprocess.PIPE,
               stdout_pipe=subprocess.PIPE,
               stderr_pipe=subprocess.PIPE,
               stderr_prefix=None,
               timeout=None):
    threading.Thread.__init__(self)
    self.timeout = timeout
    self.stderr_prefix = stderr_prefix
    self.p = subprocess.Popen(args, stdin=stdin_pipe, stdout=stdout_pipe, stderr=stderr_pipe)

  def run(self):
    try:
      self.pipeToStdErr(self.p.stderr)
      self.return_code = self.p.wait(self.timeout)
      self.error_message = None
    except (SystemError, OSError, TimeoutExpired):
      self.return_code = -1
      self.error_message = "The process crashed or produced too much output."

  # Reads bytes from the stream and writes them to sys.stderr prepending lines
  # with self.stderr_prefix.
  # We are not reading by lines to guard against the case when EOL is never
  # found in the stream.
  def pipeToStdErr(self, stream):
    new_line = True
    while True and stream is not None:
      chunk = stream.readline(1024)

      if not chunk:
        return

      chunk = chunk.decode("UTF-8")

      if new_line and self.stderr_prefix:
        chunk = self.stderr_prefix + chunk
        new_line = False

      sys.stderr.write(chunk)

      if chunk.endswith("\n"):
        new_line = True

      sys.stderr.flush()


class PooledGA(pygad.GA):

    def __init__(self, num_generations, num_parents_mating, fitness_func, initial_population=None, sol_per_pop=None, num_genes=None, init_range_low=-4, init_range_high=4, gene_type=float, parent_selection_type="sss", keep_parents=-1, K_tournament=3, crossover_type="single_point", crossover_probability=None, mutation_type="random", mutation_probability=None, mutation_by_replacement=False, mutation_percent_genes='default', mutation_num_genes=None, random_mutation_min_val=-1, random_mutation_max_val=1, gene_space=None, allow_duplicate_genes=True, on_start=None, on_fitness=None, on_parents=None, on_crossover=None, on_mutation=None, callback_generation=None, on_generation=None, on_stop=None, delay_after_gen=0, save_best_solutions=False, save_solutions=False, suppress_warnings=False, stop_criteria=None, gann=None):
        
        super().__init__(num_generations, num_parents_mating, fitness_func, initial_population=initial_population, sol_per_pop=sol_per_pop, num_genes=num_genes, init_range_low=init_range_low, init_range_high=init_range_high, gene_type=gene_type, parent_selection_type=parent_selection_type, keep_parents=keep_parents, K_tournament=K_tournament, crossover_type=crossover_type, crossover_probability=crossover_probability, mutation_type=mutation_type, mutation_probability=mutation_probability, mutation_by_replacement=mutation_by_replacement, mutation_percent_genes=mutation_percent_genes, mutation_num_genes=mutation_num_genes, random_mutation_min_val=random_mutation_min_val, random_mutation_max_val=random_mutation_max_val, gene_space=gene_space, allow_duplicate_genes=allow_duplicate_genes, on_start=on_start, on_fitness=on_fitness, on_parents=on_parents, on_crossover=on_crossover, on_mutation=on_mutation, callback_generation=callback_generation, on_generation=on_generation, on_stop=on_stop, delay_after_gen=delay_after_gen, save_best_solutions=save_best_solutions, save_solutions=save_solutions, suppress_warnings=suppress_warnings, stop_criteria=stop_criteria)

        self.gann = gann
        self.on_generation = partial(PooledGA.callback, self=self)     

    @staticmethod
    def callback(ga_instance, self=None):
        
        population_matrices = pygad.gann.population_as_matrices(population_networks=self.gann.population_networks, population_vectors=ga_instance.population)
        
        self.gann.update_population_trained_weights(population_trained_weights=population_matrices)

        logger.info("Generation: {generation}".format(generation=ga_instance.generations_completed))
        logger.info(f'Best fitness: {np.max(ga_instance.last_generation_fitness)}')

        # Save current best
        solution, _, _ = self.best_solution()
        np.savez('current_best', solution)        

        if ga_instance.best_solution_generation != -1:
            logger.info("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

    @staticmethod
    def fitness_wrapper(idx, solution):
        return PooledGA.fitness_func(solution, idx)

    def cal_pop_fitness(self):        

        logger.debug('Populating pool')
        # with Pool(processes=self.sol_per_pop) as pool:
        with Pool() as pool:
            pop_fitness = pool.starmap(PooledGA.fitness_wrapper, list(enumerate(self.gann.population_networks)))  

        logger.debug('Pool finished')
        return np.array(pop_fitness)

    @staticmethod
    def fitness_func(solution, index):
        
        server_call = ['python3', 'gann_server.py', '-dif', '0', '-eval']        
        client_call = ['python3', 'gann_army.py']

        t_client = SubprocessThread(client_call, stderr_prefix="client debug: ", stderr_pipe=None, timeout=None)
        t_server = SubprocessThread(server_call, stdin_pipe=t_client.p.stdout, stdout_pipe=t_client.p.stdin, stderr_prefix="server debug: ", stderr_pipe=None, timeout=None)
        
        # open pipes
        client_pipe_path = '.pipes/' + str(t_client.p.pid) + '_client'
        logger.debug(f'Creating pipe {client_pipe_path}')
        if not os.path.exists(client_pipe_path):
            os.mkfifo(client_pipe_path)
            logger.debug('Created')

        server_pipe_path = '.pipes/' + str(t_server.p.pid) + '_server'
        logger.debug(f'Creating pipe {server_pipe_path}')
        if not os.path.exists(server_pipe_path):
            os.mkfifo(server_pipe_path)
            logger.debug('Created')
        
        # send solution
        nn = solution
        logger.debug(f'Opening pipe {client_pipe_path}')
        client_pipe = open(client_pipe_path, 'wb')

        logger.debug(f'Starting server ({t_server.p.pid})')
        t_server.start()
        logger.debug(f'Starting client ({t_client.p.pid})')
        t_client.start()

        logger.debug('Dumping neural network')
        pickle.dump(nn, client_pipe, pickle.HIGHEST_PROTOCOL)

        # open pipe again as read
        client_pipe.close()
        client_pipe = open(client_pipe_path, 'r')

        logger.debug(f'Opening pipe {server_pipe_path}')
        server_pipe = open(server_pipe_path, 'r')

        logger.debug(f'Joining client ({t_client.p.pid})')
        t_client.join()
        logger.debug(f'Joining server ({t_server.p.pid})')
        t_server.join()

        logger.debug('Reading score & retard')
        msg = server_pipe.read().split()
        score = int(msg[0])
        retard = float(msg[1])
        msg = client_pipe.read().split()
        valid_moves = int(msg[0])
        moves_count = int(msg[1])
        # invalid_moves = moves_count - valid_moves
        # logger.debug(f'Score {score} | Retard {retard} | Valid Moves {valid_moves} | Total Moves {moves_count}')

        # remove pipe files
        logger.debug('Closing & removing pipes')
        client_pipe.close()
        server_pipe.close()
        os.remove(client_pipe_path)
        os.remove(server_pipe_path)

        penalty = (0.85) if (t_client.return_code or t_server.return_code) else 1
        # reward = (retard / 20000) 
        # level = (score / 2000)

        # fitness = (valid_moves / mov0es_count) * penalty
        # fitness = ((level * 34) + (reward * 33) + (good_moves_ratio * 33)) * penalty
        fitness = score * penalty

        logger.debug(f'Fitness: {fitness}')
        
        return fitness

    def compute(self):

        logger.debug('Calling run')
        self.run()

        solution, solution_fitness, solution_idx = self.best_solution()
        logger.info("Parameters of the best solution : {solution}".format(solution=solution))
        logger.info("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        logger.info("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        name_append =f'{round(solution_fitness, 3)}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'

        self.save(filename=f'outputs/genetic_{name_append}')

        filename = f'outputs/results_{name_append}' 
        np.savetxt(filename + '.txt', solution, delimiter=',')
        np.savez(filename, solution)

        self.plot_fitness(save_dir=f'outputs/graph_{name_append}.png')

"""
FUNCTIONS DEFINITIONS
"""

"""
MAIN
"""
if __name__ == '__main__':

    gann = pygad.gann.GANN(num_solutions=100,
                        num_neurons_input=26,
                        num_neurons_output=17,
                        num_neurons_hidden_layers=[20, 20],
                        hidden_activations="relu",
                        output_activation="softmax")
    
    logger.info('GANN created')

    population_vectors = pygad.gann.population_as_vectors(population_networks=gann.population_networks)

    initial_population = population_vectors.copy()

    trainer = PooledGA(num_generations=1000,
                        num_parents_mating=50,
                        initial_population=initial_population,
                        fitness_func=PooledGA.fitness_func,
                        mutation_percent_genes=25,
                        # mutation_probability=0.4,
                        init_range_low=-5,
                        init_range_high=5,
                        parent_selection_type='sus',
                        crossover_type='uniform',
                        mutation_type='random',
                        # keep_parents=1,
                        allow_duplicate_genes=False,
                        save_best_solutions=False,
                        stop_criteria=["reach_2000"],
                        gann=gann)

    logger.info('PooledGA created')

    logger.info('Starting')
    trainer.compute()
    logger.info('Done')


