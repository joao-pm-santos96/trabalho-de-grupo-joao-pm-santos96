#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import os
import numpy as np
import pygad
import pygad.nn
import pygad.gann
import pickle

import sys, subprocess, threading

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
               stderr_prefix=None):
    threading.Thread.__init__(self)
    self.stderr_prefix = stderr_prefix
    self.p = subprocess.Popen(
        args, stdin=stdin_pipe, stdout=stdout_pipe, stderr=subprocess.PIPE)

  def run(self):
    try:
      self.pipeToStdErr(self.p.stderr)
      self.return_code = self.p.wait()
      self.error_message = None
    except (SystemError, OSError):
      self.return_code = -1
      self.error_message = "The process crashed or produced too much output."

  # Reads bytes from the stream and writes them to sys.stderr prepending lines
  # with self.stderr_prefix.
  # We are not reading by lines to guard against the case when EOL is never
  # found in the stream.
  def pipeToStdErr(self, stream):
    new_line = True
    while True:
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

class SoldierTraining:

    def __init__(self):
        self.gann_instance = None
        self.ga_instance = None

    @staticmethod
    def fitness_func(solution, sol_idx):
        global cenas

        server_call = ['python3', 'gann_server.py', '-dif', '0', '-eval']        
        client_call = ['python3', 'gann_army.py']

        t_client = SubprocessThread(client_call, stderr_prefix="client debug: ")
        t_server = SubprocessThread(
            server_call,
            stdin_pipe=t_client.p.stdout,
            stdout_pipe=t_client.p.stdin,
            stderr_prefix="server debug: ")

        # open nn pipe
        name = '.pipes/' + str(t_client.p.pid) + '_nn'
        os.mkfifo(name)
        
        nn = cenas.population_networks[sol_idx]    
        with open(name, 'wb') as pipe:
            pickle.dump(nn, pipe, pickle.HIGHEST_PROTOCOL)


        t_client.start()
        t_server.start()

        t_client.join()
        t_server.join()

        # open score pipe
        name = '.pipes/' + str(t_server.p.pid) + '_score'
        with open(name, 'r') as pipe:
            score = int(pipe.readline())

        penalty = 2000 if (t_client.return_code or t_server.return_code) else 0
        
        return score - penalty #TODO consider retard

    @staticmethod
    def on_generation(ga_instance):
        global cenas

        population_matrices = pygad.gann.population_as_matrices(population_networks=cenas.population_networks, population_vectors=ga_instance.population)
        cenas.update_population_trained_weights(population_trained_weights=population_matrices)

        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    def run(self):
        global cenas
        cenas = pygad.gann.GANN(num_solutions=6,
                                num_neurons_input=8,
                                num_neurons_output=5,
                                num_neurons_hidden_layers=[15, 10, 7],
                                hidden_activations="relu",
                                output_activation="softmax")

        population_vectors = pygad.gann.population_as_vectors(population_networks=cenas.population_networks)

        initial_population = population_vectors.copy()

        self.ga_instance = pygad.GA(num_generations=500,
                       num_parents_mating=4,
                       initial_population=initial_population,
                       fitness_func=SoldierTraining.fitness_func,
                       mutation_percent_genes=5,
                       init_range_low=-2,
                       init_range_high=5,
                       parent_selection_type='sss',
                       crossover_type='single_point',
                       mutation_type='random',
                       keep_parents=1,
                       on_generation=SoldierTraining.on_generation)


        self.ga_instance.run()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        self.ga_instance.plot_fitness()



    
cenas = None # TODO

"""
FUNCTIONS DEFINITIONS
"""
def main():
    train = SoldierTraining()
    train.run()

"""
MAIN
"""
if __name__ == '__main__':
    main()