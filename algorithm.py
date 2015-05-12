import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import time
from abc import ABCMeta
import sys

def logOK(text):
	print time.strftime('%H:%M:%S') + ": " + '\033[92m' + text + '\033[0m'

def logERROR(text):
	print time.strftime('%H:%M:%S') + ": " +  '\033[91m' + text + '\033[0m'

def log(text):
	print time.strftime('%H:%M:%S') + ": " +  text

class EA(object):
	__metaclass__ = ABCMeta

	def __init__(self):
		self.rng = RandomStreams()

		self.entity_mutate_rate = 0.1
		self.bit_mutate_rate = 0.05
		self.crossover_rate = 0.7
	
	def initialize_random_population(self):
		return np.random.randint(2, size = (1000, 1000)).astype(theano.config.floatX) # 1000 entities, 1000 bits

class SimpleEA(EA):
	def __init__(self, fitnessFunction, selectionFunction):
		"""
		Initialize a simple EA algorithm class.
		Creating the SimpleEA object compiles the needed Theano functions.
		
		:param fitnessFunction: The fitness function that takes a matrix of entities and and outputs a vector of fitness values for the ids given in `changes`.
		:param selectionFunction: The selection function that takes a matrix of entities and a vector of fitnesses and outputs a matrix of entities.
		"""
		super(SimpleEA, self).__init__()

		self.fitness = fitnessFunction
		self.selection = selectionFunction

	def cross(self, entities):
		"""
		TODO:
		* Really slow, optimization needed.
		"""
		def single_crossover(i, r_i, xpoint, entities):
			if T.gt(r_i, 0):
				# Parents
				e1 = entities[i]
				e2 = entities[i-1]

				# Children
				e3 = T.concatenate([e1[:xpoint], e2[xpoint:]])
				e4 = T.concatenate([e2[:xpoint], e1[xpoint:]])

				# Replace parents with children
				new_e = T.stack(e3, e4)
				return {entities: T.set_subtensor(entities[i-1:i+1], new_e)}
			else:
				return {}

		entity_pair_range = T.arange(1, entities.shape[0], 2)

		# Generate crossover bools once to save calls, use as sequence
		r = self.rng.choice(size = (entities.shape[0] / 2,), p = [1-self.crossover_rate, self.crossover_rate])

		# Generate crossover points once
		xpoints = self.rng.random_integers(size = (entities.shape[0] / 2,), low = 0, high = entities.shape[1]-1)
	
		values, updates = theano.scan(fn=single_crossover, sequences=[entity_pair_range, r, xpoints], non_sequences=[entities], name="crossover", profile=theano.config.profile)

		return updates[entities]

	def mutation(self, entities):
		"""
		TODO:
		* Results not as good as fast_mutation, while this function is more intuitive, why?
		* Slow, need to optimize.
		"""
		def single_mutate(i, entities):
			e = entities[i]
			r = self.rng.choice(size = (1,), p = [1-self.entity_mutate_rate, self.entity_mutate_rate])

			# Mutate entity with certain probability
			if T.gt(r, 0):
				# Generate random bits
				rb = self.rng.choice(size = (entities.shape[1],), p = [1-self.bit_mutate_rate, self.bit_mutate_rate])

				# Flip random bits
				e = T.set_subtensor(e[rb.nonzero()], (e[rb.nonzero()] + 1) % 2)

			return e

		entities, _ = theano.map(fn = single_mutate, sequences=[T.arange(entities.shape[0])], non_sequences=[entities])
		return entities

	def fast_mutation(self, entities):
		"""
		Randomly flips bits in the population matrix.

		TODO:
		* Contains 1 HostFromGpu
		* Contains 1 GpuFromHost
		"""
		p = self.bit_mutate_rate * self.entity_mutate_rate
		r = self.rng.choice(size = entities.shape, p = [1-p, p])

		non_zero_indices = r.nonzero()

		to_be_changed = entities[non_zero_indices]
		changed_to = (to_be_changed + 1) % 2

		entities = T.set_subtensor(to_be_changed, changed_to)

		return entities
		
	def run(self, generations = 50):
		log("Compiling...")
		
		E = self.initialize_random_population()
		F = np.zeros((E.shape[0]), dtype=theano.config.floatX)

		# Change these to shared variables
		E = theano.shared(E)
		F = theano.shared(F)

		# Create graphs
		fitness_t = self.fitness(E)
		select_t = self.selection(E, F, self.rng)
		mutate_t = self.fast_mutation(E)
		crossover_t = self.cross(E)

		# Compile functions
		fitness = theano.function([], [], updates = [(F, fitness_t)])
		select = theano.function([], [], updates = [(E, select_t)])
		mutate = theano.function([], [], updates = [(E, mutate_t)])
		crossover = theano.function([], [], updates = [(E, crossover_t)])

		logOK("Compilation successfully completed.")
		log("Running algorithm...")

		start = time.time()

		fitness()
		
		for i in range(generations):
			select()
			crossover()
			mutate()
			
			fitness()

			if theano.config.profile:
				log("Fitness: " + str(np.max(F.get_value())))

			# Early stopping
			if np.max(F.get_value()) == 100:
				break

		end = time.time()

		if theano.config.profile:
			theano.printing.pydotprint(fitness, outfile="fitness.png")
			theano.printing.pydotprint(select, outfile="select.png")
			theano.printing.pydotprint(mutate, outfile="mutate.png")
			theano.printing.pydotprint(crossover, outfile="cross.png", scan_graphs = True)

		return E.eval(), start, end, i+1
		
if __name__ == "__main__":

	def fit(entities):
		return T.sum(entities, axis=1)

	def sel(entities, fitness, rng):
		"""
		TODO:
		* Contains 1 HostFromGpu (argmax)
		"""
		# Create random integers
		r = rng.random_integers(size = (entities.shape[0]*2,), low = 0, high = entities.shape[0]-1)

		# Take randomly matched entities' fitnesses and reshape
		chosen_f = fitness[r].reshape((2, entities.shape[0]))

		# Take winner IDs (binary)
		winner_ids = T.argmax(chosen_f, axis=0)

		# Compute entity IDs
		entity_ids = r[T.arange(entities.shape[0])+(winner_ids*entities.shape[0])]

		return entities[entity_ids]

	ea = SimpleEA(fit, sel)

	if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
		times = 20
	else:
		times = 1

	total_time = 0
	total_iterations = 0

	for i in range(times):
		entities, start, end, iterations = ea.run()
		total_time += end-start
		total_iterations += iterations

	avg_time = total_time / float(times)
	avg_iterations = total_iterations / float(times)

	f = np.max(np.sum(entities, axis=1))

	logOK("Done.")
	log("Device: %s" % theano.config.device)
	log("Max fitness: %i" % f)
	log("Time taken: %.5f seconds" % avg_time)
	log("Iterations taken: %.1f" % avg_iterations)
