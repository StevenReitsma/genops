import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import time
from abc import ABCMeta, abstractmethod

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

		self.mutate_rate = 0
		self.crossover_rate = 0

	def crossover(self, e1, e2):
		# Generate random crossover point
		xpoint = self.rng.random_integers(size = (1,), low = 0, high = e1.shape[0]-1)
		s1 = T.concatenate([T.ones(xpoint), T.zeros(e1.shape[0]-xpoint)])
		s2 = T.concatenate([T.zeros(xpoint), T.ones(e1.shape[0]-xpoint)])

		return (T.concatenate([e1[s1.nonzero()], e2[s2.nonzero()]]),
		        T.concatenate([e2[s1.nonzero()], e1[s2.nonzero()]]))

	def mutate(self, e):
		# Generate random bits
		r = self.rng.choice(size = (e.shape[0],), p = self.mutate_rate)

		# Flip random bits
		e[r.nonzero()] = (e[r.nonzero()] + 1) % 2
		return e
	
	def initialize_random_population(self):
		return np.random.randint(2, size = (1000, 100)).astype(theano.config.floatX) # 1000 entities, 1000 bits

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

		self.crossover_rate = 1
		self.mutate_rate = 0.9

	def _vary(self, entities):
		r = self.rng.uniform((entities.shape[0]/2,)) < self.crossover_rate
		
		def single_crossover(i, r_i):
			if r_i:
				e1 = entities[i]
				e2 = entities[i-1]
				x = 3
				e3 = T.concatenate([e1[:x], e2[x:]])
				e4 = T.concatenate([e2[:x], e1[x:]])
				return {entities: T.set_subtensor(T.set_subtensor(entities[[i-1,i]], e3)[i], e4)}
			else:
				return {}
				
		values, updates = theano.scan(fn=single_crossover, sequences=[T.arange(1,entities.shape[0],2), self.rng.uniform((entities.shape[0]/2,)) < self.crossover_rate])
		
		r = self.rng.choice(size = entities.shape, p = [self.mutate_rate, 1 - self.mutate_rate])
		return T.set_subtensor(updates[entities][r.nonzero()], (updates[entities][r.nonzero()] + 1) % 2)
#		return T.set_subtensor(        entities [r.nonzero()], (        entities [r.nonzero()] + 1) % 2)
		
	def run(self, generations = 500):
		log("Compiling...")
		
		E = self.initialize_random_population()
		F = np.zeros((E.shape[0]), dtype=theano.config.floatX)

		# Change these to shared variables
		E = theano.shared(E)
		F = theano.shared(F)

		# Create graphs
		fitness_t = self.fitness(E)
		select_t = self.selection(E, F)
		vary_t = self._vary(E)

		# Compile functions
		fitness = theano.function([], [], updates = [(F, fitness_t)])
		select = theano.function([], [], updates = [(E, select_t)])
		vary = theano.function([], [], updates = [(E, vary_t)])

		logOK("Compilation successfully completed.")
		log("Running algorithm...")

		start = time.time()

		fitness()
		
		for i in range(generations):
			print "it #", i
			select()
			#print "Before", E.get_value().sum()
			print "Fitness", np.max(F.get_value())
			vary()
			
			fitness()

		end = time.time()

		if theano.config.profile:
			theano.printing.pydotprint(fitness, outfile="fitness.png")
			theano.printing.pydotprint(select, outfile="select.png")
			theano.printing.pydotprint(vary, outfile="vary.png")

		return E.eval(), start, end
		
if __name__ == "__main__":

	def fit(entities):
		return T.sum(entities, axis=1)

	def sel(entities, fitness):
		# Create random integers
		rng = RandomStreams()
		r = rng.random_integers(size = (entities.shape[0]*2,), low = 0, high = entities.shape[0]-1)

		# Take randomly matched entities' fitnesses and reshape
		chosen_f = fitness[r].reshape((2, entities.shape[0]))

		# Take winner IDs (binary)
		winner_ids = T.argmax(chosen_f, axis=0)

		# Compute entity IDs
		entity_ids = r[T.arange(entities.shape[0])+(winner_ids*entities.shape[0])]

		return entities[entity_ids]

	ea = SimpleEA(fit, sel)

	entities, start, end = ea.run()
	f = np.max(np.sum(entities, axis=1))

	logOK("Done!")
	log("Max fitness: %i" % f)
	log("Time taken: %.5f seconds" % (end-start))
