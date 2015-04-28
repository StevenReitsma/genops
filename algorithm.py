import theano
import theano.tensor as T
from abc import ABCMeta, abstractmethod

class EA(object):
	__metaclass__ = ABCMeta
	
	def initialize_random_population(self):
		pass

class SimpleEA(EA):
	def _vary(entities):
		pass
		
	def __init__(self, fitnessFunction, selectionFunction):
		"""
		Initialize a simple EA algorithm class.
		Creating the SimpleEA object compiles the needed Theano functions.
		
		:param fitnessFunction: The fitness function that takes a matrix of entities and and outputs a vector of fitness values for the ids given in `changes`.
		:param selectionFunction: The selection function that takes a matrix of entities and a vector of fitnesses and outputs a matrix of entities.
		"""
		
		entities = T.matrix('entities')
		fitnesses = T.vector('fitnesses')
		changes = T.vector('changes')
		
		self.fitness = theano.function([entities, changes], fitnessFunction)
		self.select = theano.function([entities, fitness], selectionFunction)
		self.vary = theano.function([entities], self._vary)
		
	def run(self, generations = 100):
		E = self.initialize_random_population()

		changes = np.zeros((E.shape[0]))
		
		E = theano.shared(E)
		changes = theano.shared(E)
		
		F = self.fitness(E, np.ones((E.shape[0])))
		
		for i in range(generations):
			S = self.select(E, F)
			S, C = self.vary(S)
			
			F = self.fitness(S, C)
			
			E = S
		
		return E.eval()
		