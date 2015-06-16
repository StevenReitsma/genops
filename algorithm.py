import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
import numpy as np
import theano.misc.pycuda_init
import theano.sandbox.cuda as cuda
import time
from abc import ABCMeta
import sys

import pycuda.autoinit
import pycuda.driver as cuda_driver
from pycuda.compiler import SourceModule

def logOK(text):
	print time.strftime('%H:%M:%S') + ": " + '\033[92m' + text + '\033[0m'

def logERROR(text):
	print time.strftime('%H:%M:%S') + ": " +  '\033[91m' + text + '\033[0m'

def log(text):
	print time.strftime('%H:%M:%S') + ": " +  text

class ChooseOp(theano.Op):
	def __eq__(self, other):
		return type(self) == type(other)
	def __hash__(self):
		return hash(type(self))
	def __str__(self):
		return self.__class__.__name__
	def make_node(self, a, choice1):
		a  = cuda.basic_ops.gpu_contiguous(
			 cuda.basic_ops.as_cuda_ndarray_variable(a))
		c1 = cuda.basic_ops.gpu_contiguous(
			 cuda.basic_ops.as_cuda_ndarray_variable(choice1))
		assert a.dtype == "float32"
		assert c1.dtype == "float32"
		return theano.Apply(self, [a, c1], [c1.type()])
	
	def make_thunk(self, node, storage_map, _, _2):
		mod = SourceModule("""
		__global__ void choose(float *dest, float *a, float *b, int N, int m) {
			const int i = threadIdx.x + blockIdx.x * blockDim.x;
			const int gene = i/m;
			if (i < N) {
				if (i%m < a[gene]) {
					dest[i]       = b[i];
					dest[i + N/2] = b[i + N/2];
				} else {
					dest[i]       = b[i + N/2];
					dest[i + N/2] = b[i];
				}
			}
		}""")
		choose_cuda = mod.get_function("choose")
		inputs  = [storage_map[v] for v in node.inputs]
		outputs = [storage_map[v] for v in node.outputs]
		def thunk():
			n = inputs[1][0].shape[1]  # Number of bit pairs
			m = n / inputs[0][0].size  # Gene length
			t = 512                    # Number of threads
			z = outputs[0]
			z[0] = cuda.CudaNdarray.zeros(inputs[1][0].shape)
			grid = (int(np.ceil(n / 512.)), 1)
			choose_cuda(z[0], inputs[0][0], inputs[1][0], 
				np.intc(n), np.intc(m),
				block=(512,1,1), grid=grid)
		return thunk
choose = ChooseOp()

class EA(object):
	__metaclass__ = ABCMeta

	def __init__(self):
		self.rng = RandomStreams()

		if 'gpu' in theano.config.device:
			self.fast_rng = CURAND_RandomStreams(seed = 42)
		else:
			self.fast_rng = None

		self.entity_mutate_rate = 0.003
		self.bit_mutate_rate = 0.05
		self.crossover_rate = 0.7
	
	def initialize_random_population(self):
		return np.random.randint(2, size = (1000, 2000)).astype(theano.config.floatX) # 1000 entities, 1000 bits

class SimpleEA(EA):
	def __init__(self, fitnessFunction):
		"""
		Initialize a simple EA algorithm class.
		Creating the SimpleEA object compiles the needed Theano functions.

		:param fitnessFunction: The fitness function that takes a matrix of entities and and outputs a vector of fitness values for the ids given in `changes`.
		:param selectionFunction: The selection function that takes a matrix of entities and a vector of fitnesses and outputs a matrix of entities.
		"""
		super(SimpleEA, self).__init__()

		self.fitness = fitnessFunction
	
	def cross(self, entities):
		n, m = entities.shape
		pop = T.reshape(entities, (2, n*m/2))

		crosslength = T.cast(self.crossover_rate * n * m / 2, 'int32')
		
		if self.fast_rng is None:
			xpoints = self.rng.random_integers(size = (n / 2,), low = 0, high = m-1)
		else:
			xpoints = self.fast_rng.uniform(size = (n / 2,), low = 0, high = m-1)
			xpoints = xpoints.astype('int32')
			xpoints = xpoints.astype('float32')
		
		c1 = choose(xpoints, pop[:,:crosslength])
		c1 = T.reshape(c1, (2, crosslength))
		return T.reshape(T.concatenate([c1, pop[:, crosslength:]], axis=1), (n,m))

	def tournament_selection(self, entities, fitness):
		"""
		TODO:
		* Contains 1 HostFromGpu (argmax)
		"""
		# Create random integers
		if self.fast_rng is None:
			r = self.rng.random_integers(size = (entities.shape[0]*2,), low = 0, high = entities.shape[0]-1)
		else:
			r = self.fast_rng.uniform(size = (entities.shape[0]*2,), low = 0, high = entities.shape[0]-1)
			r = r.astype('int32')

		# Take randomly matched entities' fitnesses and reshape
		chosen_f = fitness[r].reshape((2, entities.shape[0]))

		# Take winner IDs (binary)
		winner_ids = T.argmax(chosen_f, axis=0)

		# Compute entity IDs
		entity_ids = r[T.arange(entities.shape[0])+(winner_ids*entities.shape[0])]

		return entities[entity_ids]

	def fast_mutation(self, entities):
		"""
		Randomly flips bits in the population matrix.

		TODO:
		* Contains 1 HostFromGpu
		* Contains 1 GpuFromHost
		"""
		p = self.bit_mutate_rate * self.entity_mutate_rate

		if self.fast_rng is None:
			r = self.rng.choice(size = entities.shape, p = [1-p, p])
		else:
			r = self.fast_rng.uniform(size = entities.shape)
			r = r < p

		non_zero_indices = r.nonzero()

		to_be_changed = entities[non_zero_indices]
		changed_to = 1 - to_be_changed

		entities = T.set_subtensor(to_be_changed, changed_to)

		return entities
		
	def run(self, generations = 50, E=None):
		log("Compiling...")
		
		if E is None:
			E = self.initialize_random_population()

		n_entities = E.shape[0]
		n_bits = E.shape[1]

		F = np.zeros((E.shape[0]), dtype=theano.config.floatX)

		# Change these to shared variables
		E = theano.shared(E)
		F = theano.shared(F)

		# Create graphs
		fitness_t = self.fitness(E)
		select_t = self.tournament_selection(E, F)
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
			if np.max(F.get_value()) == n_bits:
				break

		end = time.time()

		if theano.config.profile:
			theano.printing.pydotprint(fitness, outfile="fitness.pdf", format="pdf")
			theano.printing.pydotprint(select, outfile="select.pdf", format="pdf")
			theano.printing.pydotprint(mutate, outfile="mutate.pdf", format="pdf")
			theano.printing.pydotprint(crossover, outfile="cross.pdf", scan_graphs = True, format="pdf")

		return E.eval(), start, end, i+1, F.get_value()
		
if __name__ == "__main__":

	def fit(entities):
		return T.sum(entities, axis=1)

	ea = SimpleEA(fit)

	if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
		times = 20
	else:
		times = 1

	total_time = 0
	total_iterations = 0

	for i in range(times):
		entities, start, end, iterations,f = ea.run(generations = 1500)
		total_time += end-start
		total_iterations += iterations

	avg_time = total_time / float(times)
	avg_iterations = total_iterations / float(times)

	f = np.max(f)

	logOK("Done.")
	log("Device: %s" % theano.config.device)
	log("Max fitness: %f" % f)
	log("Time taken: %.5f seconds" % avg_time)
	log("Iterations taken: %.1f" % avg_iterations)
