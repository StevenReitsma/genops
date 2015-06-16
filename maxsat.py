import csv
import theano.tensor as T
import theano
import numpy as np
import os

class MaxSat():
	
	def __init__(self, filename):
		with open(filename) as openfile:
			reader = csv.reader(openfile, delimiter=' ')
			for line in reader:
				if line[0] == 'c':
					continue
				elif line[0] == 'p':
					self.clauses = np.zeros((int(line[4]), int(line[2])))
					break
			for i,line in enumerate(reader):
				for v in line[:-1]:
					self.clauses[i, abs(int(v))-1] = 2*('-' in v)-1
		with open('data/answers.csv') as openfile:
			for line in csv.reader(openfile, delimiter=','):
				if line[1] == filename.split('/')[1]:
					self.answer = line[2]
					break
				
	def fitness(self, bits):
		q = np.sum(self.clauses != 0, axis=1)
		predicates = 2*bits.get_value() - 1
		ncorrect = T.dot(self.clauses, predicates.T)
		f = theano.function(inputs=[], outputs=[ncorrect])
		ncorrect = T.eq(ncorrect.T, q)
		f = theano.function(inputs=[], outputs=[ncorrect])
		f = theano.function(inputs=[], outputs=[T.sum(ncorrect, axis=1)])
		return T.sum(ncorrect, axis=1).astype(theano.config.floatX)

if __name__ == '__main__':
	import algorithm
	import sys
	if len(sys.argv) > 1:
		filenames = ['data/' + sys.argv[1]]
	else:
		filenames = ['data/' + f for f in os.listdir('data/') if f[-4:] == '.cnf']
	
	ttime = 0
	tfit = 0
	titrs = 0
	for filename in filenames:
		ms = MaxSat(filename)
		print "EXPECTATION: ", ms.answer
		ncl, npr = ms.clauses.shape
		ngenes = 1000
		its    = 10
		ea = algorithm.SimpleEA(ms.fitness)
		pop = np.random.randint(2, size = (ngenes, npr)).astype(theano.config.floatX)
		entities, start, end, iterations, fit = ea.run(generations = its, E=pop)
		print "TIME: ", end-start
		print "FITNESS: ", np.max(fit)
		print "ITS: ", iterations
		ttime += end-start
		tfit  += np.max(fit)
		titrs += iterations
	n = len(filenames)
	print "MEAN TIME: ", ttime / n
	print "MEAN FITNESS: ", tfit / n
	print "MEAN ITERATIONS: ", titrs / n
