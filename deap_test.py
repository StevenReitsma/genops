import random
from deap import creator, base, tools, algorithms
import time
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=2000)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.003)
toolbox.register("select", tools.selTournament, tournsize=2)

population = toolbox.population(n=1000)

start = time.time()
NGEN=5000
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.05)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    print "Iteration %i, fitness: %i" % (gen, np.max(fits))
    population = toolbox.select(offspring, k=len(population))
top10 = tools.selBest(population, k=10)
end = time.time()

print end-start

"""
OneMax problem with 1000 entities and 1000 bits.

DEAP reaches fitness ~890 in 68 seconds on CPU.
Genops reaches fitness ~890 in 60 seconds on GPU and 128 seconds on CPU.
"""
