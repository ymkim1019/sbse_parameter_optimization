import operator
import math
import random
import numpy
import os
import glob
import pandas as pd
from scipy.stats import rankdata

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedSqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return math.sqrt(-x)


class eaSimpleCustom:
    def __init__(self):
        self.current_gen = 0

    def do(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution

        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evalutions for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(ngen):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, cxpb, mutpb)
                evaluate(offspring)
                population = offspring

        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *ngen*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        .. note::

            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.

        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
           Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.current_gen = gen
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

def evalSymbReg(individual, toolbox, ea, files, indexes):
    sum_fitness = 0
    num_of_faults = 0

    for cnt, idx in enumerate(indexes[ea.current_gen]):
        df = pd.read_csv(files[idx])
        data = df.values
        func = toolbox.compile(expr=individual)
        ranks = rankdata([func(*each[1:42]) for each in data[:]], method='max')
        fault_indexes = numpy.where(data[:, 42] == 1)
        num_of_faults += len(fault_indexes)
        sum_fitness += numpy.sum([ranks[i] for i in fault_indexes])

    return sum_fitness/num_of_faults,

def f_fault_localization(params):
    """
    maximum_tree_depth : int, (2 ~ 15)
    population_size : int, (20 ~ 100)
    elitism_size : int, (0 ~ 15)
    cxpb : float, (0 ~ 1)
    mutpb : float, (0 ~ 1)

    :param params: list of parameters
    :type params: list
    :return: fitness
    :rtype: float
    """
    num_of_features = 41
    num_of_generations = 100
    num_of_samples = 5

    max_tree_depth = params[0]
    population_size = params[1]
    elitism_size = params[2]
    cxpb = params[3]
    mutpb = params[4]

    # dataset
    dataset_path = os.getcwd() + '\\fluccs_data\\'
    extension = 'csv'
    files = [i for i in glob.glob(dataset_path + '*.{}'.format(extension))]
    # randomly sample

    sample_indexes = list()
    for i in range(num_of_generations+1):
        sample_indexes.append(numpy.random.randint(0, len(files) - 1, num_of_samples))

    # GP operators from Table 3 in the FLUCCS paper
    pset = gp.PrimitiveSet("MAIN", num_of_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protectedSqrt, 1)

    # The goal is to minimize a value from the objective function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    ea = eaSimpleCustom()

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg, toolbox=toolbox, ea=ea, files=files, indexes=sample_indexes)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))

    random.seed(318)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(elitism_size)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = ea.do(pop, toolbox, cxpb, mutpb, num_of_generations, stats=mstats,
                                   halloffame=hof, verbose=True)

    sum_fitness = 0
    num_of_faults = 0
    func = toolbox.compile(expr=hof[0])
    for f in files:
        df = pd.read_csv(f)
        data = df.values
        ranks = rankdata([func(*each[1:42]) for each in data[:]], method='max')
        fault_indexes = numpy.where(data[:, 42] == 1)
        num_of_faults += len(fault_indexes)
        sum_fitness += numpy.sum([ranks[i] for i in fault_indexes])
    avg_fitness = sum_fitness/num_of_faults

    return avg_fitness,

def main():
    args = list()
    # from FLUCCS
    # args.append(8) # max_tree_depth
    # args.append(49) # population_size
    # args.append(8) # elitism_size
    # args.append(1.0) # cxpb
    # args.append(0.1) # mutpb

    args.append(8)  # max_tree_depth
    args.append(30)  # population_size
    args.append(8)  # elitism_size
    args.append(1.0)  # cxpb
    args.append(0.1)  # mutpb
    print(f_fault_localization(args))

if __name__ == "__main__":
    main()