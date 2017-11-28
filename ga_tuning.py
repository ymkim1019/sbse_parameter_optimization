from blackbox import f_mnist
from basian_opt import GPoptimizer
import random
import ga_evaluation as evo
from math import inf

class GAtuning():
    def __init__(self, parameter_conf):
        self.configs = parameter_conf
        self.population = 0
        self.samples = []
        self.result = []
        self.parent = []
        self.children = []
        self.gpo = GPoptimizer(parameter_conf, f_mnist)

        # parameter
        # self.crossover_rate =
        self.mutation_rate = 0.05
        self.elite = int(self.population * 0.4)
        
        # selection
        self.entire = 0
        self.best = -inf

    def generation(self, num, pop): # it takes number of generation and number of population
        self.samples = self.initialization(pop)
        for _ in range(num):
            self.evaluation()     
            self.gpo.update_model(self.samples, self.result)
            self.selection()
            self.crossover()
            self.mutation()
            self.basian_elitism()
        self.evaluation()
        return self.result, self.samples

    def initialization(self, pop): # it takes number of population
        self.population = pop
        samples = []
        for _ in range(pop):
            sample = []
            for parameter_conf in self.configs:
                if parameter_conf[0] == 'int':
                    value = random.randrange(parameter_conf[1],parameter_conf[2]+1)
                if parameter_conf[0] == 'float':
                    value = random.random() * (parameter_conf[2] - parameter_conf[1]) + parameter_conf[1]
                sample.append(value)
            samples.append(sample)
        return samples

    def evaluation(self): # XXX if your configs are changed then you have to changed ga_evaluation code -> this will be updated after
        self.result = []
        self.entire = 0
        self.best = -inf
        for sample in self.samples:
            fit = float(evo.run(sample))
            self.result.append(fit)
            self.entire += fit
            if fit > self.best:
                self.best = fit

    def selection(self):
        numParent = self.population
        rand = random.random()
        prob = probSet()
        t = 0
        for i in range(len(self.result)):
            x = t
            p = (self.result[i] / self.entire) * numParent # not use windowing
            t += p
            prob.append(bound(x, t))
        self.parent = self.stochastic(numParent, rand, prob)

    def stochastic(self, nump, rand, prob):
        parent = []
        for i in range(nump):
            isini = prob.isin(rand + i) 
            parent.insert(random.randint(0,len(parent)),self.samples[isini]) # prevent determined permutation
        return parent

    def crossover(self):
        l = len(self.parent)
        self.children = []
        for i in range(0,len(self.parent),2):
            if i+1 == len(self.parent):
                self.children.append(self.parent[i])
                break
            cp = random.randint(1,l)
            self.children.append(self.parent[i][0:cp] + self.parent[i+1][cp:])
            self.children.append(self.parent[i+1][0:cp] + self.parent[i][cp:])

    def mutation(self):
        for i in range(len(self.children)):
            for j in range(len(self.children[i])):
                if random.random() < self.mutation_rate:
                    if self.configs[j][0] == 'int':
                        value = random.randrange(self.configs[j][1],self.configs[j][2]+1)
                    elif self.configs[j][0] == 'float':
                        value = random.random() * (self.configs[j][2] - self.configs[j][1]) + self.configs[j][1]
                    self.children[i][j] = value

    def basian_elitism(self):
        prob = self.gpo.compute_PI(self.children)
        prob = list(prob)
        sorted_children = [x for _,x in sorted(zip(prob,self.children))] # small number front
        sorted_samples = [x for _,x in sorted(zip(self.result, self.samples))] # big number front
        sorted_samples.reverse()
        self.samples = sorted_children[self.elite:] + sorted_samples[:self.elite]

        
# Utils--------------------------------------------------------------
class probSet:
    def __init__(self):
        self.budget = []

    def append(self, bound):
        self.budget.append(bound)

    def isin(self, rand):
        for i in range(len(self.budget)):
            if self.budget[i].isin(rand):
                break
        return i

class bound:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def isin(self, rand):
        if rand >= self.start and rand < self.end:
            return True
        else:
            return False

if __name__ == "__main__":
    configs = [['float',0,1.0],    # mutation_rate
               ['float',0,1.0],  # crossover_rate
               ['int',1,4]]  # selection_function
    ga = GAtuning(configs)
    print (ga.generation(2,2))
