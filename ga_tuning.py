from blackbox import f_mnist
from basian_opt import GPoptimizer
import random
import ga_evaluation as evo
from math import inf

class GAtuning():
    # parameter_conf : list of list, pop : number, gennum : number, fitness_ftn : list -> number, basian : boolean 
    def __init__(self, parameter_conf, pop, gennum, fitness_ftn, basian, log_file = None, round = None):
        # initialize
        self.configs = parameter_conf
        self.population = pop
        self.generation_number = gennum
        self.fitness_ftn = fitness_ftn
        self.basian = basian
        self.log_file = log_file
        self.round = round

        # generation
        self.samples = []
        self.result = []
        self.parent = []
        self.children = []
        if self.basian:
            self.gpo = GPoptimizer(parameter_conf, self.fitness_ftn) # XXX

        # parameter
        # self.crossover_rate =
        self.mutation_rate = 0.05
        self.elite = int(self.population * 0.4)
        self.rank_param = 1.5 # 1 <= s <= 2
        
        # selection
        self.entire = 0
        self.best = -inf
        self.rank = []

    def generation(self): # it takes number of generation and number of population
        num = self.generation_number
        pop = self.population
        self.samples = self.initialization(pop)
        best_score = 0
        for gen in range(num):
            self.evaluation()

            best_score = max([best_score] + self.result)
            for i in range(len(self.result)):
                data = str.format('{}   {}  {}  {}\n', self.round, gen * self.population + i
                                  , self.result[i], best_score)
                if self.log_file is not None:
                    self.log_file.write(data)
                    self.log_file.flush()
                print(data)

            if self.basian:
                self.gpo.update_model(self.samples, self.result)
            self.linear_rank_selection()
            self.crossover()
            self.mutation()
            if self.basian:
                self.basian_elitism()
            else:
                self.elitism()
        self.evaluation()
        best_score = max([best_score] + self.result)
        for i in range(len(self.result)):
            data = str.format('{}   {}  {}  {}\n', self.round, gen * self.population + i
                              , self.result[i], best_score)
            if self.log_file is not None:
                self.log_file.write(data)
                self.log_file.flush()
            print(data)

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
            print(sample)
            fit = self.fitness_ftn(sample)
            self.result.append(fit)
            self.entire += fit
            if fit > self.best:
                self.best = fit
        seq = sorted(self.result)
        self.rank = [seq.index(v) for v in x]

    def selection(self):
        if self.basian:
            numParent = self.population
        else:
            numParent = self.population - self.elite
        rand = random.random()
        prob = probSet()
        t = 0
        for i in range(len(self.result)):
            x = t
            p = (self.result[i] / self.entire) * numParent # not use windowing
            t += p
            prob.append(bound(x, t))
        self.parent = self.stochastic(numParent, rand, prob)

    def linear_rank_selection(self):
        if self.basian:
            numParent = self.population
        else:
            numParent = self.population - self.elite
        rand = random.random()
        prob = probSet()
        t = 0
        k1 = (2-s)/self.population
        k2 = (s-1)/sum(self.rank)
        for i in range(len(self.rank)):
            x = t
            p = (k1 + k2*self.rank[i]) * numParent
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
        assert len(self.samples) == self.population

    def elitism(self):
        sorted_samples = [x for _,x in sorted(zip(self.result, self.samples))] # big number front
        sorted_samples.reverse()
        self.samples = self.children + sorted_samples[:self.elite]
        assert len(self.samples) == self.population


        
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
    ga = GAtuning(configs,2,2,evo.run,True)
    print (ga.generation())
