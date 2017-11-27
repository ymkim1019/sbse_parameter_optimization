from functions.fault_localization import f_fault_localization
from basian_opt import GPoptimizer

configs = [['int',2,15],    #maximum_tree_depth
           ['int',20,100],  #population_size
           ['int',0,15],  #elitism_size
           ['float',0,1],   #cxpb
           ['float',0,1],    #mutpb
           ]

def evaluate(f,samples):
    scores = []
    for sample in samples:
        scores.append(f(sample))
    return scores

f = open("fault_localization_bo.txt", 'w')
gpo = GPoptimizer(configs,f_fault_localization)
best_score = 0
initial_samples = gpo.random_sample(3)
for i in range(300):
    scores = evaluate(f_fault_localization,initial_samples)
    if best_score < max(scores):
        best_score = max(scores)

    data = str.format('{}  {}', (i+1)*3, best_score)
    f.write(data)
    print(data)

    gpo.update_model(initial_samples,scores) #새로운 데이터 추가

    next_samples = gpo.random_sample(3)
    prob = gpo.compute_PI(next_samples)

print(best_score)
