from blackbox import f_mnist
from basian_opt import GPoptimizer

configs = [['int',32,128],    #conv_output_size
           ['float',0.1,0.99],  #conv_dropout_rate
           ['int',2,4],  #maxpooling_size
           ['int',2,5],   #num_of_dense_layers
           ['int',32,128],    #dense_output_size
           ['float',0.1,0.99],  #dense_drop_out_rate
           ['float',0.0001,0.001]]  #learning_rate

def evaluate(f,samples):
    scores = []
    for sample in samples:
        scores.append(f(sample))
    return scores

gpo = GPoptimizer(configs,f_mnist)

initial_samples = gpo.random_sample(3)
scores = evaluate(f_mnist,initial_samples)

gpo.update_model(initial_samples,scores) #새로운 데이터 추가

next_samples = gpo.random_sample(3)
prob = gpo.compute_PI(next_samples)

print(prob)
