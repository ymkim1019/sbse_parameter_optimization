import argparse
import datetime
from basian_opt import GPoptimizer
from functions.fault_localization import f_fault_localization
from functions.mnist import f_mnist

def evaluate(f,samples):
    scores = []
    for sample in samples:
        print('params =', sample)
        scores.append(f(sample))
    return scores

def main(args):
    configs = None
    fitness_func = None
    gp_batch_size = 5

    if args.f == 'f_mnist':
        fitness_func = f_mnist
        configs = [['int', 32, 128],  # conv_output_size
                    ['float', 0.1, 0.99],  # conv_dropout_rate
                    ['int', 2, 4],  # maxpooling_size
                    ['int', 2, 5],  # num_of_dense_layers
                    ['int', 32, 128],  # dense_output_size
                    ['float', 0.1, 0.99],  # dense_drop_out_rate
                    ['float', 0.0001, 0.001]]  # learning_rate
    elif args.f == 'f_fault_localization':
        fitness_func = f_fault_localization
        configs = [['int', 2, 15],  # maximum_tree_depth
                    ['int', 1, 5],  # elitism_size
                    ['float', 0, 1],  # cxpb
                    ['float', 0, 1],  # mutpb
                    ]

    # log file
    now = datetime.datetime.now()
    f = open(str.format("{}_{}_{}_{}_{}.txt", args.f, args.algo, now.day, now.hour, now.minute), 'w')
    f.write(str.format('{}  [}  {}  {}\n', 'round', 'i', 'fitness', 'best_fitness'))

    if args.algo == 'BO':
        for i in range(args.n_evals):

            gpo = GPoptimizer(configs, fitness_func)
            best_score = 0
            cnt = 0
            while True:
                n = gp_batch_size if (args.n_samples - cnt) >= gp_batch_size else args.n_samples - cnt
                samples = gpo.random_sample(n)
                scores = evaluate(fitness_func, samples)
                if best_score < max(scores):
                    best_score = max(scores)

                for k in range(n):
                    data = str.format('{}   {}  {}  {}\n', i, cnt+k, scores[k], best_score)
                    f.write(data)
                    f.flush()
                    print(data)

                cnt += n

                gpo.update_model(samples, scores)
                if cnt == args.n_samples:
                    break

            print('best fitness =', best_score)

        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, choices=['f_mnist', 'f_fault_localization'])
    parser.add_argument("-algo", "--algo", type=str, choices=['BO','GA','GABO'])
    parser.add_argument("-n_samples", "--n_samples", default=100, type=int)
    parser.add_argument("-n_evals", "-n_evals", default=5, type=int)

    args = parser.parse_args()

    main(args)

    # e.g.
    # python optimizer.py -f f_fault_localization -algo BO -n_samples 100 -n_evals 10
