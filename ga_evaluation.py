from subprocess import *

# need to change this jar path to own path
jar_path = "evosuite-1.0.5.jar"

def exec_evosuite_n_get_fitness(*args):
    fitness = -1
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        outline = line[:-1].decode('utf-8')
        if len(outline) != outline.count("\n"):
            print(outline)
            ret.append(outline)
            if "fitness:" in outline:
                fitness = outline.split('fitness:')[1].strip()

    # below for process error message
    stdout, stderr = process.communicate()
    print(stderr)
    #ret += stderr.split('\n')

    # total report
    # print(ret)
    return fitness

def run (arglist): 
    arg1,arg2,arg3 = arglist # XXX this code have to be changed
    selection_function  = { 1 : 'RANK', 2 : 'ROULETTEWHEEL', 3 : 'TOURNAMENT', 4 : 'BINARY_TOURNAMENT' }
    arg3 = selection_function[arg3]

    args = [jar_path, '-class', 'tutorial.Stack', '-projectCP', 'Tutorial_Stack/target/classes', '-Dmutation_rate', str(arg1), '-Dcrossover_rate', str(arg2), '-Dselection_function', str(arg3)]
    # for more weeker criterion (only use branch coverage)
    #args = [jar_path, '-class', 'tutorial.Stack', '-projectCP', 'target/classes', '-criterion', 'branch','-Dmutation_rate', str(arg1), '-Dcrossover_rate', str(arg2), '-Dselection_function', str(arg3)]
    print(args)

    # There is some runtime error if fitness result is 1
    result = exec_evosuite_n_get_fitness(*args)
    return float(result)


if __name__ == "__main__":
    mutation_rate = { 1 : 0 , 2 : 0.2, 3 : 0.5 ,4 : 0.75 ,5 : 1.0 }
    crossover_rate = { 1 : 0 , 2 : 0.2 , 3 : 0.5 ,4 : 0.75 ,5 : 1.0 }
    # need 5??, 1 more hyper value?
    selection_function  = { 1 : 'RANK', 2 : 'ROULETTEWHEEL', 3 : 'TOURNAMENT', 4 : 'BINARY_TOURNAMENT' }

    arg1 = mutation_rate[1]
    arg2 = crossover_rate[2]
    arg3 = selection_function[1]
    args = [arg1,arg2,arg3]

    result = run(args)

    print("fitness : ", str(result))
