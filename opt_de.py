import random
import numpy as np

def differential_evolution(func, bounds, popsize=20, mutation=0.8, recombination=0.7, maxiter=100, tol=1e-6):
    """
    Differential Evolution optimization algorithm with binary encoding for multiple discrete decision variables.

    Parameters:
    func (function): Objective function to minimize.
    bounds (list): List of tuples with the lower and upper bounds of each decision variable.
    popsize (int): Population size.
    mutation (float): Mutation rate.
    recombination (float): Recombination rate.
    maxiter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.

    Returns:
    best_solution (list): Best solution found.
    best_fitness (float): Best fitness value found.
    """
    num_vars = len(bounds)
    
    # Create bins for each discrete variable
    bins = []
    for i in range(num_vars):
        lower, upper = bounds[i]
        num_bins = int(upper - lower) + 1
        bins.append(np.linspace(lower, upper, num_bins))
    
    # Convert binary string to continuous variables
    def decode_chromosome(chromosome):
        variables = []
        for i in range(num_vars):
            num_bits = int(np.ceil(np.log2(len(bins[i]))))
            bin_str = ''.join(str(bit) for bit in chromosome[i*num_bits:(i+1)*num_bits])
            bin_int = int(bin_str, 2)
            variables.append(bins[i][bin_int])
        return variables
    
    # Encode continuous variables as binary strings
    def encode_chromosome(variables):
        chromosome = []
        for i in range(num_vars):
            num_bits = int(np.ceil(np.log2(len(bins[i]))))
            bin_int = np.abs(bins[i] - variables[i]).argmin()
            bin_str = bin(bin_int)[2:].zfill(num_bits)
            chromosome.extend([int(bit) for bit in bin_str])
        return chromosome
    
    # Initialize population with random solutions
    population = [encode_chromosome([random.choice(bins[i]) for i in range(num_vars)]) for _ in range(popsize)]
    fitness = [func(decode_chromosome(chromosome)) for chromosome in population]
    
    # Find best solution
    best_idx = np.argmin(fitness)
    best_solution = decode_chromosome(population[best_idx])
    best_fitness = fitness[best_idx]
    
    # Main loop
    for i in range(maxiter):
        for j in range(popsize):
            # Select three unique solutions
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = random.sample(idxs, 3)
            
            # Mutation
            mutant = [population[a][k] + mutation * (population[b][k] - population[c][k]) for k in range(num_vars)]
            mutant = [max(min(mutant[k], 1), 0) for k in range(num_vars)] # Ensure values are between 0 and 1
            
            # Crossover
            trial = []
            for k in range(num_vars):
                if random.random() < recombination:
                    trial.append(mutant[k])
                else:
                    trial.append(population[j][k])
            
            # Convert trial solution to binary string and decode
            trial_decoded = decode_chromosome(encode_chromosome(trial))
            trial_fitness = func(trial_decoded)
            
            # Update population and best solution
            if trial_fitness < fitness[j]:
                population[j] = encode_chromosome(trial)
                fitness[j] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_solution = trial_decoded
                    best_fitness = trial_fitness
                    
        # Check convergence
        if best_fitness < tol:
            break
            
    return best_solution, best_fitness


# Define objective function to minimize
def rosenbrock(x):
    return (100.0 *x[0]+5*x[1]**+x[2])

# Define bounds of each variable
bounds = [(-5, 5), (-5, 5), (-5, 5)]

# Call differential_evolution function
best_solution, best_fitness = differential_evolution(rosenbrock, bounds)

# Print best solution and fitness
print('Best solution:', best_solution)
print('Best fitness:', best_fitness)