

import random
import math
import numpy
import matplotlib.pyplot as plt
from collections import deque


def decide_to(p):
    r = random.uniform(0, 1)
    return r <= p


def mutate(ind, N):
    #Mutates the individual ind by swapping genomes
    i = random.randrange(0, N)
    j = random.randrange(0, N)
    ind[i], ind[j] = ind[j], ind[i]
    return


def crossover(father, mother, child1, child2, N):

    #Crossover between two genome

    i = random.randrange(0, N)
    j = random.randrange(0, N)
    if i > j:
        i, j = j, i

    check1 = numpy.zeros(N)
    check2 = numpy.zeros(N)

    for x in range(i, j + 1):
        child1[x] = father[x]  
        child2[x] = mother[x]
        check1[father[x]] = 1
        check2[mother[x]] = 1

    x = 0
    index = 0 + ((i == 0) * (j + 1))
    while x < N and index < N:
        if not check1[mother[x]]:
            child1[index] = mother[x]
            index += 1
        if index == i:
            index = j + 1
        x += 1
    x = 0
    index = 0 + ((i == 0) * (j + 1))
    while x < N and index < N:
        if not check2[father[x]]:
            child2[index] = father[x]
            index += 1
        if index == i:
            index = j + 1
        x += 1
    return


def distance(points, order, N):
    #Calculation of euclidean distance between points
    
    x0 = points[0][0]
    y0 = points[0][1]
    xi = points[order[0]][0]
    yi = points[order[0]][1]
    s = math.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
    for i in range(1, N):
        x1 = points[order[i - 1]][0]
        y1 = points[order[i - 1]][1]
        x2 = points[order[i]][0]
        y2 = points[order[i]][1]
        s += round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    #xn = points[order[N]][0]
    #yn = points[order[N]][1]
    xn = points[order[N-1]][0]
    yn = points[order[N-1]][1]
    s = s + math.sqrt((x0 - xn) ** 2 + (y0 - yn) ** 2)
    return s


def init_population(K, N):
    
    # Initializes the population
    
    pop = numpy.zeros((K, N), dtype=numpy.int32)

    seq = list(range(N))
    for i in range(K):
        random.shuffle(seq)
        pop[i] = seq
    return pop


def compute_population_fitness(pop, points, K, N):
    # Fitness for the population
    fit = numpy.zeros(K)
    for k in range(K):
        # fitness of each chromosome is the negative of its cycle distance
        fit[k] = -distance(points, pop[k], N)
    return fit


def find_cumulative_distribution(arr, K):
    cd = numpy.zeros(K)
    acc = 0
    s = arr.sum()
    for i in range(K):
        acc += arr[i] / s
        cd[i] = acc
    return cd


def select_parent(fitness, K):
    # the parent is selected based by roulette wheel technique.
    
    local_absolute_fitness = fitness - fitness.min()
    cd = find_cumulative_distribution(local_absolute_fitness, K)
    roulette = random.uniform(0, 1)
    ind = 0
    while roulette > cd[ind]:
        ind += 1
    return ind


def create_new_population(pop, fitness, K, N, crossover_probability, mutation_probability):

    #Creates a new population of K chromosomes of N genomes by built function 

    new_pop = numpy.zeros((K, N), dtype=numpy.int32)
    for k in range(K // 2):  # 2 children are created in each iteration
        father_ind = select_parent(fitness, K)
        mother_ind = select_parent(fitness, K)

        father = pop[father_ind]
        mother = pop[mother_ind]
        child1 = father.copy()
        child2 = mother.copy()

        if decide_to(crossover_probability):
            crossover(father, mother, child1, child2, N)
        if decide_to(mutation_probability):
            mutate(child1, N)
        if decide_to(mutation_probability):
            mutate(child2, N)

        new_pop[k * 2] = child1
        new_pop[k * 2 + 1] = child2 
    return new_pop


def find_best_individual(pop, fitness, best_individual, best_fit):
    
#    Finds the best one and its fitness in all the generations.
    
    current_best_index = fitness.argmax()
    current_best_fit = fitness[current_best_index]
    current_best_individual = pop[current_best_index]

    if best_fit < current_best_fit:
        return current_best_individual, current_best_fit
    else:
        return best_individual, best_fit 
    
    
# If the best individual is not as good as the current individual, then the current individual is the best

def read_input(path, N):
    
#     Read the input from the  file
    
    points = numpy.zeros((N, 2))
    file = open(path)
    lines = file.readlines()
    lines = [x.replace(',',' ') for x in lines] #Read the file line by line, replacing ‘,’ with spaces
    file.close()
    for i in range(N):
        points[i][0], points[i][1] = map(int, lines[i].split())
    return points


def plot_individual_path(individual, points, title, index):

    x = []
    y = []
    for i in individual:
        x.append(points[i][0])
        y.append(points[i][1])
    x.append(x[0])
    y.append(y[0])

    plt.subplot(3, 5, index)
    plt.title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.plot(x, y, 'r*')
    plt.plot(x, y, 'g--')
    return


def plot_results(best_last_15, points):

    for i in range(0,len(best_last_15)):
        plot_individual_path(best_last_15[i][0], points, str(round(best_last_15[i][1], 2)), i+1)
    plt.show()
    return


def TSP_genetic(n, k, max_generation, crossover_probability, mutation_probability, path):

    points = read_input(path, n)
    population = init_population(k, n)
    best_individual = population[0]  
    best_fitness = -distance(points, best_individual, n)  
    old_best_fitness = best_fitness
    best_last_15 = deque([], maxlen=15)  

    for generation in range(1, max_generation + 1):
        
        fitness = compute_population_fitness(population, points, k, n)
        
        best_individual, best_fitness = find_best_individual(population, fitness, best_individual, best_fitness)
        
        if old_best_fitness != best_fitness:
            old_best_fitness = best_fitness
            best_last_15.append((best_individual.copy(), -best_fitness))
        
        population = create_new_population(population, fitness, k, n, crossover_probability, mutation_probability)
        
        print("Generation = ", generation,'\t',"Path length = ",-best_fitness)

    solution = best_individual
    cycle_distance = -best_fitness

    plot_results(best_last_15, points)
    return solution+1
