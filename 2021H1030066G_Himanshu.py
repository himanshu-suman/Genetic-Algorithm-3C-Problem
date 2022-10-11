import random
import time
from random import randint

import networkx as nx
from matplotlib import pyplot as plt
from numpy.random import rand

from Graph_Creator import *

random.seed(47)

# implement genetic algorithm for 3 coloring problem for graph with 50 vertices
# finding a state with the best fitness function value
# TODO: use random.choice instead of random.randint


def reproduce(parent1, parent2):
    # code to create a child from two parents
    pivot = random.choice([i for i in range(49)])
    return parent1[:pivot] + parent2[pivot:]


def genetic_algorithm(graph, population_size, mutation_rate, max_gen):
    # initialize a population of random states
    population = [create_state(graph) for i in range(population_size)]
    # print(len(population))
    # print(population[0])
    # track the best state and fitness
    fitness = [fitness_function(state, graph) for state in population]
    best_state = population[fitness.index(max(fitness))]
    best_fitness = max(fitness)
    track_best_state = []
    track_best_fitness = []
    curr_gen = 0
    start = time.time()
    while (int(round(time.time())) - start <= 45) and (curr_gen < max_gen):
        # while (curr_gen < max_gen):
        # calculate the fitness of each state in the population
        fitness = [fitness_function(state, graph) for state in population]
        # keep track of the best state
        if max(fitness) > best_fitness:
            best_state = population[fitness.index(max(fitness))]
            best_fitness = max(fitness)
        track_best_state.append(best_state)
        track_best_fitness.append(best_fitness)
        population2 = []  # new population
        for _ in range(population_size):
            # select two parents
            # apply try except to handle total weight = 0
            temp_weights = [i+1 for i in fitness]
            parent1, parent2 = random.choices(
                population, weights=temp_weights, k=2)
            # create a child from the two parents
            child = reproduce(parent1, parent2)
            # mutate the child
            if rand() <= 0.5:
                child = mutate(child, mutation_rate)
            # add the child to the next generation
            population2.append(child)
        # replace the old population with the new population
        population = population2
        # increment the current generation
        curr_gen += 1
        # checkpoint
        print("Generation = {} ===> Fitness = {}".format(curr_gen, best_fitness))
    # print total time taken
    print("Total time taken: {} seconds".format(time.time() - start))
    # plot the best fitness values
    plot_fitness(track_best_fitness, curr_gen)
    # return the best state and its fitness
    return best_state, best_fitness, track_best_fitness, curr_gen


def create_state(graph):
    # code to create a random state
    state = []
    for i in range(len(graph)):
        state.append(randint(0, 2))
    return state


def fitness_function(state, graph):
    # code to calculate the fitness function
    fitness = 0
    for i in range(len(graph)):
        same = False
        for j in range(len(graph)):
            if i != j and graph[i][j] and state[i] == state[j]:
                same = True
        if not same:
            fitness += 1
    return fitness


def mutate(child, mutation_rate):
    # code to mutate the child, with a probability of mutation_rate
    lst = [0, 1, 2]
    for i in range(len(child)):
        if rand() < mutation_rate:
            child[i] = random.choice(lst)
    return child


def plot_fitness(track_best_fitness, curr_gen):
    x = np.arange(1, curr_gen+1)
    y = np.array(track_best_fitness)
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness vs. Generation')
    plt.savefig('fitness.png')
    plt.close()


def color_graph(best_state, edges):
    # code to color the graph using the best state
    # print(len(edges))
    colors = ['red', 'green', 'blue']
    G = nx.Graph()
    G.add_edges_from(edges)
    color_map = []
    # print(len(best_state))
    for i in range(len(best_state)):
        color_map.append(colors[best_state[i]])
    # print(len(color_map))
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.savefig('graph_color_{}.png'.format(len(edges)))
    plt.close()


def visualize_graph(edges):
    # code to visualize the graph using networkx library
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True)
    plt.savefig('graph_{}.png'.format(len(edges)))
    plt.close()


def create_adjacency_matrix(edges):
    # code to create adjacency matrix
    graph = [[0 for _ in range(50)] for _ in range(50)]
    for i in range(len(edges)):
        graph[edges[i][0]][edges[i][1]] = 1
        graph[edges[i][1]][edges[i][0]] = 1
    return graph


# def plot_fitness_values(fitness_values, edges, curr_gen):
#     x = np.arange(1, curr_gen+1)
#     for i in range(len(fitness_values)):
#         y = np.array(fitness_values[i])
#         plt.plot(x, y, label='{}'.format(edges[i]))
#     plt.xlabel('Generation')
#     plt.ylabel('Fitness')
#     plt.title('Fitness vs. Generation')
#     plt.legend()
#     plt.savefig('fitness_output.png')
#     plt.close()


def main():
    gc = Graph_Creator()
    # question 1
    # lst = [100, 200, 300, 400, 500]  # list of number of edges
    # plot_x = []
    # params
    population_size = 100
    mutation_rate = 1/50
    max_gen = 5000
    # select the best fitness value of particular edge after 10 runs
    # for edges in lst:
    #     print("\n===========================================\n")
    #     print("Total edges = ", edges)
    #     overall_best_state = []
    #     overall_best_fitness = 0
    #     overall_fitness_arr = []
    #     for _ in range(1):
    #         # create random edge list and generate graph
    #         edge_lst = gc.CreateGraphWithRandomEdges(edges)
    #         graph = create_adjacency_matrix(edge_lst)
    #         # mutation_rate = 1/edges
    #         # implement genetic algorithm
    #         curr_best_state, curr_best_fitness, curr_fitness_arr, curr_gen = genetic_algorithm(
    #             graph, population_size, mutation_rate, max_gen, edges)
    #         # select the best fitness value
    #         if curr_best_fitness > overall_best_fitness:
    #             overall_best_state = curr_best_state
    #             overall_best_fitness = curr_best_fitness
    #             overall_fitness_arr = curr_fitness_arr
    #         print("===> Current fitness = ", curr_best_fitness)
    #     plot_x.append(overall_fitness_arr)
    #     print('Best state: {}'.format(overall_best_state))
    #     print('Best fitness: {}'.format(overall_best_fitness))
    #     print("\n===========================================\n")
    #     plot_fitness(overall_fitness_arr, curr_gen, edges)
    # plot fitness values for different edges
    # plot_fitness_values(plot_x, lst, curr_gen, name="output")

    # actual test case
    edges = gc.ReadGraphfromCSVfile("50")
    print("Total edges = ", len(edges))
    visualize_graph(edges)
    # create adjacency matrix
    graph = create_adjacency_matrix(edges)
    # implement genetic algorithm
    best_state, best_fitness, track_best_fitness, curr_gen = genetic_algorithm(
        graph, population_size, mutation_rate, max_gen)
    print('Best state: {}'.format(best_state))
    print('Best fitness: {}'.format(best_fitness))
    # color the graph using the best state
    # color_graph(best_state, edges)
# end of main


if __name__ == '__main__':
    main()
