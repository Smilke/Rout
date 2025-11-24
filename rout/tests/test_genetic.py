import numpy as np
import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from genetic_algorithm import GeneticAlgorithm, CarGenome, NN_WEIGHTS_SIZE, GENE_BOUNDS

# Testa a inicialização da população do GA
def test_ga_initialization():
    ga = GeneticAlgorithm(
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=2,
        goal_x=100,
        obstacles=[]
    )

    assert len(ga.pop) == 10
    assert ga.pop.shape == (10, 6 + NN_WEIGHTS_SIZE)

# Testa se a mutação altera efetivamente um indivíduo
def test_ga_mutation_changes_individual():
    ga = GeneticAlgorithm(10, 0.5, 0.8, 2, goal_x=100, obstacles=[])
    vec = np.zeros(6 + NN_WEIGHTS_SIZE)
    mutated = ga._mutate(vec.copy())
    assert not np.array_equal(vec, mutated)

# Testa se o crossover retorna indivíduos com o mesmo formato dos pais
def test_ga_crossover():
    ga = GeneticAlgorithm(10, 0.1, 0.8, 2, goal_x=100, obstacles=[])
    p1 = np.ones(NN_WEIGHTS_SIZE)
    p2 = np.zeros(NN_WEIGHTS_SIZE)
    c1, c2 = ga._crossover(p1, p2)
    assert c1.shape == p1.shape
    assert c2.shape == p2.shape

# Testa se a população inicial possui diversidade (não é toda igual)
def test_initial_population_diversity():
    ga = GeneticAlgorithm(20, 0.1, 0.8, 2, goal_x=100, obstacles=[])
    unique_individuals = np.unique(ga.pop, axis=0)
    assert len(unique_individuals) > 1

# Testa se a mutação mantém os genes dentro dos limites definidos
def test_mutation_within_bounds():
    ga = GeneticAlgorithm(10, 1.0, 0.8, 2, goal_x=100, obstacles=[])
    vec = np.zeros(6 + NN_WEIGHTS_SIZE)
    mutated = ga._mutate(vec.copy())
    lows = GENE_BOUNDS[:, 0]
    highs = GENE_BOUNDS[:, 1]
    assert np.all(mutated >= lows)
    assert np.all(mutated <= highs)

# Testa se o crossover realmente gera filhos diferentes dos pais
def test_crossover_produces_different_children():
    ga = GeneticAlgorithm(10, 0.1, 1.0, 2, goal_x=100, obstacles=[])  # crossover rate=1
    p1 = np.ones(6 + NN_WEIGHTS_SIZE)
    p2 = np.zeros(6 + NN_WEIGHTS_SIZE)
    c1, c2 = ga._crossover(p1, p2)
    assert not np.array_equal(c1, c2)
    assert not np.array_equal(c1, p1) or not np.array_equal(c2, p2)

# Testa se a seleção por torneio retorna um indivíduo válido da população
def test_tournament_selection_returns_individual():
    ga = GeneticAlgorithm(10, 0.1, 0.8, 3, goal_x=100, obstacles=[])
    fitness = ga._evaluate_population()  
    idx = ga._tournament_select(fitness)
    selected = ga.pop[idx]  
    assert selected.shape == (6 + NN_WEIGHTS_SIZE,)

# Testa se a avaliação de fitness retorna valores corretos e finitos
def test_evaluate_fitness_values():
    ga = GeneticAlgorithm(5, 0.1, 0.8, 2, goal_x=100, obstacles=[])
    fitness = ga._evaluate_population()
    assert fitness.shape == (5,)
    assert np.all(np.isfinite(fitness))  

# Testa se uma geração do GA realmente atualiza a população
def test_generation_step_updates_population():
    ga = GeneticAlgorithm(10, 0.5, 0.8, 2, goal_x=100, obstacles=[])
    old_pop = ga.pop.copy()
    fitness = ga._evaluate_population()
    new_pop = []

    for _ in range(len(ga.pop)//2):
        p1_idx = ga._tournament_select(fitness)
        p2_idx = ga._tournament_select(fitness)
        p1 = ga.pop[p1_idx]
        p2 = ga.pop[p2_idx]
        c1, c2 = ga._crossover(p1, p2)
        new_pop.extend([ga._mutate(c1), ga._mutate(c2)])

    ga.pop = np.array(new_pop)
    assert not np.array_equal(old_pop, ga.pop), "A população não foi atualizada corretamente."

# Testa se a mutação com taxa zero não altera os genes que representam pesos da rede neural
def test_mutation_rate_zero_no_change():
    ga = GeneticAlgorithm(10, 0.0, 0.8, 2, goal_x=100, obstacles=[])
    vec = np.random.rand(6 + NN_WEIGHTS_SIZE)
    mutated = ga._mutate(vec.copy())
    assert np.allclose(vec[6:], mutated[6:])
