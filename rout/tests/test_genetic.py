import numpy as np
import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



from rout.genetic_algorithm import (
    CarGenome,
    GeneticAlgorithm,
    clamp_vector,
    fitness_of,
    simulate_car,
    GENE_BOUNDS,
    NN_WEIGHTS_SIZE,
    COLLISION_PENALTY
)


@pytest.fixture
def ga_instance(mocker):
    """Retorna uma instância básica do GeneticAlgorithm para testes de métodos internos."""
    # precisa usar uma seed fixa para que os resultados sejam reproduzíveis
    return GeneticAlgorithm(
        population_size=10, 
        mutation_rate=0.5, 
        crossover_rate=0.5, 
        seed=42, # Seed fixa
        goal_x=10.0,
        obstacles=[]
    )

@pytest.fixture
def dummy_genome():
    """Cria um CarGenome com valores típicos para testes."""
    return CarGenome(
        wheel_radius=0.2,
        motor_power=200.0,
        fuel_tank=50.0,
        drag=0.5,
        grip=1.5,
        steering=0.5,
        nn_weights=np.zeros(NN_WEIGHTS_SIZE)
    )


# Testa a inicialização da população do GA
def test_ga_init_population_size(ga_instance):
    """Verifica se a população inicial é criada com o tamanho e forma corretos."""
    pop = ga_instance._init_population()
    
    # Deve ser (population_size, num_genes)
    expected_shape = (ga_instance.population_size, GENE_BOUNDS.shape[0])
    assert pop.shape == expected_shape

def test_ga_init_population_bounds(ga_instance):
    """Verifica se todos os genes da população inicial estão dentro dos limites."""
    pop = ga_instance._init_population()
    
    # Verifica se o mínimo de toda a população é >= ao mínimo do GENE_BOUNDS
    assert np.all(pop >= GENE_BOUNDS[:, 0])
    # Verifica se o máximo de toda a população é <= ao máximo do GENE_BOUNDS
    assert np.all(pop <= GENE_BOUNDS[:, 1])


def test_car_genome_serialization_integrity(dummy_genome):
    """Verifica se a serialização para vetor e desserialização mantém os valores originais."""
    vector = dummy_genome.to_vector()
    restored_genome = CarGenome.from_vector(vector, NN_WEIGHTS_SIZE)

    assert restored_genome.wheel_radius == dummy_genome.wheel_radius
    assert restored_genome.motor_power == dummy_genome.motor_power
    assert restored_genome.fuel_tank == dummy_genome.fuel_tank

    assert np.allclose(restored_genome.nn_weights, dummy_genome.nn_weights)

def test_clamp_vector_min_max():
    """Verifica se os valores fora dos limites são corretamente restringidos (clamped)."""
    v_out_of_bounds = np.array([0.0] * GENE_BOUNDS.shape[0])
    v_out_of_bounds[0] = -1.0 # Abaixo do mínimo (0.05)
    v_out_of_bounds[1] = 1000.0 # Acima do máximo (400.0)

    clamped_v = clamp_vector(v_out_of_bounds)

    assert np.isclose(clamped_v[0], GENE_BOUNDS[0, 0]) # Deve ser 0.05
    assert np.isclose(clamped_v[1], GENE_BOUNDS[1, 1]) # Deve ser 400.0

def test_clamp_vector_in_bounds():
    """Verifica se um vetor dentro dos limites não é alterado."""
    # Usa a média dos limites como um vetor dentro dos limites
    v_in_bounds = np.mean(GENE_BOUNDS, axis=1)

    clamped_v = clamp_vector(v_in_bounds)

    assert np.allclose(clamped_v, v_in_bounds)


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

def test_ga_crossover_no_crossover(ga_instance, mocker):
    """Verifica se com crossover_rate=0.0, os filhos são cópias exatas dos pais."""
    ga_instance.crossover_rate = 0.0
    parent_a = np.ones(GENE_BOUNDS.shape[0]) * 1.0
    parent_b = np.ones(GENE_BOUNDS.shape[0]) * 4.0

    mock_rng = mocker.MagicMock()
    mock_rng.random.side_effect = [0.1, np.zeros(GENE_BOUNDS.shape[0])] # Retorna 0.1 para que 0.1 > crossover_rate (0.0)
    ga_instance.rng = mock_rng
    
    child1, child2 = ga_instance._crossover(parent_a, parent_b)
    
    assert np.allclose(child1, parent_a)
    assert np.allclose(child2, parent_b)

def test_ga_crossover_full_blending(ga_instance, mocker):
    """Verifica se o crossover de ponto único (blending) ocorre corretamente."""
    ga_instance.crossover_rate = 1.0
    parent_a = np.ones(GENE_BOUNDS.shape[0]) * 10.0
    parent_b = np.ones(GENE_BOUNDS.shape[0]) * 2.0  
    
    mock_rng = mocker.MagicMock()
    mock_rng.random.side_effect = [0.0, np.full(GENE_BOUNDS.shape[0], 0.5)]
    ga_instance.rng = mock_rng
    
    child1, child2 = ga_instance._crossover(parent_a, parent_b)
    
    expected_mean = (parent_a + parent_b) / 2.0
    
    assert np.allclose(child1, expected_mean)
    assert np.allclose(child2, expected_mean)

def test_ga_mutate_no_mutation(ga_instance, mocker):
    """Verifica se com mutation_rate=0.0, o indivíduo permanece inalterado."""
    ga_instance.mutation_rate = 0.0
    individual = np.mean(GENE_BOUNDS, axis=1)
    
    mock_rng = mocker.MagicMock()
    mock_rng.random.return_value = np.full(GENE_BOUNDS.shape[0], 0.1)
    ga_instance.rng = mock_rng
    
    mutated_individual = ga_instance._mutate(individual.copy())
    
    assert np.allclose(mutated_individual, individual)

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

def test_ga_tournament_selection(ga_instance, mocker):
    """Testa se o torneio seleciona corretamente o indivíduo com o maior fitness."""
    ga_instance.population_size = 5 
    ga_instance.tournament_size = 3
    
    # Mock do gerador aleatório para que ele sempre sorteie os mesmos índices [0, 2, 4]
    mock_rng = mocker.MagicMock()
    mock_rng.integers.return_value = np.array([0, 2, 4])
    ga_instance.rng = mock_rng
    fitness = np.array([10.0, 5.0, 20.0, 8.0, 1.0])
    selected_index = ga_instance._tournament_select(fitness)
    assert selected_index == 2

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

def test_simulate_car_reaches_goal_simple():
    """Verifica se um carro otimista (alta potência, baixo arrasto) atinge o objetivo."""
    optimistic_genome = CarGenome(
        wheel_radius=0.5,
        motor_power=500.0, # Muito forte
        fuel_tank=100.0,
        drag=0.0, # Sem arrasto
        grip=1.0,
        steering=0.0, # Linha reta
        nn_weights=np.array([]) 
    )
    goal_x = 20.0 # Objetivo perto
    max_steps = 100 # Poucos passos
    
    x_reached, collision, _ = simulate_car(optimistic_genome, goal_x=goal_x, max_steps=max_steps)
    
    assert x_reached >= goal_x
    assert collision is False

def test_simulate_car_collision_penalty():
    """Verifica se o carro colide com um obstáculo circular em seu caminho."""
    obstacle = [(10.0, 0.0, 1.0)]
    
    # Carro com potência razoável, indo em linha reta (steering=0.0)
    test_genome = CarGenome(
        wheel_radius=0.5,
        motor_power=500.0,
        fuel_tank=10.0,
        drag=0.1,
        grip=1.0,
        steering=0.0,
        nn_weights=np.array([])
    )
    
    x_reached, collision, _ = simulate_car(test_genome, goal_x=100.0, obstacles=obstacle, max_steps=500)
    
    # A colisão deve ocorrer antes de chegar ao objetivo
    assert collision is True
    assert x_reached < 10.0 # Deve parar pouco antes do obstáculo em

def test_fitness_time_and_collision_penalty(mocker):
    """Verifica se a função fitness aplica corretamente a penalidade de tempo e colisão."""
    # Mock da função simulate_car para resultados conhecidos
   
    mocker.patch(
        'rout.genetic_algorithm.simulate_car', 
        return_value=(50.0, True, [(0,0)] * 50) 
    )
    
    goal_x = 100.0
    time_cost = 0.02 
    
    expected_fitness = 50.0 - (time_cost * 49) - COLLISION_PENALTY
    
    result_fitness = fitness_of(CarGenome.from_vector(np.zeros(GENE_BOUNDS.shape[0]), NN_WEIGHTS_SIZE), goal_x)
    
    assert np.isclose(result_fitness, expected_fitness)


def test_simulate_car_with_neural_network(dummy_genome):
    """
    Testa o caminho 'else' dentro de simulate_car, onde a Rede Neural 
    decide a aceleração e direção, em vez da lógica fixa.
    """
    # Preenche os pesos com valores aleatórios (mas determinísticos para o teste)
    rng = np.random.default_rng(42)
    dummy_genome.nn_weights = rng.standard_normal(NN_WEIGHTS_SIZE)
    
    # Roda apenas 1 passo para verificar se não quebra e se atualiza o estado
    x_start, y_start = 0.0, 0.0
    x_end, collision, traj = simulate_car(dummy_genome, goal_x=10.0, max_steps=1)
    
    # Se a rede neural funcionou, o carro deve ter se movido (ou tentado)
    # e a trajetória deve ter tamanho 2 (início + 1 passo)
    assert len(traj) == 2
    assert not collision


def test_ga_run_full_loop_with_callback():
    """Roda o GA por poucas gerações para garantir que o loop principal não trava."""
    ga = GeneticAlgorithm(population_size=4, goal_x=10.0)
    
    mock_callback_data = {'called': False}
    def on_gen(gen, pop, fit):
        mock_callback_data['called'] = True
        mock_callback_data['last_gen'] = gen

    best, fit = ga.run(generations=2, verbose=False, on_generation=on_gen)
    
    assert best is not None
    assert mock_callback_data['called'] is True
    assert mock_callback_data['last_gen'] == 1 # 0 e 1 (2 gerações)


def test_potential_field_generation_and_sampling():
    """
    Cobre as funções 'compute_potential_field' e 'sample_potential'.
    Verifica se o algoritmo de Dijkstra cria um gradiente onde o objetivo é 0
    e longe do objetivo é maior que 0.
    """
    # Cria um campo pequeno (0 a 10) com resolução 1.0
    goal_pos = (5.0, 5.0)
    # Adiciona um obstáculo no (2,2) para garantir que o loop de obstáculos roda
    obstacles = [(2.0, 2.0, 0.5)]
    
    field = compute_potential_field(
        obstacles=obstacles,
        goal_pos=goal_pos,
        x_min=0, x_max=10,
        y_min=0, y_max=10,
        resolution=1.0
    )
    
    # 1. Verifica se o valor exato no objetivo é 0.0 (ou muito próximo)
    val_at_goal = sample_potential(field, 5.0, 5.0)
    assert math.isclose(val_at_goal, 0.0, abs_tol=1e-3)
    
    # 2. Verifica se um ponto longe tem custo maior que o objetivo
    val_far = sample_potential(field, 9.0, 9.0)
    assert val_far > 1.0
    
    # 3. Verifica se amostrar fora do mapa retorna infinito
    val_out = sample_potential(field, -50.0, -50.0)
    assert val_out == float('inf')


def test_segment_obstacle_collision_and_logic(dummy_genome):
    """
    Cobre a lógica de colisão específica para SEGMENTOS ('seg'),
    que é diferente da lógica de círculos.
    """
    # Cria uma parede vertical na posição X=5.0
    # Formato: ('seg', x1, y1, x2, y2, raio/espessura)
    wall_obstacle = [('seg', 5.0, -10.0, 5.0, 10.0, 0.1)]
    
    # Configura o carro para andar reto e rápido
    dummy_genome.steering = 0.0
    dummy_genome.motor_power = 200.0
    
    # Roda a simulação
    x_reached, collision, _ = simulate_car(
        dummy_genome, 
        goal_x=10.0, 
        obstacles=wall_obstacle, 
        max_steps=100
    )
    
    # O carro deve ter batido (collision=True) e não pode ter passado de X=5.5
    assert collision is True
    assert x_reached < 5.5


def test_ray_math_miss_scenarios():
    """
    Cobre as ramificações matemáticas onde os raios NÃO acertam os obstáculos.
    (Funções _ray_circle_t e _ray_capsule_t retornando None)
    """
    # 1. Teste de Raio vs Círculo (Erra o alvo)
    # Raio sai de (0,0) apontando para cima (0,1)
    # Círculo está na direita (10,0)
    t_circle = _ray_circle_t(
        Ox=0.0, Oy=0.0, 
        dxr=0.0, dyr=1.0,  # Aponta para Y
        cx=10.0, cy=0.0, cr=1.0, 
        car_radius=0.0
    )
    assert t_circle is None

    # 2. Teste de Raio vs Cápsula (Erra o alvo)
    # Raio aponta para cima, segmento está na direita verticalmente
    t_capsule = _ray_capsule_t(
        Ox=0.0, Oy=0.0,
        dxr=0.0, dyr=1.0,
        ax=10.0, ay=-5.0, bx=10.0, by=5.0,
        seg_rad=0.5, car_radius=0.0
    )
    assert t_capsule is None