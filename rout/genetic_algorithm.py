"""genetic_algorithm.py

Um algoritmo genético simples para evoluir parâmetros de um "carro" que
deve percorrer um percurso 2D até um objetivo. A representação do indivíduo
é um vetor contínuo com parâmetros físicos do carro (raio da roda, potência,
massa, arrasto, aderência e resposta de direção).

Esta implementação é intencionalmente simples e auto-contida para facilitar
experimentos educacionais.
"""
from dataclasses import dataclass
import math
from typing import List, Tuple, Optional

import numpy as np

# use numpy's Generator for reproducible RNG and better performance
from numpy.random import default_rng


# NN architecture constants (fixed)
SENSOR_COUNT = 3
NN_INPUT = SENSOR_COUNT + 1
NN_HIDDEN = 6
NN_OUTPUT = 4
NN_WEIGHTS_SIZE = NN_HIDDEN * NN_INPUT + NN_HIDDEN + NN_OUTPUT * NN_HIDDEN + NN_OUTPUT


@dataclass
class CarGenome:
	# físicos / sensores
	wheel_radius: float  # metros
	motor_power: float   # unidade arbitrária (bigger -> more thrust)
	fuel_tank: float     # tamanho do tanque (maior -> mais massa/energia)
	drag: float          # coeficiente de arrasto (0..2)
	grip: float          # coeficiente de aderência (0.5..2.0)
	steering: float      # capacidade de virar (0..1)
	sensor_range: float  # alcance dos sensores (m)
	# rede neural (pesos flatten)
	nn_weights: np.ndarray

	def to_vector(self) -> np.ndarray:
		phys = np.array([
			self.wheel_radius,
			self.motor_power,
			self.fuel_tank,
			self.drag,
			self.grip,
			self.steering,
			self.sensor_range,
		], dtype=float)
		return np.concatenate([phys, self.nn_weights.astype(float)])

	@staticmethod
	def from_vector(v: np.ndarray, nn_weights_size: int) -> "CarGenome":
		phys = v[:7]
		weights = v[7:7+nn_weights_size]
		return CarGenome(
			wheel_radius=float(phys[0]),
			motor_power=float(phys[1]),
			fuel_tank=float(phys[2]),
			drag=float(phys[3]),
			grip=float(phys[4]),
			steering=float(phys[5]),
			sensor_range=float(phys[6]),
			nn_weights=weights.copy(),
		)


# limites para inicialização e mutação (min, max)
_phys_bounds = np.array([
	[0.05, 0.5],     # wheel_radius (m)
	[20.0, 400.0],   # motor_power
	[5.0, 200.0],    # fuel_tank (arbitrary volume)
	[0.0, 2.0],      # drag
	[0.5, 2.0],      # grip
	[0.0, 1.0],      # steering
	[5.0, 60.0],     # sensor_range (m)
])

# weight bounds for NN parameters
_weight_bounds = np.tile(np.array([[-2.0, 2.0]]), (NN_WEIGHTS_SIZE, 1))

GENE_BOUNDS = np.vstack([_phys_bounds, _weight_bounds])

# penalidade aplicada quando um carro colide (obstáculo ou borda)
COLLISION_PENALTY = 5.0


def clamp_vector(v: np.ndarray) -> np.ndarray:
	"""Clamp a vector to the predefined `GENE_BOUNDS` using vectorized numpy ops."""
	low = GENE_BOUNDS[:, 0]
	high = GENE_BOUNDS[:, 1]
	return np.clip(v, low, high).astype(float)


def simulate_car(genome: CarGenome,
				 goal_x: float = 100.0,
				 obstacles: Optional[List[Tuple[float, float, float]]] = None,
				 dt: float = 0.1,
				 max_steps: int = 1000,
				 env_bounds: Tuple[float, float] = (-25.0, 25.0)) -> Tuple[float, bool, List[Tuple[float, float]]]:
	"""
	Simula o carro começando em (0,0) com orientação 0 (eixo +x). O objetivo
	fica em (goal_x, 0). Retorna (x_reached, collision_flag, trajectory).

	A dinâmica é deliberadamente simples: aceleração proporcionada pela
	potência do motor dividida pela massa, atenuada por arrasto proporcional a v^2.
	A direção é ajustada gradualmente em direção ao objetivo baseado em steering.
	"""
	if obstacles is None:
		obstacles = []

	x, y = 0.0, 0.0
	heading = 0.0  # rad
	v = 0.0
	traj = [(x, y)]

	# pequenos parâmetros derivados
	car_radius = 0.5 * genome.wheel_radius + 0.2  # aproximação do tamanho do veículo

	# massa agora é derivada: massa_base + motor_contrib + fuel_contrib
	mass_base = 50.0
	motor_mass_coeff = 0.05
	fuel_mass_coeff = 0.2
	mass = mass_base + motor_mass_coeff * genome.motor_power + fuel_mass_coeff * genome.fuel_tank

	# sensores fixos (em relação ao heading): esquerda, centro, direita
	sensor_angles = [-math.pi / 6, 0.0, math.pi / 6]

	# loop principal da simulação: cada iteração representa dt segundos
	for step in range(max_steps):
		# direção desejada para o objetivo (usada apenas como referência)
		dx = goal_x - x
		dy = -y
		dist_to_goal = math.hypot(dx, dy)
		if dist_to_goal < 1e-3:
			return x, False, traj

		# sensores: calcular distância mínima em cada sensor
		sensor_readings = []
		for sa in sensor_angles:
			ray_angle = heading + sa
			sr = genome.sensor_range
			min_dist = sr
			# checar paredes (env_bounds)
			if abs(math.sin(ray_angle)) > 1e-8:
				if math.sin(ray_angle) > 0:
					dist_to_top = (env_bounds[1] - y) / math.sin(ray_angle)
					if 0 <= dist_to_top < min_dist:
						min_dist = dist_to_top
				else:
					dist_to_bottom = (env_bounds[0] - y) / math.sin(ray_angle)
					if 0 <= dist_to_bottom < min_dist:
						min_dist = dist_to_bottom

			# helpers: ray-circle intersection (returns smallest t>=0 or None)
			def _ray_circle_t(ox, oy, orad):
				dxr = math.cos(ray_angle)
				dyr = math.sin(ray_angle)
				fx = x - ox
				fy = y - oy
				a = dxr*dxr + dyr*dyr
				b = 2*(fx*dxr + fy*dyr)
				c = fx*fx + fy*fy - (orad + car_radius)**2
				disc = b*b - 4*a*c
				if disc < 0:
					return None
				t1 = (-b - math.sqrt(disc)) / (2*a)
				t2 = (-b + math.sqrt(disc)) / (2*a)
				cands = [t for t in (t1, t2) if t >= 0]
				if not cands:
					return None
				return min(cands)

			# helper: ray-capsule (segment with radius) intersection
			def _ray_capsule_t(ax, ay, bx, by, seg_rad):
				# Solve ray (O + t d) vs capsule around segment A->B with radius = seg_rad + car_radius
				Ox, Oy = x, y
				dx = math.cos(ray_angle)
				dy = math.sin(ray_angle)
				R = seg_rad + car_radius
				# vector u = B - A
				u_x = bx - ax
				u_y = by - ay
				uu = u_x*u_x + u_y*u_y
				# check infinite cylinder around line (projected) by solving quadratic
				# w0 = O - A
				w0x = Ox - ax
				w0y = Oy - ay
				ud = u_x*dx + u_y*dy
				uw0 = u_x*w0x + u_y*w0y
				# v = d - u*(ud/uu)
				if uu == 0:
					# degenerate segment -> treat as circle at A
					return _ray_circle_t(ax, ay, seg_rad)
				v_x = dx - u_x*(ud/uu)
				v_y = dy - u_y*(ud/uu)
				q_x = w0x - u_x*(uw0/uu)
				q_y = w0y - u_y*(uw0/uu)
				A_q = v_x*v_x + v_y*v_y
				B_q = 2*(q_x*v_x + q_y*v_y)
				C_q = q_x*q_x + q_y*q_y - R*R
				solutions = []
				if abs(A_q) > 1e-12:
					discq = B_q*B_q - 4*A_q*C_q
					if discq >= 0:
						t1 = (-B_q - math.sqrt(discq)) / (2*A_q)
						t2 = (-B_q + math.sqrt(discq)) / (2*A_q)
						for t in (t1, t2):
							if t >= 0:
								# compute s = (uw0 + t*ud)/uu
								s = (uw0 + t*ud) / uu
								if 0.0 <= s <= 1.0:
									solutions.append(t)
				# also check end-circles A and B
				for (cx, cy) in ((ax, ay), (bx, by)):
					tc = _ray_circle_t(cx, cy, seg_rad)
					if tc is not None:
						solutions.append(tc)
				if not solutions:
					return None
				return min(solutions)

			# checar obstáculos (agora suportamos círculos e segmentos/cápsulas)
			for o in obstacles:
				if isinstance(o, tuple) and len(o) == 3:
					ox, oy, orad = o
					t = _ray_circle_t(ox, oy, orad)
					if t is not None and 0 <= t < min_dist:
						min_dist = t
				elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
					# ('seg', x1,y1,x2,y2, radius)
					_, ax, ay, bx, by, srad = o
					t = _ray_capsule_t(ax, ay, bx, by, srad)
					if t is not None and 0 <= t < min_dist:
						min_dist = t
				else:
					# unknown format - try to interpret as circle fallback
					try:
						ox, oy, orad = o
						t = _ray_circle_t(ox, oy, orad)
						if t is not None and 0 <= t < min_dist:
							min_dist = t
					except Exception:
						continue
			sensor_readings.append(min_dist)

		# normalize sensor readings to [0,1]
		sens_norm = np.minimum(1.0, np.array(sensor_readings, dtype=float) / float(genome.sensor_range))
		speed_norm = float(min(1.0, abs(v) / 20.0))

		# use NN to decide actions (genome must carry nn_weights attribute)
		if not hasattr(genome, 'nn_weights') or len(genome.nn_weights) != NN_WEIGHTS_SIZE:
			# fallback: acelera reto
			thrust = genome.motor_power
			accel = (thrust / mass) - genome.drag * (v ** 2)
			speed_factor = max(0.05, 1.0 - 0.2 * abs(v) / (10.0 * genome.grip))
			v += accel * dt * speed_factor
			v = max(0.0, v)
			x += v * math.cos(heading) * dt
			y += v * math.sin(heading) * dt
			traj.append((x, y))
		else:
			# fast forward pass (unpack weights)
			w = genome.nn_weights
			p = 0
			W1 = w[p:p + NN_HIDDEN * NN_INPUT].reshape((NN_HIDDEN, NN_INPUT)); p += NN_HIDDEN * NN_INPUT
			b1 = w[p:p + NN_HIDDEN]; p += NN_HIDDEN
			W2 = w[p:p + NN_OUTPUT * NN_HIDDEN].reshape((NN_OUTPUT, NN_HIDDEN)); p += NN_OUTPUT * NN_HIDDEN
			b2 = w[p:p + NN_OUTPUT]; p += NN_OUTPUT
			inp = np.concatenate([sens_norm, np.array([speed_norm], dtype=float)])
			h = np.tanh(W1.dot(inp) + b1)
			nn_out = 1.0 / (1.0 + np.exp(-(W2.dot(h) + b2)))
			accel_out, brake_out, steer_left_out, steer_right_out = map(float, nn_out)

			# map outputs to physics
			throttle = accel_out - brake_out
			thrust = max(0.0, throttle) * genome.motor_power
			braking = max(0.0, -throttle) + brake_out
			accel = (thrust / mass) - genome.drag * (v ** 2)
			speed_factor = max(0.05, 1.0 - 0.2 * abs(v) / (10.0 * genome.grip))
			v += accel * dt * speed_factor
			v -= braking * 0.5 * dt
			v = max(0.0, v)

			steer_signal = steer_right_out - steer_left_out
			max_turn = genome.steering * 0.4
			heading += float(np.clip(steer_signal, -1.0, 1.0)) * max_turn

			x += v * math.cos(heading) * dt
			y += v * math.sin(heading) * dt
			traj.append((x, y))

		# checa colisões com obstáculos (suporta círculos e segmentos/cápsulas)
		for o in obstacles:
			if isinstance(o, tuple) and len(o) == 3:
				ox, oy, orad = o
				if math.hypot(x - ox, y - oy) <= (orad + car_radius):
					return x, True, traj
			elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
				_, ax, ay, bx, by, srad = o
				ux = bx - ax
				uy = by - ay
				l2 = ux * ux + uy * uy
				if l2 == 0:
					dist = math.hypot(x - ax, y - ay)
				else:
					t = ((x - ax) * ux + (y - ay) * uy) / l2
					t = max(0.0, min(1.0, t))
					cx = ax + ux * t
					cy = ay + uy * t
					dist = math.hypot(x - cx, y - cy)
				if dist <= (srad + car_radius):
					return x, True, traj
			else:
				# fallback: try interpret as circle
				try:
					ox, oy, orad = o
					if math.hypot(x - ox, y - oy) <= (orad + car_radius):
						return x, True, traj
				except Exception:
					continue

		# chegou ao objetivo
		if x >= goal_x:
			return x, False, traj

		# checar limites verticais - colisão com parede
		if y < env_bounds[0] or y > env_bounds[1]:
			return x, True, traj

	# se esgotou os passos
	return x, False, traj


def fitness_of(genome: CarGenome, goal_x: float = 100.0, obstacles=None) -> float:
	"""Calcula a aptidão (fitness) que queremos maximizar.

	Estratégia: fitness linear com a distância X alcançada (limitada a goal_x),
	com penalidade proporcional ao tempo (número de passos) para favorecer chegar
	mais rápido. Não aplicamos penalidade fixa por colisão aqui — colisões reduzem
	x_reached e assim já prejudicam o fitness.

	Retorna um float (maior = melhor).
	"""
	x_reached, collision, traj = simulate_car(genome, goal_x=goal_x, obstacles=obstacles)
	# garante que o valor de x não exceda o objetivo
	x_used = min(float(x_reached), float(goal_x))
	# penalidade por tempo: cada passo custa "time_cost" unidades de fitness
	steps = max(0, len(traj) - 1)
	time_cost = 0.02  # metros de penalidade por passo (ajustável)
	fitness = x_used - (time_cost * steps)
	# aplica penalidade adicional se houve colisão (com obstáculos ou bordas)
	if collision:
		fitness -= COLLISION_PENALTY
	return float(fitness)


class GeneticAlgorithm:
	def __init__(self,
				 population_size: int = 50,
				 mutation_rate: float = 0.2,
				 crossover_rate: float = 0.8,
				 tournament_size: int = 3,
				 seed: Optional[int] = None,
				 goal_x: float = 100.0,
				 obstacles: Optional[List[Tuple[float, float, float]]] = None):
		self.population_size = int(population_size)
		self.mutation_rate = float(mutation_rate)
		self.crossover_rate = float(crossover_rate)
		self.tournament_size = int(tournament_size)
		self.goal_x = float(goal_x)
		self.obstacles = list(obstacles) if obstacles is not None else []

		# random generator
		self.rng = default_rng(seed)

		# population stored as numpy arrays
		self.pop = self._init_population()

	def _init_population(self) -> np.ndarray:
		low = GENE_BOUNDS[:, 0]
		high = GENE_BOUNDS[:, 1]
		# sample uniformly per-gene using broadcasting
		samples = self.rng.random((self.population_size, low.size))
		return (low + samples * (high - low)).astype(float)

	def _evaluate_population(self) -> np.ndarray:
		"""Avalia a população retornando um array de fitness.

		Interrompe a avaliação da geração assim que pelo menos 2 carros alcançarem
		a linha de chegada (self.goal_x). Para compatibilidade não alteramos
		a função pública `fitness_of` — aqui a avaliação é feita inline chamando
		`simulate_car` diretamente, para permitir detectar chegada.
		"""
		fitness = np.zeros(self.population_size, dtype=float)
		reached_count = 0
		for i in range(self.population_size):
			genome = CarGenome.from_vector(self.pop[i], NN_WEIGHTS_SIZE)
			x_reached, collision, traj = simulate_car(genome, goal_x=self.goal_x, obstacles=self.obstacles)
			x_used = min(float(x_reached), float(self.goal_x))
			steps = max(0, len(traj) - 1)
			time_cost = 0.02
			fitness[i] = x_used - (time_cost * steps)
			if collision:
				fitness[i] -= COLLISION_PENALTY
			if x_reached >= self.goal_x:
				reached_count += 1
				if reached_count >= 2:
					if i + 1 < self.population_size:
						fitness[i + 1:] = 0.0
					break
		return fitness

	def _tournament_select(self, fitness: np.ndarray) -> int:
		# retorna índice de indivíduo selecionado (using RNG)
		contestants = self.rng.integers(0, self.population_size, size=self.tournament_size)
		best = int(contestants[0])
		for c in contestants:
			if fitness[int(c)] > fitness[best]:
				best = int(c)
		return best

	def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if self.rng.random() > self.crossover_rate:
			return parent_a.copy(), parent_b.copy()
		alpha = self.rng.random(parent_a.shape)
		child1 = alpha * parent_a + (1.0 - alpha) * parent_b
		child2 = (1.0 - alpha) * parent_a + alpha * parent_b
		return child1, child2

	def _mutate(self, individual: np.ndarray) -> np.ndarray:
		# gaussian perturbation where a mask indicates mutated genes
		mask = self.rng.random(individual.shape) < self.mutation_rate
		if mask.any():
			lows = GENE_BOUNDS[:, 0]
			highs = GENE_BOUNDS[:, 1]
			scales = (highs - lows) * 0.08
			perturb = self.rng.normal(loc=0.0, scale=scales) * mask
			individual = individual + perturb
		return clamp_vector(individual)

	def run(self, generations: int = 100, verbose: bool = True, on_generation=None) -> Tuple[Optional[CarGenome], float]:
		best_genome: Optional[CarGenome] = None
		best_fitness = -1e9

		for gen in range(int(generations)):
			fitness = self._evaluate_population()
			if on_generation is not None:
				try:
					on_generation(gen, self.pop.copy(), fitness.copy())
				except Exception:
					pass

			# keep best
			idx_best = int(np.argmax(fitness))
			if fitness[idx_best] > best_fitness:
				best_fitness = float(fitness[idx_best])
				best_genome = CarGenome.from_vector(self.pop[idx_best].copy(), NN_WEIGHTS_SIZE)

			if verbose and (gen % max(1, int(generations) // 10) == 0 or gen == int(generations) - 1):
				print(f"Generation {gen:4d}: best fitness = {best_fitness:.3f}")

			# create new population
			new_pop = np.empty_like(self.pop)
			i = 0
			while i < self.population_size:
				a_idx = self._tournament_select(fitness)
				b_idx = self._tournament_select(fitness)
				a = self.pop[a_idx]
				b = self.pop[b_idx]
				c1, c2 = self._crossover(a, b)
				c1 = self._mutate(c1)
				c2 = self._mutate(c2)
				new_pop[i] = c1
				if i + 1 < self.population_size:
					new_pop[i + 1] = c2
				i += 2

			self.pop = new_pop

		return best_genome, best_fitness


if __name__ == "__main__":
	# pequeno teste rápido
	ga = GeneticAlgorithm(population_size=50, mutation_rate=0.1, crossover_rate=0.9)
	best, fit = ga.run(generations=90)
	print("Best fitness:", fit)
	print(best)

