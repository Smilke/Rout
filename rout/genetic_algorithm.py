"""genetic_algorithm.py

Algoritmo genético simples para evoluir parâmetros de um "carro" que
deve percorrer um percurso 2D até um objetivo. A representação do indivíduo
é um vetor contínuo com parâmetros físicos do carro (raio da roda, potência,
massa, arrasto, aderência e resposta de direção) e os pesos de uma pequena
rede neural que governa as ações do veículo.

Esta implementação é intencionalmente simples e auto-contida para facilitar
experimentos educacionais e estudos sobre algoritmos evolutivos e controle.
"""
from dataclasses import dataclass
import math
from typing import List, Tuple, Optional
import heapq

import numpy as np

# usar o Generator do numpy para RNG reproduzível e melhor desempenho
from numpy.random import default_rng


# constantes da arquitetura da rede neural (fixas)
SENSOR_COUNT = 7
NN_INPUT = SENSOR_COUNT + 1
NN_HIDDEN = 6
NN_OUTPUT = 4
NN_WEIGHTS_SIZE = NN_HIDDEN * NN_INPUT + NN_HIDDEN + NN_OUTPUT * NN_HIDDEN + NN_OUTPUT


@dataclass
class CarGenome:
	# físicos / sensores
	wheel_radius: float  # metros
	motor_power: float   # unidade arbitrária (maior -> mais empuxo)
	fuel_tank: float     # tamanho do tanque (maior -> mais massa/energia)
	drag: float          # coeficiente de arrasto (0..2)
	grip: float          # coeficiente de aderência (0.5..2.0)
	steering: float      # capacidade de virar (0..1)
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
		], dtype=float)
		return np.concatenate([phys, self.nn_weights.astype(float)])

	@staticmethod
	def from_vector(v: np.ndarray, nn_weights_size: int) -> "CarGenome":
		phys = v[:6]
		weights = v[6:6+nn_weights_size]
		return CarGenome(
			wheel_radius=float(phys[0]),
			motor_power=float(phys[1]),
			fuel_tank=float(phys[2]),
			drag=float(phys[3]),
			grip=float(phys[4]),
			steering=float(phys[5]),
			nn_weights=weights.copy(),
		)


# limites para inicialização e mutação (min, max)
_phys_bounds = np.array([
	[0.05, 0.5],     # raio_roda (m)
	[20.0, 400.0],   # potencia_motor
	[5.0, 200.0],    # tanque_combustivel (volume arbitrário)
	[0.0, 2.0],      # arrasto
	[0.5, 2.0],      # aderencia (grip)
	[0.0, 1.0],      # responsividade_direcao
])

# limites dos pesos da rede neural
_weight_bounds = np.tile(np.array([[-2.0, 2.0]]), (NN_WEIGHTS_SIZE, 1))

GENE_BOUNDS = np.vstack([_phys_bounds, _weight_bounds])

# penalidade aplicada quando um carro colide (obstáculo ou borda)
COLLISION_PENALTY = 5.0


def clamp_vector(v: np.ndarray) -> np.ndarray:
	"""Restringe um vetor aos limites definidos em `GENE_BOUNDS`.

	Opera de forma vetorizada com numpy para cortar cada gene ao seu
	intervalo [min, max] correspondente.
	"""
	low = GENE_BOUNDS[:, 0]
	high = GENE_BOUNDS[:, 1]
	return np.clip(v, low, high).astype(float)


def simulate_car(genome: CarGenome,
				 goal_x: float = 100.0,
				 obstacles: Optional[List[Tuple[float, float, float]]] = None,
				 dt: float = 0.1,
				 max_steps: int = 1000,
				 env_bounds: Tuple[float, float] = (-25.0, 25.0),
				 env_x_bounds: Tuple[float, float] = (-10.0, 110.0)) -> Tuple[float, bool, List[Tuple[float, float]]]:
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

	# sensores fixos (em relação ao heading): gerar N sensores com o central em 0
	# sensores distribuídos simetricamente em relação ao heading; um sensor central
	# fica apontando exatamente para frente
	span = math.pi / 3.0  # cobertura total de +/- 60 graus
	sensor_angles = list(np.linspace(-span, span, SENSOR_COUNT))

	# pré-processa obstáculos em listas tipadas para evitar checagens repetidas
	circles = []  # lista de (ox, oy, orad)
	segs = []     # lista de (ax, ay, bx, by, srad)
	for o in (obstacles or []):
		if isinstance(o, tuple) and len(o) == 3:
			circles.append(o)
		elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
			_, ax, ay, bx, by, srad = o
			segs.append((ax, ay, bx, by, srad))
		else:
			try:
				ox, oy, orad = o
				circles.append((ox, oy, orad))
			except Exception:
				continue

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
			min_dist = float('inf')
			# checar paredes verticais (y) e horizontais (x) como obstáculos
			# vertical (top/bottom)
			if abs(math.sin(ray_angle)) > 1e-8:
				if math.sin(ray_angle) > 0:
					dist_to_top = (env_bounds[1] - y) / math.sin(ray_angle)
					if 0 <= dist_to_top < min_dist:
						min_dist = dist_to_top
				else:
					dist_to_bottom = (env_bounds[0] - y) / math.sin(ray_angle)
					if 0 <= dist_to_bottom < min_dist:
						min_dist = dist_to_bottom
			# horizontal (left/right)
			if abs(math.cos(ray_angle)) > 1e-8:
				if math.cos(ray_angle) > 0:
					dist_to_right = (env_x_bounds[1] - x) / math.cos(ray_angle)
					if 0 <= dist_to_right < min_dist:
						min_dist = dist_to_right
				else:
					dist_to_left = (env_x_bounds[0] - x) / math.cos(ray_angle)
					if 0 <= dist_to_left < min_dist:
						min_dist = dist_to_left

			# calcula a direção do raio (vetor unitário) apenas uma vez
			dxr = math.cos(ray_angle)
			dyr = math.sin(ray_angle)
			Ox = x
			Oy = y
			# checa interseção com círculos
			for (ox, oy, orad) in circles:
				t = _ray_circle_t(Ox, Oy, dxr, dyr, ox, oy, orad, car_radius)
				if t is not None and 0.0 <= t < min_dist:
					min_dist = t
			# checa interseção com segmentos (cápsulas)
			for (ax, ay, bx, by, srad) in segs:
				t = _ray_capsule_t(Ox, Oy, dxr, dyr, ax, ay, bx, by, srad, car_radius)
				if t is not None and 0.0 <= t < min_dist:
					min_dist = t
			sensor_readings.append(min_dist)

			# normaliza leituras dos sensores para (0,1], valor maior = obstáculo mais próximo
		# usa 1/(1+dist) para manter monotonicidade (distância grande -> valor pequeno)
		sens_norm = 1.0 / (1.0 + np.array(sensor_readings, dtype=float))
		speed_norm = float(min(1.0, abs(v) / 20.0))

			# usa a rede neural do genoma para decidir ações (se disponível)
		if not hasattr(genome, 'nn_weights') or len(genome.nn_weights) != NN_WEIGHTS_SIZE:
				# comportamento alternativo simples: acelera em linha reta
			thrust = genome.motor_power
			accel = (thrust / mass) - genome.drag * (v ** 2)
			speed_factor = max(0.05, 1.0 - 0.2 * abs(v) / (10.0 * genome.grip))
			v += accel * dt * speed_factor
			v = max(0.0, v)
			x += v * math.cos(heading) * dt
			y += v * math.sin(heading) * dt
			traj.append((x, y))
		else:
			# passagem direta pela rede neural: desempacota pesos e calcula saída
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

			# mapeia saídas da RN para parâmetros físicos (aceleração, freio, direção)
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
				# fallback: tenta interpretar genericamente como círculo
				try:
					ox, oy, orad = o
					if math.hypot(x - ox, y - oy) <= (orad + car_radius):
						return x, True, traj
				except Exception:
					continue

		# chegou ao objetivo
		if x >= goal_x:
			return x, False, traj

		# checar limites verticais/horizontais - colisão com parede
		if y < env_bounds[0] or y > env_bounds[1]:
			return x, True, traj
		if x < env_x_bounds[0] or x > env_x_bounds[1]:
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


def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
	dx = x2 - x1
	dy = y2 - y1
	if dx == 0 and dy == 0:
		return math.hypot(px - x1, py - y1)
	t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
	t = max(0.0, min(1.0, t))
	projx = x1 + t * dx
	projy = y1 + t * dy
	return math.hypot(px - projx, py - projy)



def _ray_circle_t(Ox: float, Oy: float, dxr: float, dyr: float, cx: float, cy: float, cr: float, car_radius: float):
	"""Retorna o menor t>=0 onde o raio O + t*d colide com o círculo em (cx,cy).

	O círculo tem raio `cr` aumentado pelo `car_radius` do veículo. Retorna
	`None` se não houver interseção válida.
	"""
	fx = Ox - cx
	fy = Oy - cy
	a = dxr * dxr + dyr * dyr
	b = 2.0 * (fx * dxr + fy * dyr)
	c = fx * fx + fy * fy - (cr + car_radius) ** 2
	disc = b * b - 4.0 * a * c
	if disc < 0.0:
		return None
	sd = math.sqrt(disc)
	t1 = (-b - sd) / (2.0 * a)
	t2 = (-b + sd) / (2.0 * a)
	cand = None
	if t1 >= 0.0:
		cand = t1
	if t2 >= 0.0:
		if cand is None or t2 < cand:
			cand = t2
	return cand



def _ray_capsule_t(Ox: float, Oy: float, dxr: float, dyr: float, ax: float, ay: float, bx: float, by: float, seg_rad: float, car_radius: float):
	"""Raio vs cápsula (segmento com raio): retorna menor t>=0 ou None.

	Considera o segmento com raio `seg_rad` e soma o `car_radius` para a
	colisão entre cápsulas.
	"""
	# vector along segment
	u_x = bx - ax
	u_y = by - ay
	uu = u_x * u_x + u_y * u_y
	R = seg_rad + car_radius
	# project ray onto coordinates avoiding creating many temporaries
	w0x = Ox - ax
	w0y = Oy - ay
	ud = u_x * dxr + u_y * dyr
	uw0 = u_x * w0x + u_y * w0y
	if uu == 0.0:
		return _ray_circle_t(Ox, Oy, dxr, dyr, ax, ay, seg_rad, car_radius)
	v_x = dxr - u_x * (ud / uu)
	v_y = dyr - u_y * (ud / uu)
	q_x = w0x - u_x * (uw0 / uu)
	q_y = w0y - u_y * (uw0 / uu)
	A_q = v_x * v_x + v_y * v_y
	B_q = 2.0 * (q_x * v_x + q_y * v_y)
	C_q = q_x * q_x + q_y * q_y - R * R
	solutions = []
	if abs(A_q) > 1e-12:
		discq = B_q * B_q - 4.0 * A_q * C_q
		if discq >= 0.0:
			sd = math.sqrt(discq)
			t1 = (-B_q - sd) / (2.0 * A_q)
			t2 = (-B_q + sd) / (2.0 * A_q)
			for t in (t1, t2):
				if t >= 0.0:
					s = (uw0 + t * ud) / uu
					if 0.0 <= s <= 1.0:
						solutions.append(t)
	# checagem das "end-circles" (cantos do segmento)
	tc = _ray_circle_t(Ox, Oy, dxr, dyr, ax, ay, seg_rad, car_radius)
	if tc is not None:
		solutions.append(tc)
	tc = _ray_circle_t(Ox, Oy, dxr, dyr, bx, by, seg_rad, car_radius)
	if tc is not None:
		solutions.append(tc)
	if not solutions:
		return None
	return min(solutions)


def compute_potential_field(obstacles,
							goal_pos: Tuple[float, float],
							x_min: float = -10.0,
							x_max: float = 110.0,
							y_min: float = -25.0,
							y_max: float = 25.0,
							resolution: float = 0.5) -> dict:
	"""Computa um campo de custo até o objetivo (Dijkstra) propagado de `goal_pos`.

	Retorna um dicionário com chaves: 'grid' (array 2D com forma (nx, ny)),
	'origin' (x_min, y_min), 'res' (resolução) e 'shape' (nx, ny).
	Obstáculos suportados: tuplas de círculo (x,y,r) e segmentos ('seg', x1,y1,x2,y2,r).
	"""
	xs = np.arange(x_min + resolution / 2.0, x_max, resolution)
	ys = np.arange(y_min + resolution / 2.0, y_max, resolution)
	nx = xs.size
	ny = ys.size
	if nx <= 0 or ny <= 0:
		return {'grid': np.full((0, 0), float('inf')), 'origin': (x_min, y_min), 'res': resolution, 'shape': (nx, ny)}

	# cria uma grade de centros de célula com indexação 'ij' (grid[i,j] -> xs[i], ys[j])
	XX, YY = np.meshgrid(xs, ys, indexing='ij')
	blocked = np.zeros((nx, ny), dtype=bool)

	for o in (obstacles or []):
		if isinstance(o, tuple) and len(o) == 3:
			ox, oy, orad = o
			d2 = (XX - ox) ** 2 + (YY - oy) ** 2
			blocked |= (d2 <= (orad) ** 2)
		elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
			_, x1, y1, x2, y2, orad = o
			dx = x2 - x1
			dy = y2 - y1
			denom = dx * dx + dy * dy
			if denom == 0:
				d2 = (XX - x1) ** 2 + (YY - y1) ** 2
			else:
				t = ((XX - x1) * dx + (YY - y1) * dy) / denom
				t = np.clip(t, 0.0, 1.0)
				projx = x1 + t * dx
				projy = y1 + t * dy
				d2 = (XX - projx) ** 2 + (YY - projy) ** 2
			blocked |= (d2 <= (orad) ** 2)
		else:
			try:
				ox, oy, orad = o
				d2 = (XX - ox) ** 2 + (YY - oy) ** 2
				blocked |= (d2 <= (orad) ** 2)
			except Exception:
				continue

	INF = float('inf')
	pot = np.full((nx, ny), INF, dtype=float)

	gx, gy = goal_pos
	# mapeia a posição do objetivo para o índice da célula mais próxima
	ix = int(round((gx - (x_min + resolution / 2.0)) / resolution))
	iy = int(round((gy - (y_min + resolution / 2.0)) / resolution))
	if ix < 0 or ix >= nx or iy < 0 or iy >= ny or blocked[ix, iy]:
		return {'grid': pot, 'origin': (x_min, y_min), 'res': resolution, 'shape': (nx, ny)}

	pot[ix, iy] = 0.0
	heap = [(0.0, ix, iy)]
	neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
	costs = {( -1, 0): resolution, (1, 0): resolution, (0, -1): resolution, (0, 1): resolution,
			 (-1, -1): resolution * math.sqrt(2), (-1, 1): resolution * math.sqrt(2),
			 (1, -1): resolution * math.sqrt(2), (1, 1): resolution * math.sqrt(2)}

	while heap:
		cost, ci, cj = heapq.heappop(heap)
		if cost != pot[ci, cj]:
			continue
		for dx_cell, dy_cell in neigh:
			ni = ci + dx_cell
			nj = cj + dy_cell
			if not (0 <= ni < nx and 0 <= nj < ny):
				continue
			if blocked[ni, nj]:
				continue
			step_cost = costs[(dx_cell, dy_cell)]
			newc = cost + step_cost
			if newc < pot[ni, nj]:
				pot[ni, nj] = newc
				heapq.heappush(heap, (newc, ni, nj))

	return {'grid': pot, 'origin': (x_min, y_min), 'res': resolution, 'shape': (nx, ny)}


def sample_potential(field: dict, x: float, y: float) -> float:
	if field is None:
		return float('inf')
	grid = field['grid']
	x0, y0 = field['origin']
	res = field['res']
	nx, ny = field['shape']
	if nx == 0 or ny == 0:
		return float('inf')
	fx = (x - (x0 + res / 2.0)) / res
	fy = (y - (y0 + res / 2.0)) / res
	ix = int(math.floor(fx))
	iy = int(math.floor(fy))
	wx = fx - ix
	wy = fy - iy
	if ix < 0 or iy < 0 or ix + 1 >= nx or iy + 1 >= ny:
		return float('inf')
	v00 = grid[ix, iy]
	v10 = grid[ix + 1, iy]
	v01 = grid[ix, iy + 1]
	v11 = grid[ix + 1, iy + 1]
	if not np.isfinite(v00) and not np.isfinite(v10) and not np.isfinite(v01) and not np.isfinite(v11):
		return float('inf')
	BIG = 1e6
	vals = [v00 if np.isfinite(v00) else BIG,
			v10 if np.isfinite(v10) else BIG,
			v01 if np.isfinite(v01) else BIG,
			v11 if np.isfinite(v11) else BIG]
	v = (vals[0] * (1 - wx) * (1 - wy) + vals[1] * wx * (1 - wy) + vals[2] * (1 - wx) * wy + vals[3] * wx * wy)
	return float(v)


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

		# gerador aleatório (reprodutível usando `seed`)
		self.rng = default_rng(seed)

		# população armazenada como arrays numpy
		self.pop = self._init_population()

		# pré-computa um campo de potencial para avaliar trajetórias curvas
		# limites escolhidos para combinar com o mundo usado em main.py; resolução configurável
		try:
			self.potential_field = compute_potential_field(self.obstacles,
									(goal_x if False else (self.goal_x, 0.0)),
								x_min=-10.0,
								x_max=self.goal_x + 10.0,
								y_min=-25.0,
								y_max=25.0,
								resolution=0.5)
		except Exception:
			self.potential_field = None

	def _init_population(self) -> np.ndarray:
		low = GENE_BOUNDS[:, 0]
		high = GENE_BOUNDS[:, 1]
		# amostra uniformemente por gene usando broadcasting
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
			steps = max(0, len(traj) - 1)
			time_cost = 0.02
			# se um campo de potencial estiver disponível, usa-o (custo menor = melhor)
			if getattr(self, 'potential_field', None) is not None:
				final_x, final_y = traj[-1]
				pot = sample_potential(self.potential_field, final_x, final_y)
				if not np.isfinite(pot):
					fitness[i] = -1e3 - (time_cost * steps)
				else:
					fitness[i] = -pot - (time_cost * steps)
			else:
				x_used = min(float(x_reached), float(self.goal_x))
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
		# retorna índice do indivíduo selecionado (usando RNG interno)
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
		# perturbação gaussiana; uma máscara indica quais genes serão mutados
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

			# atualiza o melhor indivíduo encontrado
			idx_best = int(np.argmax(fitness))
			if fitness[idx_best] > best_fitness:
				best_fitness = float(fitness[idx_best])
				best_genome = CarGenome.from_vector(self.pop[idx_best].copy(), NN_WEIGHTS_SIZE)

			if verbose and (gen % max(1, int(generations) // 10) == 0 or gen == int(generations) - 1):
				print(f"Generation {gen:4d}: best fitness = {best_fitness:.3f}")

			# gera nova população por seleção, crossover e mutação
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

