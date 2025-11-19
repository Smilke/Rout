"""Runner simples para o algoritmo genético de `genetic_algorithm.py`.

Executar este arquivo inicia a otimização e imprime o melhor indivíduo encontrado.
"""
import argparse
from pprint import pprint
import math
import numpy as np
try:
	import pygame
	PYGAME_AVAILABLE = True
except Exception:
	PYGAME_AVAILABLE = False

from genetic_algorithm import GeneticAlgorithm, CarGenome, simulate_car, NN_WEIGHTS_SIZE


def default_obstacles():
	# retorna uma lista de obstáculos (x, y, radius)
	return [
		(30.0, 0.0, 3.0),
		(60.0, 2.0, 4.0),
	]


def run(args):
	# estrutura de obstáculos: lista principal e grupos (para desfazer por clique direito)
	obstacles = []
	obstacles_groups = []
	if args.obstacles:
		init_group = default_obstacles()
		obstacles.extend(init_group)
		obstacles_groups.append(init_group)
	ga = GeneticAlgorithm(population_size=args.pop,
						  mutation_rate=args.mut,
						  crossover_rate=0.9,
						  tournament_size=3,
						  seed=args.seed,
						  goal_x=args.goal,
						  obstacles=obstacles)
	# visualization state
	env_y_bounds = (-25.0, 25.0)
	x_min = -10.0
	x_max = args.goal + 10.0

	# Modo Jogo (apenas): verificar disponibilidade do pygame
	if not PYGAME_AVAILABLE:
		print("Pygame não está disponível. Instale pygame (pip install pygame) e tente novamente.")
		return
	# inicializa pygame
	pygame.init()
	screen_w = 1000
	screen_h = 600
	screen = pygame.display.set_mode((screen_w, screen_h))
	pygame.display.set_caption("GA Car Simulation - Game View")
	clock = pygame.time.Clock()

	def animate_generation_pygame(gen, population, fitness):
		# determinar quais indivíduos desenhar: ou os top-k ou toda a população
		if getattr(args, 'show_all', False):
			idxs = np.arange(len(population))
		else:
			topk = args.top
			idxs = np.argsort(fitness)[-topk:][::-1]
		# precompute trajectories
		trajs = []
		max_len = 0
		for idx in idxs:
			genome = CarGenome.from_vector(population[idx], NN_WEIGHTS_SIZE)
			_, _, traj = simulate_car(genome, goal_x=args.goal, obstacles=obstacles, env_bounds=env_y_bounds)
			# opcionalmente amostrar o trajeto para reduzir frames
			if args.frame_step > 1:
				traj = traj[::args.frame_step]
			trajs.append(traj)
			if len(traj) > max_len:
				max_len = len(traj)

		# world -> screen mapping
		def world_to_screen(px, py):
			sx = int((px - x_min) / (x_max - x_min) * screen_w)
			sy = int(screen_h - (py - env_y_bounds[0]) / (env_y_bounds[1] - env_y_bounds[0]) * screen_h)
			return sx, sy

		def screen_to_world(sx, sy):
			px = x_min + (sx / float(screen_w)) * (x_max - x_min)
			py = env_y_bounds[0] + ((screen_h - sy) / float(screen_h)) * (env_y_bounds[1] - env_y_bounds[0])
			return px, py

		running = True
		step = 0
		paused = False
		# estado de desenho de linha: None ou (wx, wy)
		line_start = None
		# parâmetros para converter uma linha em obstáculos circulares
		line_obstacle_radius = 2.0
		line_obstacle_spacing = 1.0
		while running and step < max_len:
			for ev in pygame.event.get():
				if ev.type == pygame.QUIT:
					pygame.quit()
					raise SystemExit()
				elif ev.type == pygame.MOUSEBUTTONDOWN:
					# clique esquerdo inicia/finaliza uma linha
					if ev.button == 1:
						sx, sy = ev.pos
						wx, wy = screen_to_world(sx, sy)
						if line_start is None:
							line_start = (wx, wy)
						else:
							# criar obstáculos ao longo da linha entre line_start e (wx,wy)
							x1, y1 = line_start
							x2, y2 = wx, wy
							dx = x2 - x1
							dy = y2 - y1
							dist = math.hypot(dx, dy)
							if dist == 0:
								line_start = None
								continue
							steps = max(1, int(dist / line_obstacle_spacing))
							group = []
							for i in range(steps + 1):
								t = i / float(steps)
								px = x1 + dx * t
								py = y1 + dy * t
								group.append((px, py, line_obstacle_radius))
							obstacles.extend(group)
							obstacles_groups.append(group)
							# também atualizar o GA para que a simulação use os novos obstáculos
							try:
								ga.obstacles = list(obstacles)
							except Exception:
								pass
							line_start = None
					# clique direito remove o último grupo de obstáculos adicionado
					elif ev.button == 3:
						if obstacles_groups:
							group = obstacles_groups.pop()
							for o in group:
								if o in obstacles:
									obstacles.remove(o)
							try:
								ga.obstacles = list(obstacles)
							except Exception:
								pass
						# cancelar desenho em andamento, se houver
						line_start = None
				elif ev.type == pygame.KEYDOWN:
					if ev.key == pygame.K_SPACE:
						paused = not paused
					elif ev.key == pygame.K_RIGHT:
						step += 1
					elif ev.key == pygame.K_ESCAPE:
						pygame.quit()
						raise SystemExit()

			if paused:
				clock.tick(10)
				continue

			screen.fill((30, 30, 30))
			# desenhar obstáculos
			for (ox, oy, orad) in obstacles:
				s_ox, s_oy = world_to_screen(ox, oy)
				s_r = max(2, int(orad / (x_max - x_min) * screen_w))
				pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)

			# se houver uma linha em desenho, desenhá-la (feedback visual)
			if line_start is not None:
				sx1, sy1 = world_to_screen(*line_start)
				sx2, sy2 = pygame.mouse.get_pos()
				pygame.draw.line(screen, (200, 200, 100), (sx1, sy1), (sx2, sy2), 2)

			# desenhar meta
			gx1, _ = world_to_screen(args.goal, env_y_bounds[0])
			gx2, _ = world_to_screen(args.goal, env_y_bounds[1])
			pygame.draw.line(screen, (200, 30, 30), (gx1, 0), (gx1, screen_h), 2)

			# desenhar carros
			colors = [(50,200,50),(50,150,240),(240,200,50),(200,100,240),(240,120,120)]
			for i, traj in enumerate(trajs):
				color = colors[i % len(colors)]
				if step < len(traj):
					px, py = traj[step]
				else:
					px, py = traj[-1]
				sx, sy = world_to_screen(px, py)
				pygame.draw.circle(screen, color, (sx, sy), 6)
				# desenhar rastro (trail)
				for t in range(max(0, step-30), min(len(traj), step)):
					tx, ty = traj[t]
					tsx, tsy = world_to_screen(tx, ty)
					alpha = int(255 * (t - max(0, step-30)) / 30)
					pygame.draw.circle(screen, color, (tsx, tsy), 2)

			# HUD
			font = pygame.font.SysFont(None, 24)
			txt = font.render(f"Gen {gen}  step {step}/{max_len}  press SPACE to pause", True, (220,220,220))
			screen.blit(txt, (8,8))
			# info de obstáculos
			txt2 = font.render(f"Obstáculos ativos ({len(obstacles)}): left click start/end, right click undo", True, (200,200,200))
			screen.blit(txt2, (8, 32))

			pygame.display.flip()
			clock.tick(args.fps)
			step += 1

	# executar GA com callback do pygame (visualização por geração)
	print("Iniciando execução (modo jogo). Feche a janela ou pressione ESC para sair.")
	try:
		ga.run(generations=args.gens, verbose=True, on_generation=animate_generation_pygame)
	finally:
		pygame.quit()


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--pop", type=int, default=40, help="tamanho da população")
	p.add_argument("--gens", type=int, default=80, help="número de gerações")
	p.add_argument("--mut", type=float, default=0.15, help="taxa de mutação")
	p.add_argument("--goal", type=float, default=100.0, help="posição x do objetivo")
	p.add_argument("--obstacles", action="store_true", help="usar obstáculos de exemplo")
	p.add_argument("--seed", type=int, help="seed aleatória")
	p.add_argument("--top", type=int, default=3, help="quantos melhores carros desenhar por snapshot")
	p.add_argument("--show-all", action="store_true", dest="show_all", help="desenhar toda a população em vez dos top-k")
	# o modo game é obrigatório; não há modo texto/matplotlib
	# compatibilidade: --game flag removida (apenas modo jogo)
	p.add_argument("--fps", type=int, default=200, help="frames por segundo na animação do jogo")
	p.add_argument("--frame-step", dest="frame_step", type=int, default=1, help="amostragem dos pontos de trajetória para animação (1 = todos)")
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(args)

