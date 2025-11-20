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

	# helper mappers (world <-> screen)
	def world_to_screen(px, py):
		sx = int((px - x_min) / (x_max - x_min) * screen_w)
		sy = int(screen_h - (py - env_y_bounds[0]) / (env_y_bounds[1] - env_y_bounds[0]) * screen_h)
		return sx, sy

	def screen_to_world(sx, sy):
		px = x_min + (sx / float(screen_w)) * (x_max - x_min)
		py = env_y_bounds[0] + ((screen_h - sy) / float(screen_h)) * (env_y_bounds[1] - env_y_bounds[0])
		return px, py

	# Fase de construção: o usuário desenha o mapa antes da simulação
	def build_map_pygame():
		building = True
		line_start = None
		# parâmetros para converter uma linha em um segmento contínuo
		line_obstacle_radius = 2.0
		# instruções
		info_font = pygame.font.SysFont(None, 22)
		while building:
			for ev in pygame.event.get():
				if ev.type == pygame.QUIT:
					pygame.quit()
					raise SystemExit()
				elif ev.type == pygame.MOUSEBUTTONDOWN:
					if ev.button == 1:
						sx, sy = ev.pos
						wx, wy = screen_to_world(sx, sy)
						if line_start is None:
							line_start = (wx, wy)
						else:
							x1, y1 = line_start
							x2, y2 = wx, wy
							dx = x2 - x1
							dy = y2 - y1
							dist = math.hypot(dx, dy)
							if dist != 0:
								seg = ('seg', x1, y1, x2, y2, line_obstacle_radius)
								obstacles.append(seg)
								obstacles_groups.append([seg])
								try:
									# atualizar GA caso já exista
									ga.obstacles = list(obstacles)
								except Exception:
									pass
							line_start = None
					elif ev.button == 3:
						# undo last group
						if obstacles_groups:
							group = obstacles_groups.pop()
							for o in group:
								if o in obstacles:
									obstacles.remove(o)
							try:
								ga.obstacles = list(obstacles)
							except Exception:
								pass
						line_start = None
				elif ev.type == pygame.KEYDOWN:
					if ev.key == pygame.K_RETURN or ev.key == pygame.K_s:
						# start simulation
						building = False
					elif ev.key == pygame.K_ESCAPE:
						pygame.quit()
						raise SystemExit()

			screen.fill((30, 30, 30))
			# desenhar obstáculos existentes
			for o in obstacles:
				if isinstance(o, tuple) and len(o) == 3:
					ox, oy, orad = o
					s_ox, s_oy = world_to_screen(ox, oy)
					s_r = max(2, int(orad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)
				elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
					_, ax, ay, bx, by, srad = o
					s_ax, s_ay = world_to_screen(ax, ay)
					s_bx, s_by = world_to_screen(bx, by)
					th = max(2, int((srad / (x_max - x_min)) * screen_w * 2))
					pygame.draw.line(screen, (200, 80, 80), (s_ax, s_ay), (s_bx, s_by), th)
					end_r = max(2, int(srad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ax, s_ay), end_r)
					pygame.draw.circle(screen, (200, 80, 80), (s_bx, s_by), end_r)
				else:
					try:
						ox, oy, orad = o
						s_ox, s_oy = world_to_screen(ox, oy)
						s_r = max(2, int(orad / (x_max - x_min) * screen_w))
						pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)
					except Exception:
						pass

			# if a line is in progress, draw it
			if line_start is not None:
				sx1, sy1 = world_to_screen(*line_start)
				sx2, sy2 = pygame.mouse.get_pos()
				pygame.draw.line(screen, (200, 200, 100), (sx1, sy1), (sx2, sy2), 2)

			# draw spawn marker (agents spawn at world (0,0))
			spawn_px, spawn_py = 0.0, 0.0
			s_sx, s_sy = world_to_screen(spawn_px, spawn_py)
			# solid green marker
			pygame.draw.circle(screen, (50, 200, 50), (s_sx, s_sy), 8)
			# optional avoidance ring (world meters)
			avoid_radius_world = 4.0
			avoid_r_px = max(10, int(avoid_radius_world / (x_max - x_min) * screen_w))
			pygame.draw.circle(screen, (50, 200, 50), (s_sx, s_sy), avoid_r_px, 2)
			# label
			try:
				lbl = info_font.render("Spawn", True, (200, 255, 200))
				screen.blit(lbl, (s_sx + 10, s_sy - 10))
			except Exception:
				pass

			# HUD/instruções
			txt1 = info_font.render("Desenhe obstáculos: left click start/end, right click undo", True, (220,220,220))
			txt2 = info_font.render("Press ENTER or S to start simulation | ESC to quit", True, (200,200,200))
			screen.blit(txt1, (8,8))
			screen.blit(txt2, (8,32))

			pygame.display.flip()
			clock.tick(60)
		return obstacles


 

	# executar GA com callback do pygame (visualização por geração)
	# primeiro: fase de construção — permitir ao usuário desenhar o mapa
	print("Fase de construção: desenhe o mapa na janela. Pressione ENTER ou S para iniciar a simulação.")
	build_map_pygame()

	# criar o GA agora com os obstáculos desenhados
	ga = GeneticAlgorithm(population_size=args.pop,
				  mutation_rate=args.mut,
				  crossover_rate=0.9,
				  tournament_size=3,
				  seed=args.seed,
				  goal_x=args.goal,
				  obstacles=obstacles)

	# Visualize potential field (greenish) before starting simulation
	def visualize_potential_field(field):
		if field is None:
			return
		grid = field.get('grid')
		if grid is None or grid.size == 0:
			return
		x0, y0 = field['origin']
		res = field['res']
		nx, ny = field['shape']
		# choose vmax for color scaling (avoid extreme outliers)
		finite = grid[np.isfinite(grid)]
		if finite.size == 0:
			vmax = 1.0
		else:
			vmax = float(np.percentile(finite, 95))
			if vmax <= 0:
				vmax = float(finite.max()) if finite.size else 1.0

		overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)

		cell_w = max(1, int(res / (x_max - x_min) * screen_w))
		cell_h = max(1, int(res / (env_y_bounds[1] - env_y_bounds[0]) * screen_h))

		for i in range(nx):
			for j in range(ny):
				v = grid[i, j]
				# center of cell in world coords
				wx = x0 + (i + 0.5) * res
				wy = y0 + (j + 0.5) * res
				sx, sy = world_to_screen(wx, wy)
				# map to top-left of rect
				rect = pygame.Rect(max(0, sx - cell_w // 2), max(0, sy - cell_h // 2), cell_w, cell_h)
				if not np.isfinite(v):
					color = (30, 30, 30, 120)
				else:
					intensity = 1.0 - min(v / vmax, 1.0)
					g = int(60 + intensity * 195)
					color = (20, g, 20, 120)
				pygame.draw.rect(overlay, color, rect)

		# show overlay with legend/instruction
		showing = True
		font = pygame.font.SysFont(None, 20)
		while showing:
			screen.fill((30, 30, 30))
			# draw base obstacles on background
			for o in obstacles:
				if isinstance(o, tuple) and len(o) == 3:
					ox, oy, orad = o
					s_ox, s_oy = world_to_screen(ox, oy)
					s_r = max(2, int(orad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)
				elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
					_, ax, ay, bx, by, srad = o
					s_ax, s_ay = world_to_screen(ax, ay)
					s_bx, s_by = world_to_screen(bx, by)
					th = max(2, int((srad / (x_max - x_min)) * screen_w * 2))
					pygame.draw.line(screen, (200, 80, 80), (s_ax, s_ay), (s_bx, s_by), th)
					end_r = max(2, int(srad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ax, s_ay), end_r)
					pygame.draw.circle(screen, (200, 80, 80), (s_bx, s_by), end_r)
			# blit overlay
			screen.blit(overlay, (0, 0))
			# legend
			txt = font.render("Campo Potencial (verde = vusto pequeno). Pressione ENTER/S para começar, ESC para sair.", True, (220, 220, 220))
			screen.blit(txt, (8, 8))
			pygame.display.flip()
			for ev in pygame.event.get():
				if ev.type == pygame.QUIT:
					pygame.quit()
					raise SystemExit()
				elif ev.type == pygame.KEYDOWN:
					if ev.key == pygame.K_RETURN or ev.key == pygame.K_s:
						showing = False
					elif ev.key == pygame.K_ESCAPE:
						pygame.quit()
						raise SystemExit()
			clock.tick(30)

	# call visualization (if available)
	visualize_potential_field(getattr(ga, 'potential_field', None))

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
					# during simulation we ignore mouse drawing events; map editing is only
					# allowed in the build phase. Keep mouse for potential future use.
					pass
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
			# desenhar obstáculos (suporta círculos e segmentos/cápsulas)
			for o in obstacles:
				if isinstance(o, tuple) and len(o) == 3:
					ox, oy, orad = o
					s_ox, s_oy = world_to_screen(ox, oy)
					s_r = max(2, int(orad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)
				elif isinstance(o, tuple) and len(o) == 6 and o[0] == 'seg':
					_, ax, ay, bx, by, srad = o
					s_ax, s_ay = world_to_screen(ax, ay)
					s_bx, s_by = world_to_screen(bx, by)
					# thickness in pixels (approx)
					th = max(2, int((srad / (x_max - x_min)) * screen_w * 2))
					pygame.draw.line(screen, (200, 80, 80), (s_ax, s_ay), (s_bx, s_by), th)
					# caps at ends
					end_r = max(2, int(srad / (x_max - x_min) * screen_w))
					pygame.draw.circle(screen, (200, 80, 80), (s_ax, s_ay), end_r)
					pygame.draw.circle(screen, (200, 80, 80), (s_bx, s_by), end_r)
				else:
					# fallback: try draw as circle
					try:
						ox, oy, orad = o
						s_ox, s_oy = world_to_screen(ox, oy)
						s_r = max(2, int(orad / (x_max - x_min) * screen_w))
						pygame.draw.circle(screen, (200, 80, 80), (s_ox, s_oy), s_r)
					except Exception:
						pass

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
			# info de obstáculos (somente criados na fase de construção)
			txt2 = font.render(f"Obstáculos ativos ({len(obstacles)}): criados na fase de construção", True, (200,200,200))
			screen.blit(txt2, (8, 32))

			pygame.display.flip()
			clock.tick(args.fps)
			step += 1

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
	p.add_argument("--fps", type=int, default=200, help="frames por segundo na animação do jogo")
	p.add_argument("--frame-step", dest="frame_step", type=int, default=1, help="amostragem dos pontos de trajetória para animação (1 = todos)")
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(args)

