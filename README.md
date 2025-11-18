# Rout
Repositório criado para desenvolvimento do projeto da cadeira de Inteligencia Artificial 

## Como executar

Requisitos (veja `requirements.txt`): numpy e pygame.

Modo "game" (visualização em tempo real com Pygame) — OBS: o modo texto/matplotlib foi removido;
o programa roda exclusivamente com a interface tipo jogo.

1. Instale dependências:

```powershell
python -m pip install -r requirements.txt
```

2. Execute a simulação (ex.: com obstáculos e 4 carros mostrados por geração):

```powershell
python rout\main.py --obstacles --top 4 --fps 12
```

Controles durante a animação:
- Espaço: pausar / continuar
- Seta para a direita: avançar um passo
- ESC ou fechar a janela: sair

Use `--frame-step` para reduzir número de frames (ex.: --frame-step 2 pula metade dos pontos da trajetória).
