# -- coding: utf-8 --
"""
Created on Fri Sep 26 11:50:50 2025

@author: Yefry
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------- FITNESS FUNCTION (N-Reinas) -----------
def fitness_function(board):
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == abs(i - j):  # conflicto en diagonal
                conflicts += 1
    return conflicts

# ----------- MUTACIÓN (Swap Mutation) -----------
def mutate(board, mutation_rate=0.1):
    child = board.copy()
    if np.random.rand() < mutation_rate:
        n = len(board)
        a, b = np.random.choice(n, 2, replace=False)
        child[a], child[b] = child[b], child[a]
    return child

# ----------- CROSSOVER (PMX simplificado) -----------
def crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = -np.ones(n, dtype=int)
    # copiar segmento
    child[a:b+1] = parent1[a:b+1]
    # rellenar con genes de parent2 en orden
    fill = [g for g in parent2 if g not in child]
    idx = [i for i in range(n) if child[i] == -1]
    for i, pos in enumerate(idx):
        child[pos] = fill[i]
    return child

# ----------- SELECCIÓN POR TORNEO -----------
def tournament_selection(population, fitness, k=3):
    idxs = np.random.choice(len(population), k, replace=False)
    best_idx = idxs[np.argmin([fitness[i] for i in idxs])]
    return population[best_idx]

# ----------- Visualizar tablero -----------
def mostrar_tablero(solucion):
    n = len(solucion)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(n):
        for j in range(n):
            color = "white" if (i + j) % 2 == 0 else "gray"
            rect = plt.Rectangle((j, n - i - 1), 1, 1, facecolor=color)
            ax.add_patch(rect)
    for col, fila in enumerate(solucion):
        ax.text(col + 0.5, n - fila - 0.5, "♛", ha="center", va="center", fontsize=28, color="red")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Solución de las {n} Reinas")
    plt.show()

# ----------- Estrategia Evolutiva Mejorada (μ + λ) -----------
def evolutionary_strategy_plus(mu, lambd, n, generations, mutation_rate=0.1, verbose=False):
    # Inicializar población con permutaciones aleatorias
    population = [np.random.permutation(n) for _ in range(mu)]
    best_fitness_history = []

    for gen in range(generations):
        # Evaluar fitness actual
        fitness_pop = np.array([fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness_pop)
        best_fit = fitness_pop[best_idx]
        best_fitness_history.append(best_fit)

        if verbose:
            print(f"Generación {gen+1}: Mejor fitness = {best_fit}")

        if best_fit == 0:
            break

        # Generar λ descendientes
        offspring = []
        for _ in range(lambd):
            parent1 = tournament_selection(population, fitness_pop)
            parent2 = tournament_selection(population, fitness_pop)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            offspring.append(child)

        # Evaluar descendientes
        offspring_fitness = np.array([fitness_function(ind) for ind in offspring])

        # Combinar y seleccionar los μ mejores
        combined_population = population + offspring
        combined_fitness = np.concatenate((fitness_pop, offspring_fitness))
        best_indices = np.argsort(combined_fitness)[:mu]
        population = [combined_population[i] for i in best_indices]

    best_solution = population[0]
    return best_solution, fitness_function(best_solution), best_fitness_history

# ----------- Ejecuciones múltiples para análisis -----------
def run_experiments(mu, lambd, n, generations, mutation_rate, runs=10):
    histories = []
    gens_needed = []

    for r in range(runs):
        sol, fit, history = evolutionary_strategy_plus(mu, lambd, n, generations, mutation_rate)
        histories.append(history)
        if fit == 0:
            gens_needed.append(len(history))
        else:
            gens_needed.append(generations)

    max_len = max(len(h) for h in histories)
    padded = [h + [h[-1]]*(max_len - len(h)) for h in histories]
    histories = np.array(padded)

    mean_history = histories.mean(axis=0)
    std_history = histories.std(axis=0)

    return mean_history, std_history, gens_needed

# ----------- Ejemplo de ejecución con N=8 -----------
mu, lambd, n, generations, mutation_rate = 30, 70, 6, 10, 0.1
mean_history, std_history, gens_needed = run_experiments(mu, lambd, n, generations, mutation_rate, runs=10)

print(f"\nConfiguración (μ={mu}, λ={lambd}, N={n}, mutation={mutation_rate})")
print(f"Generaciones promedio para solución: {np.mean(gens_needed):.2f} ± {np.std(gens_needed):.2f}")
print(f"Mejor solución encontrada en última ejecución: {gens_needed[-1]} generaciones")

# ----------- Graficar convergencia promedio -----------
plt.figure(figsize=(8,5))
plt.plot(mean_history, label="Promedio fitness")
plt.fill_between(range(len(mean_history)), mean_history-std_history, mean_history+std_history, alpha=0.3, label="±1 Desv. Est.")
plt.xlabel("Generaciones")
plt.ylabel("Conflictos")
plt.title(f"Convergencia Evolutiva (μ+λ) - N={n}")
plt.legend()
plt.grid()
plt.show()

# ----------- Mostrar tablero final de un run -----------
best_sol, best_fit, _ = evolutionary_strategy_plus(mu, lambd, n, generations, mutation_rate, verbose=True)
print("\nMejor solución encontrada:", best_sol)
print("Conflictos (0 = solución válida):", best_fit)
mostrar_tablero(best_sol)