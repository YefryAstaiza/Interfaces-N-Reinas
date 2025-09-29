from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import time
import json
from datetime import datetime

app = Flask(__name__)

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
    child[a:b+1] = parent1[a:b+1]
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
def generar_tablero_img(solucion):
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
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

# ----------- Gráfico de convergencia -----------
def generar_grafico_convergencia(history, n, mu, lambd):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, 'b-', linewidth=2, label='Mejor Fitness')
    ax.set_xlabel('Generación')
    ax.set_ylabel('Conflictos (Fitness)')
    ax.set_title(f'Convergencia - N={n}, μ={mu}, λ={lambd}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

# ----------- Algoritmo Evolutivo Principal -----------
def evolutionary_strategy_plus(mu, lambd, n, generations, mutation_rate=0.1):
    start_time = time.time()
    population = [np.random.permutation(n) for _ in range(mu)]
    best_fitness_history = []
    generation_details = []

    for gen in range(generations):
        fitness_pop = np.array([fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness_pop)
        best_fit = fitness_pop[best_idx]
        avg_fit = np.mean(fitness_pop)
        best_fitness_history.append(best_fit)
        
        generation_details.append({
            'generation': gen + 1,
            'best_fitness': int(best_fit),
            'average_fitness': float(avg_fit),
            'best_solution': population[best_idx].tolist()
        })

        if best_fit == 0:
            break

        offspring = []
        for _ in range(lambd):
            parent1 = tournament_selection(population, fitness_pop)
            parent2 = tournament_selection(population, fitness_pop)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            offspring.append(child)

        offspring_fitness = np.array([fitness_function(ind) for ind in offspring])
        combined_population = population + offspring
        combined_fitness = np.concatenate((fitness_pop, offspring_fitness))
        best_indices = np.argsort(combined_fitness)[:mu]
        population = [combined_population[i] for i in best_indices]

    execution_time = time.time() - start_time
    best_solution = population[0]
    final_fitness = fitness_function(best_solution)
    
    return best_solution, final_fitness, best_fitness_history, execution_time, generation_details

# ----------- Ejecuciones múltiples para análisis -----------
def run_experiments(mu, lambd, n, generations, mutation_rate, runs=10):
    histories = []
    execution_times = []
    gens_needed = []
    success_count = 0
    
    for r in range(runs):
        start_time = time.time()
        sol, fit, history, exec_time, _ = evolutionary_strategy_plus(mu, lambd, n, generations, mutation_rate)
        execution_times.append(exec_time)
        histories.append(history)
        
        if fit == 0:
            gens_needed.append(len(history))
            success_count += 1
        else:
            gens_needed.append(generations)

    success_rate = (success_count / runs) * 100
    
    # Pad histories for averaging
    max_len = max(len(h) for h in histories)
    padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]
    histories_array = np.array(padded)
    
    mean_history = histories_array.mean(axis=0)
    std_history = histories_array.std(axis=0)
    
    stats = {
        'mean_generations': np.mean(gens_needed),
        'std_generations': np.std(gens_needed),
        'mean_time': np.mean(execution_times),
        'std_time': np.std(execution_times),
        'success_rate': success_rate,
        'mean_history': mean_history.tolist(),
        'std_history': std_history.tolist()
    }
    
    return stats

# ----------- Rutas Flask -----------
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    error = None
    analysis_result = None

    if request.method == "POST":
        try:
            if 'analyze' in request.form:
                # Análisis comparativo
                n_values = json.loads(request.form["n_values"])
                mu_values = json.loads(request.form["mu_values"])
                mutation_rates = json.loads(request.form["mutation_rates"])
                generations = int(request.form["generations"])
                runs = int(request.form["runs"])
                
                analysis_data = []
                
                for n in n_values:
                    for mu in mu_values:
                        for mutation_rate in mutation_rates:
                            lambd = mu * 2  # λ = 2*μ como valor por defecto
                            
                            stats = run_experiments(mu, lambd, n, generations, mutation_rate, runs)
                            
                            analysis_data.append({
                                'n': n,
                                'mu': mu,
                                'lambd': lambd,
                                'mutation_rate': mutation_rate,
                                'mean_generations': round(stats['mean_generations'], 2),
                                'std_generations': round(stats['std_generations'], 2),
                                'mean_time': round(stats['mean_time'], 4),
                                'std_time': round(stats['std_time'], 4),
                                'success_rate': round(stats['success_rate'], 1)
                            })
                
                analysis_result = {
                    'data': analysis_data,
                    'total_configurations': len(analysis_data)
                }
                
            else:
                # Ejecución individual
                n = int(request.form["n"])
                mu = int(request.form["mu"])
                lambd = int(request.form["lambd"])
                generations = int(request.form["generations"])
                mutation_rate = float(request.form["mutation_rate"])
                runs = int(request.form["runs"])

                if n <= 0 or mu <= 0 or lambd <= 0 or generations <= 0 or runs <= 0:
                    error = "Todos los valores deben ser mayores a 0"
                elif mutation_rate < 0 or mutation_rate > 1:
                    error = "La tasa de mutación debe estar entre 0 y 1"
                elif n > 20:
                    error = "El tamaño del tablero (N) no puede ser mayor a 20 por razones de rendimiento"
                else:
                    # Ejecutar algoritmo
                    solucion, fitness, history, exec_time, gen_details = evolutionary_strategy_plus(
                        mu, lambd, n, generations, mutation_rate
                    )
                    
                    # Ejecutar análisis estadístico
                    stats = run_experiments(mu, lambd, n, generations, mutation_rate, runs)
                    
                    img_base64 = generar_tablero_img(solucion)
                    convergence_img = generar_grafico_convergencia(history, n, mu, lambd)

                    resultado = {
                        "solucion": solucion.tolist(),
                        "fitness": fitness,
                        "generaciones_usadas": len(history),
                        "generaciones_maximas": generations,
                        "n": n,
                        "mu": mu,
                        "lambd": lambd,
                        "mutation_rate": mutation_rate,
                        "runs": runs,
                        "es_solucion_valida": (fitness == 0),
                        "conflictos": fitness,
                        "tiempo_ejecucion": round(exec_time, 4),
                        "tablero": img_base64,
                        "convergencia": convergence_img,
                        "historia_fitness": history,
                        "detalles_generaciones": gen_details,
                        "estadisticas": stats
                    }

        except ValueError as e:
            error = "Por favor ingrese valores numéricos válidos en todos los campos"
        except Exception as e:
            error = f"Error inesperado: {str(e)}"

    return render_template("index.html", resultado=resultado, error=error, analysis_result=analysis_result)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    n = data['n']
    mu = data['mu']
    lambd = data['lambd']
    generations = data['generations']
    mutation_rate = data['mutation_rate']
    runs = data['runs']
    
    stats = run_experiments(mu, lambd, n, generations, mutation_rate, runs)
    
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True)