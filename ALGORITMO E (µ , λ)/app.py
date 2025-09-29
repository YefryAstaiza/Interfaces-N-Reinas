from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import random
import io
import base64
import time

app = Flask(__name__)
app.secret_key = 'nreinas_secret_key' # Necesario para usar sesiones

# --- Funciones del algoritmo (fitness, mutación, cruce, etc.) ---
def fitness_function(queens):
    n = len(queens)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(queens[i] - queens[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def generate_random_board(n):
    return np.random.permutation(n)

def mutate(individual, mutation_rate=0.8):
    if random.random() < mutation_rate:
        n = len(individual)
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    point1, point2 = sorted(random.sample(range(n), 2))
    child1 = parent2.copy()
    child2 = parent1.copy()
    mapping1, mapping2 = {}, {}
    for i in range(point1, point2 + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        mapping1[parent1[i]] = parent2[i]
        mapping2[parent2[i]] = parent1[i]
    for i in list(range(0, point1)) + list(range(point2 + 1, n)):
        while child1[i] in parent1[point1:point2 + 1]:
            child1[i] = mapping1.get(child1[i], child1[i])
        while child2[i] in parent2[point1:point2 + 1]:
            child2[i] = mapping2.get(child2[i], child2[i])
    return child1, child2

def evolutionary_strategy_n_queens(mu, lambd, n, generations, mutation_rate=0.8):
    if lambd <= mu:
        raise ValueError("λ debe ser mayor que μ")
    
    population = [generate_random_board(n) for _ in range(mu)]
    best_solution = None
    log = []
    generations_to_solution = generations  # Por defecto, asumimos que no encontró solución

    log.append("=" * 60)
    log.append(f"PROBLEMA DE LAS {n}-REINAS")
    log.append("=" * 60)
    log.append(f"Estrategia Evolutiva (μ={mu}, λ={lambd}, Mutación={mutation_rate})")
    log.append(f"Generaciones máximas: {generations}")
    log.append("")
    
    log.append("POBLACIÓN INICIAL (primeros 3 individuos):")
    log.append("-" * 40)
    for i in range(min(3, mu)):
        fitness_val = fitness_function(population[i])
        log.append(f"Individuo {i+1}: {population[i].tolist()}")
        log.append(f"  → Conflictos: {fitness_val}")
    log.append("")

    for gen in range(generations):
        offspring = []
        for _ in range(lambd // 2):
            p1, p2 = np.random.randint(mu), np.random.randint(mu)
            while p2 == p1:
                p2 = np.random.randint(mu)
            child1, child2 = pmx_crossover(population[p1], population[p2])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            offspring.extend([child1, child2])
        if len(offspring) < lambd:
            p1, p2 = np.random.randint(mu), np.random.randint(mu)
            child1, _ = pmx_crossover(population[p1], population[p2])
            child1 = mutate(child1, mutation_rate)
            offspring.append(child1)
        fitness_values = np.array([fitness_function(ind) for ind in offspring])
        best_indices = np.argsort(fitness_values)[:mu]
        population = [offspring[i] for i in best_indices]
        best_fitness = fitness_values[best_indices[0]]
        if best_solution is None or best_fitness < fitness_function(best_solution):
            best_solution = offspring[best_indices[0]].copy()
        
        # Log cada 10 generaciones o cuando hay mejora significativa
        if (gen + 1) % 10 == 0 or best_fitness == 0 or (gen < 10 and (gen + 1) % 2 == 0):
            log.append(f"Generación {gen + 1:3d}: Mejor fitness = {best_fitness:2d}")
            
        if best_fitness == 0:
            log.append("")
            log.append("🎉 ¡SOLUCIÓN ÓPTIMA ENCONTRADA!")
            log.append(f"   Generación: {gen + 1}")
            generations_to_solution = gen + 1
            break

    log.append("")
    log.append("=" * 60)
    log.append("RESULTADOS FINALES")
    log.append("=" * 60)
    log.append(f"Mejor solución encontrada: {best_solution.tolist()}")
    log.append(f"Número de conflictos: {fitness_function(best_solution)}")
    
    if fitness_function(best_solution) == 0:
        log.append("Estado: ✅ SOLUCIÓN ÓPTIMA")
    elif fitness_function(best_solution) <= 2:
        log.append("Estado: ⚡ BUENA SOLUCIÓN (pocos conflictos)")
    else:
        log.append("Estado: ⚠️  SOLUCIÓN CON CONFLICTOS")
    
    return best_solution, fitness_function(best_solution), log, generations_to_solution

# --- Función para convertir el tablero a imagen base64 ---
def board_to_base64(queens):
    n = len(queens)
    # Colores del tablero estilo ajedrez
    board_colors = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board_colors[i, j] = [1, 0.9, 0.8]  # Beige
            else:
                board_colors[i, j] = [0.3, 0.3, 0.3]  # Negro

    plt.figure(figsize=(5,5))
    plt.imshow(board_colors, interpolation='none')
    
    # Dibujar las reinas
    for col, row in enumerate(queens):
        color = 'black' if (row + col) % 2 == 0 else 'white'  # Contraste con el fondo
        plt.text(col, row, '♛', ha='center', va='center', fontsize=28, color=color)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# --- Inicializar historial en la sesión ---
@app.before_request
def initialize_history():
    if 'execution_history' not in session:
        session['execution_history'] = []
    else:
        # Limpiar historial antiguo que no tenga execution_time
        cleaned_history = []
        for record in session['execution_history']:
            if 'execution_time' in record:
                cleaned_history.append(record)
        session['execution_history'] = cleaned_history

# --- Ruta principal ---
@app.route('/', methods=['GET', 'POST'])
def index():
    best_solution = None
    best_fit = None
    tablero_img = None
    log = []
    generations_to_solution = 0
    analysis_data = {}
    execution_time = 0

    # Valores por defecto para GET request
    n = 8
    mu = 30
    lambd = 70
    generations = 50
    mutation_rate = 0.1

    if request.method == 'POST':
        try:
            n = int(request.form['n'])
            mu = int(request.form['mu'])
            lambd = int(request.form['lambd'])
            generations = int(request.form['generations'])
            mutation_rate = float(request.form['mutation_rate'])

            # Medir tiempo de ejecución
            start_time = time.time()
            
            best_solution, best_fit, log, generations_to_solution = evolutionary_strategy_n_queens(
                mu, lambd, n, generations, mutation_rate
            )
            
            execution_time = time.time() - start_time
            best_solution = best_solution.tolist()
            tablero_img = board_to_base64(best_solution)
            
            # Guardar en el historial
            execution_record = {
                'n': n,
                'mu': mu,
                'lambd': lambd,
                'generations': generations_to_solution,
                'conflicts': best_fit,
                'mutation_rate': mutation_rate,
                'execution_time': round(execution_time, 4),
                'solution_found': best_fit == 0
            }
            
            # Actualizar el historial en la sesión
            history = session.get('execution_history', [])
            history.append(execution_record)
            session['execution_history'] = history
            
            # Preparar datos para el análisis
            analysis_data = {
                'n': n,
                'mu': mu,
                'lambd': lambd,
                'generations_to_solution': generations_to_solution,
                'mutation_rate': mutation_rate,
                'conflicts': best_fit,
                'solution_found': best_fit == 0,
                'total_generations': generations,
                'execution_time': execution_time
            }
            
        except Exception as e:
            best_solution = []
            best_fit = f"Error: {str(e)}"
            tablero_img = None
            log.append(f"Error: {str(e)}")

    # Obtener historial de ejecuciones y revertir el orden
    execution_history = session.get('execution_history', [])
    # Revertir el orden para mostrar los más recientes primero
    execution_history_reversed = list(reversed(execution_history))

    return render_template(
        'index.html',
        best_solution=best_solution,
        best_fit=best_fit,
        tablero_img=tablero_img,
        log=log,
        execution_history=execution_history_reversed,
        analysis_data=analysis_data,
        execution_time=execution_time,
        n=n,
        mu=mu,
        lambd=lambd,
        generations=generations,
        mutation_rate=mutation_rate
    )

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['execution_history'] = []
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)