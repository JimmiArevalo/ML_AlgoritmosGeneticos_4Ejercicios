# ============================================================
# Clase base AlgoritmoGenetico 
# ============================================================
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

# Establecer una semilla para reproducibilidad 
random.seed(42)
np.random.seed(42)

"""## 9. Módulo 1: Clase `AlgoritmoGenetico`

Esta clase encapsula la lógica principal del Algoritmo Genético. Es genérica, parametrizable y contiene las implementaciones de los operadores genéticos fundamentales.

### **Espacio de búsqueda y mapeo: Codificación y Decodificación**

Aunque la clase `AlgoritmoGenetico` opera en el espacio cromosómico `S = {0, 1}^n` (genotipos), la interpretación en el espacio de soluciones reales `X` (fenotipos) se maneja a través de funciones de decodificación (`decode_func`) que se pasan como parámetro. La codificación (`e: X -> S`) es a menudo implícita en cómo se construye el espacio de genotipos inicial o cómo se genera la población aleatoria.

La función `e_tilde: S -> X` será proporcionada por cada problema específico.
"""

class AlgoritmoGenetico:
    """
    Implementación genérica de un Algoritmo Genético Canónico.

    Atributos:
        population_size (int): Número de individuos en la población.
        chromosome_length (int): Longitud del cromosoma (número de bits/genes).
        pc (float): Probabilidad de cruzamiento (crossover).
        pm (float): Probabilidad de mutación.
        elitism (bool): Si se aplica elitismo (preservar al mejor individuo).
        fitness_func (callable): Función que calcula la aptitud de un fenotipo.
        decode_func (callable): Función que decodifica un genotipo a un fenotipo.
        selection_method (str): Método de selección ('roulette' o 'tournament').
        tournament_size (int, opcional): Tamaño del torneo para la selección por torneo.

    Principios Bioinspirados en el código:
    *   **Población**: Lista de `individuos`.
    *   **Individuo / Cromosoma**: Representado como una lista de enteros (0 o 1).
    *   **Gen / Alelo**: Cada entero (0 o 1) en la lista del cromosoma.
    *   **Locus**: El índice de cada gen dentro del cromosoma.
    """

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        pc: float,
        pm: float,
        fitness_func: callable,
        decode_func: callable,
        selection_method: str = 'roulette',
        elitism: bool = True,
        tournament_size: int = 3,
    ):
        # Validación de parámetros para asegurar que son coherentes y evitan errores lógicos.
        if not (0 <= pc <= 1 and 0 <= pm <= 1):
            raise ValueError("pc y pm deben estar entre 0 y 1.")
        if population_size <= 0 or chromosome_length <= 0:
            raise ValueError("population_size y chromosome_length deben ser positivos.")
        if selection_method not in ['roulette', 'tournament']:
            raise ValueError("selection_method debe ser 'roulette' o 'tournament'.")

        # Asignación de parámetros a atributos de la instancia.
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.pc = pc
        self.pm = pm
        self.elitism = elitism
        self.fitness_func = fitness_func
        self.decode_func = decode_func
        self.selection_method = selection_method
        self.tournament_size = tournament_size

        # Atributos para almacenar el estado del algoritmo genético.
        self.population: list[list[int]] = [] # La población actual de individuos (genotipos binarios).
        self.max_fitness_history: list[float] = [] # Historial del fitness máximo por generación.
        self.avg_fitness_history: list[float] = [] # Historial del fitness promedio por generación.
        self.best_individual_genotype: list[int] = [] # El mejor genotipo encontrado hasta ahora.
        self.best_individual_fitness: float = -float('inf') # El fitness del mejor genotipo encontrado hasta ahora.

    def _initialize_population(self) -> None:
        """
        Inicializa la población con cromosomas binarios generados aleatoriamente.
        Corresponde a la fase inicial de generación de individuos aleatorios en una población.
        Cada individuo es una lista de bits (0 o 1) de longitud `chromosome_length`.
        """
        self.population = [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def _calculate_all_fitness(self, population: list[list[int]]) -> tuple[list[float], list]:
        """
        Calcula la aptitud (fitness) para cada individuo en la población.
        Para cada cromosoma (genotipo), primero lo decodifica a su representación real (fenotipo),
        y luego evalúa la aptitud de ese fenotipo utilizando `fitness_func`.

        Args:
            population (list[list[int]]): La población actual de genotipos.

        Returns:
            tuple[list[float], list]: Una tupla que contiene la lista de valores de aptitud
                                     y la lista de fenotipos decodificados.
        Principios Bioinspirados en el código:
        *   **Función de Aptitud (Fitness)**: Cuantifica la 'calidad' del individuo.
        *   **Fenotipo**: Se obtiene del genotipo para evaluar la aptitud.
        """
        fitness_values = []
        phenotypes = []
        for chromosome in population:
            phenotype = self.decode_func(chromosome) # Decodificación e_tilde: S -> X (genotipo a fenotipo)
            fitness = self.fitness_func(phenotype) # Evaluación del fenotipo
            fitness_values.append(fitness)
            phenotypes.append(phenotype)
        return fitness_values, phenotypes
    def _select_proportional(self, population: list[list[int]], fitness_values: list[float]) -> [list[int]]:
        """
        Realiza la selección de individuos utilizando el método de la Ruleta de Holland.
        Individuos con mayor aptitud tienen una mayor probabilidad de ser seleccionados para la siguiente generación.
        Maneja valores de aptitud no negativos mediante un desplazamiento si es necesario para evitar probabilidades negativas.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes a cada individuo.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección, con el mismo tamaño que la población original.

        Principios Bioinspirados en el código:
        *   **Selección Natural**: Individuos con mayor aptitud tienen mayor probabilidad de ser seleccionados.
        *   **Ruleta de Holland**: La probabilidad de selección P(b_i) = f(b_i) / sum(f_k).
        """
        # Manejo de aptitudes no negativas: desplazar si el mínimo es negativo.
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            # Si hay aptitudes negativas, las desplazamos para que sean todas >= 0.
            # Se añade un pequeño valor (1e-6) para evitar divisiones por cero si todas las aptitudes ajustadas fueran 0.
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            # Si todas son no negativas, se usan directamente.
            adjusted_fitness = fitness_values

        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            # Si todas las aptitudes son cero (después del ajuste), seleccionar aleatoriamente para evitar errores.
            return random.choices(population, k=self.population_size)

        # Calcular las probabilidades de selección proporcionales a la aptitud ajustada.
        selection_probabilities = [f / total_fitness for f in adjusted_fitness]

        # La selección crea el 'pool de apareamiento' para la próxima generación.
        new_population = random.choices(population, weights=selection_probabilities, k=self.population_size)
        return new_population

    def _select_tournament(self, population: list[list[int]], fitness_values: list[float]) -> list[list[int]]:
        """
        Realiza la selección de individuos utilizando el método por Torneo.
        En cada paso, se seleccionan aleatoriamente `tournament_size` individuos y el de mayor aptitud entre ellos es elegido.
        Este proceso se repite `population_size` veces para formar la nueva población.

        Args:
            population (list[list[int]]): La población actual.
            fitness_values (list[float]): Los valores de aptitud correspondientes.

        Returns:
            list[list[int]]: La nueva población (pool de apareamiento) después de la selección.

        Principios Bioinspirados en el código:
        *   **Selección Natural**: Individuos compiten; los 'más fuertes' (mayor aptitud) prevalecen.
        """
        new_population = []
        for _ in range(self.population_size):
            # Seleccionar 'tournament_size' individuos aleatoriamente para el torneo.
            tournament_contestants_indices = random.sample(range(self.population_size), self.tournament_size)

            # Encontrar el mejor individuo (el de mayor fitness) del torneo.
            best_contestant_index = tournament_contestants_indices[0]
            for i in tournament_contestants_indices:
                if fitness_values[i] > fitness_values[best_contestant_index]:
                    best_contestant_index = i
            # El ganador del torneo se añade a la nueva población (se realiza una copia profunda para evitar referencias).
            new_population.append(copy.deepcopy(population[best_contestant_index]))
        return new_population

    def _crossover_one_point(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Realiza el cruzamiento de un punto entre dos padres para producir dos hijos.
        Con una probabilidad `pc`, se elige un punto de corte aleatorio y los segmentos de los padres se intercambian.
        Si no se produce cruzamiento, los hijos son copias de los padres.

        Args:
            parent1 (list[int]): Genotipo del primer padre.
            parent2 (list[int]): Genotipo del segundo padre.

        Returns:
            tuple[list[int], list[int]]: Una tupla con los genotipos de los dos hijos resultantes.

        Principios Bioinspirados en el código:
        *   **Recombinación / Cruzamiento**: Intercambio de material genético entre padres.
        *   **Cromosomas homólogos**: Los cromosomas de los padres se alinean para el intercambio.
        """
        if random.random() < self.pc:
            # Punto de cruzamiento aleatorio (excluyendo los extremos 0 y chromosome_length).
            point = random.randint(1, self.chromosome_length - 1)
            # Se combinan los segmentos de los padres para formar los hijos.
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            # Si no hay cruzamiento, los hijos son copias idénticas de los padres.
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate_flip_bit(self, chromosome: list[int]) -> list[int]:
        """
        Realiza la mutación bit a bit en un cromosoma.
        Por cada gen en el cromosoma, con una probabilidad `pm`, su valor se invierte (0 a 1, o 1 a 0).

        Args:
            chromosome (list[int]): El genotipo a mutar.

        Returns:
            list[int]: El genotipo mutado.

        Principios Bioinspirados en el código:
        *   **Mutación**: Cambios aleatorios en el material genético (genotipo).
        *   **Alelo**: El valor de un gen cambia (0 a 1 o 1 a 0).
        """
        mutated_chromosome = copy.deepcopy(chromosome)
        for i in range(self.chromosome_length):
            if random.random() < self.pm:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]  # Invertir el bit (0->1, 1->0).
        return mutated_chromosome

    def _apply_elitism(self, new_population: list[list[int]]) -> list[list[int]]:
        """
        Aplica el mecanismo de elitismo, preservando al mejor individuo de la generación anterior
        (almacenado como `self.best_individual_genotype`) en la nueva población.
        El individuo con menor aptitud de la nueva población es reemplazado por el élite.

        Args:
            new_population (list[list[int]]): La población recién generada (después de crossover y mutación).

        Returns:
            list[list[int]]: La población con el individuo élite insertado.

        Principios Bioinspirados en el código:
        *   **Elitismo**: El individuo con mayor aptitud de la generación se garantiza que sobreviva a la siguiente.
        *   **Convergencia Monotónica**: El fitness máximo nunca decrece de una generación a otra.
        """
        if not self.best_individual_genotype: # Si aún no se ha encontrado un mejor individuo.
            return new_population # No hay élite para aplicar.

        # Encontrar el peor individuo en la nueva población para ser reemplazado.
        new_fitness_values, _ = self._calculate_all_fitness(new_population)
        worst_individual_index = np.argmin(new_fitness_values)

        # Reemplazar al peor individuo de la nueva población con una copia del élite de la generación anterior.
        new_population[worst_individual_index] = copy.deepcopy(self.best_individual_genotype)
        return new_population

    def run(self, num_generations: int) -> tuple[list[int], float]:
        """
        Ejecuta el Algoritmo Genético para un número dado de generaciones.
        Este es el bucle principal del AG, que coordina la inicialización, evaluación, selección, cruzamiento y mutación
        a lo largo de múltiples generaciones.

        Args:
            num_generations (int): El número de generaciones a ejecutar.

        Returns:
            tuple[list[int], float]: El genotipo del mejor individuo encontrado en toda la ejecución y su aptitud.
        """
        self._initialize_population() # Inicializa la población con individuos aleatorios.

        for generation in range(num_generations):
            # 1. Evaluar la aptitud de la población actual.
            fitness_values, _ = self._calculate_all_fitness(self.population)

            # Actualizar el mejor individuo global (élite) si se encuentra uno mejor en la generación actual.
            current_best_index = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_index]
            current_best_genotype = copy.deepcopy(self.population[current_best_index])

            if current_best_fitness > self.best_individual_fitness:
                self.best_individual_fitness = current_best_fitness
                self.best_individual_genotype = current_best_genotype

            # Registrar el historial de fitness máximo y promedio para la visualización.
            self.max_fitness_history.append(self.best_individual_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))

            # 2. Selección: Crear el 'pool de apareamiento'.
            if self.selection_method == 'roulette':
                mating_pool = self._select_proportional(self.population, fitness_values)
            else:  # tournament
                mating_pool = self._select_tournament(self.population, fitness_values)

            # 3. Cruzamiento y Mutación para crear la nueva generación.
            new_population = []
            # Asegurarse de que el 'mating_pool' tenga un número par de individuos para el cruzamiento por pares.
            if len(mating_pool) % 2 != 0:
                mating_pool.append(random.choice(mating_pool))

            random.shuffle(mating_pool) # Mezclar para formar pares aleatorios para el cruzamiento.

            # Realizar cruzamiento y mutación en pares de padres para generar hijos.
            for i in range(0, self.population_size, 2):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]

                child1, child2 = self._crossover_one_point(parent1, parent2) # Aplicar cruzamiento.

                child1 = self._mutate_flip_bit(child1) # Aplicar mutación al primer hijo.
                child2 = self._mutate_flip_bit(child2) # Aplicar mutación al segundo hijo.

                new_population.extend([child1, child2]) # Añadir los hijos a la nueva población.

            # Asegurarse de que la nueva población tenga el tamaño correcto (puede ser ligeramente mayor por el ajuste de pares).
            self.population = new_population[:self.population_size]

            # 4. Elitismo (si está habilitado): Asegura que el mejor individuo no se pierda.
            if self.elitism:
                self.population = self._apply_elitism(self.population)

        # Retornar el mejor individuo encontrado y su aptitud final.
        return self.best_individual_genotype, self.best_individual_fitness


# ============================================================
# Ejercicio 1
# ============================================================

"""## Ejercicio 1: Maximización de f(x) = x^3 - 4x^2 + 5x

Se adapta el caso didáctico visto en clase (f(x) = x^2) para maximizar la
función cúbica f(x) = x^3 - 4x^2 + 5x. Analíticamente, f'(x) = 3x^2 - 8x + 5,
cuyas raíces son x = 1 y x = 5/3. Evaluando la segunda derivada
f''(x) = 6x - 8, se confirma que x = 1 es un máximo local (f''(1) = -2 < 0)
y x = 5/3 es un mínimo local (f''(5/3) = 2 > 0).

Como la cúbica crece sin límite para x grandes, se acota el espacio de
búsqueda a X = [0, 1.8] para que el AG converja claramente al máximo local
en x = 1 (f(1) = 2), sin que la rama ascendente de la función lo opaque
(en ese rango, f(1.8) ≈ 1.87 < 2).

### Espacio de Búsqueda y Mapeo:
*   **Genotipo (S)**: Cadenas binarias de 6 bits -> 2^6 = 64 niveles de resolución.
*   **Fenotipo (X)**: Números reales en el rango [0, 1.8].
*   **Decodificación (e_tilde)**: Se convierte el genotipo a entero (0-63) y
    se reescala linealmente al rango [X_MIN_CUBIC, X_MAX_CUBIC].
"""

# --- Definición de funciones específicas para el Ejercicio 1 ---

CHROM_LEN_CUBIC = 6      # 6 bits -> 64 posibles valores discretos de x.
X_MIN_CUBIC = 0.0        # Límite inferior del espacio de búsqueda.
X_MAX_CUBIC = 1.8        # Límite superior: evita que la rama ascendente domine al máximo local.

def decode_cubic(genotype: list[int]) -> float:
    """
    Decodifica un genotipo binario a un valor real x en el rango [X_MIN_CUBIC, X_MAX_CUBIC].
    Corresponde a la función e_tilde: S -> X.
    Primero convierte la cadena de bits a un entero, y luego reescala
    linealmente ese entero al rango real deseado.
    """
    max_int = 2 ** CHROM_LEN_CUBIC - 1  # Máximo entero representable (63 para 6 bits).
    decimal_value = int("".join(map(str, genotype)), 2)
    # Reescalado lineal de [0, max_int] a [X_MIN_CUBIC, X_MAX_CUBIC].
    x = X_MIN_CUBIC + (decimal_value / max_int) * (X_MAX_CUBIC - X_MIN_CUBIC)
    return x

def fitness_cubic(x: float) -> float:
    """
    Calcula la aptitud para el Ejercicio 1: maximizar f(x) = x^3 - 4x^2 + 5x.
    La aptitud es directamente el valor de la función objetivo, ya que f(x)
    es no negativa en todo el rango de búsqueda elegido [0, 1.8].
    """
    return x**3 - 4 * x**2 + 5 * x

# --- Parámetros del AG para el Ejercicio 1 ---
POP_SIZE_CUBIC = 20          # Tamaño de la población.
PC_CUBIC = 0.8               # Probabilidad de cruzamiento del 80%.
PM_CUBIC = 0.02              # Probabilidad de mutación del 2% por bit.
NUM_GENERATIONS_CUBIC = 40   # Número de generaciones a ejecutar.

print("\n--- Ejecutando AG para el Ejercicio 1: f(x) = x^3 - 4x^2 + 5x ---")
# Se reutiliza la clase AlgoritmoGenetico definida en el Módulo 1 del notebook.
cubic_ga = AlgoritmoGenetico(
    population_size=POP_SIZE_CUBIC,
    chromosome_length=CHROM_LEN_CUBIC,
    pc=PC_CUBIC,
    pm=PM_CUBIC,
    fitness_func=fitness_cubic,
    decode_func=decode_cubic,
    selection_method='roulette',
    elitism=True,
)

# Ejecutar el Algoritmo Genético para el número especificado de generaciones.
final_best_genotype_cubic, final_best_fitness_cubic = cubic_ga.run(NUM_GENERATIONS_CUBIC)
# Decodificar el mejor genotipo encontrado para obtener el valor real de x.
final_best_x_cubic = decode_cubic(final_best_genotype_cubic)

print(f"\n--- Resultados Finales Ejercicio 1 ---")
print(f"Mejor Genotipo: {''.join(map(str, final_best_genotype_cubic))}")
print(f"Mejor x encontrado: {final_best_x_cubic:.4f}")
print(f"Mejor Fitness f(x): {final_best_fitness_cubic:.4f}")
print(f"Óptimo teórico esperado: x = 1.0, f(1) = 2.0")

# --- Gráfica de Convergencia: Ejercicio 1 ---
plt.figure(figsize=(6, 5))
plt.plot(range(1, NUM_GENERATIONS_CUBIC + 1), cubic_ga.max_fitness_history, label='Fitness Máximo')
plt.plot(range(1, NUM_GENERATIONS_CUBIC + 1), cubic_ga.avg_fitness_history, label='Fitness Promedio')
plt.title('Convergencia AG: Ejercicio 1 - f(x) = x^3 - 4x^2 + 5x')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
