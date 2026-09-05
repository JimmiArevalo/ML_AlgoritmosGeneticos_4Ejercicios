# MACHINE LEARNING - ALGORITMOS GENETICOS

**Universidad de Cundinamarca - Seccional Ubaté**
**Programa de Ingeniería de Sistemas y Computación**
**Docente:** Fabio Alejandro Sastoque Rincón

## Descripción del proyecto

Este repositorio contiene el desarrollo de la actividad de laboratorio *"Exploración y Optimización Matemática"*, en la cual se modifica el código base de un algoritmo genético para adaptarlo a distintos paisajes matemáticos, ajustando las funciones de decodificación, aptitud, mutación y elitismo.

## Requisitos

- Python 3.x
- Jupyter (extensión instalada en Visual Studio Code)
- Entorno virtual (`venv` o `env`)

## Configuración del entorno

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows)
venv\Scripts\activate

# Activar entorno virtual (Linux/Mac)
source venv/bin/activate

# Instalar dependencias necesarias
pip install numpy matplotlib
```

> Se recomienda excluir el entorno virtual del repositorio mediante un archivo `.gitignore`.

## Estructura del repositorio

```
├── ejercicio_1.py        # Maximización cúbica
├── ejercicio_2.py        # Optimización multivariable
├── ejercicio_3.py        # Impacto de la tasa de mutación
├── ejercicio_4.py        # Elitismo ampliado
└── README.md
```

## Ejercicios desarrollados

### Ejercicio 1 - Maximización Cúbica
Se adapta la función de aptitud para maximizar la ecuación **f(x) = x³ - 4x² + 5x**, definiendo un espacio de búsqueda donde la función presenta un máximo local claro. Se ajustó la decodificación binaria del cromosoma para mapear correctamente a dicho rango.

### Ejercicio 2 - Optimización Multivariable
Se modifica el cromosoma para soportar dos variables (**x** e **y**), dividiendo la cadena binaria en dos mitades (4 bits para x, 4 bits para y). Se minimiza la función **f(x, y) = x² + y²**, devolviendo el valor negativo en la función de aptitud.

### Ejercicio 3 - Impacto de la Tasa de Mutación
Se ejecuta el algoritmo del Ejercicio 1 en tres escenarios distintos, variando únicamente la probabilidad de mutación:

- `pm = 0.01`
- `pm = 0.1`
- `pm = 0.5`

Se generan las respectivas gráficas de convergencia para comparar el efecto de cada tasa sobre la evolución de la población.

### Ejercicio 4 - Elitismo Ampliado
Se modifica la función interna del algoritmo genético encargada del elitismo, de forma que en lugar de preservar únicamente al mejor individuo, se preserven de manera intacta **los 3 mejores individuos** de la generación actual hacia la siguiente generación.

## Flujo de trabajo con Git

El desarrollo se realizó siguiendo buenas prácticas de control de versiones:

- Cada ejercicio se desarrolló en una rama independiente a partir de `main` (ej. `feature/ejercicio4-elitismo`).
- Los cambios se integraron a `main` mediante **Pull Requests**.
- Los commits describen de forma clara los cambios realizados en cada ejercicio.

## Autor(es)

- Jimmi Arevalo

## Cómo ejecutar

1. Clonar el repositorio:
   ```bash
   git clone <URL-del-repositorio>
   ```
2. Activar el entorno virtual y instalar dependencias (ver sección de configuración).
3. Abrir los archivos `.py` o los notebooks de Jupyter en Visual Studio Code.
4. Ejecutar cada ejercicio de forma independiente para observar los resultados y las gráficas de convergencia.
