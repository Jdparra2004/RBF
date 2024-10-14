# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Configuración de constantes
k = 1.0         # Conductividad térmica [W/m/K]
omega = 1.0     # Coeficiente convectivo
a = 1.0         # Parámetro para la función de perfusión
b = 1.0         # Parámetro para la función de perfusión
m = 1.0         # Parámetro para la generación de calor
p = 0.0         # Constante para la generación de calor
T_s = 37.0      # Temperatura promedio de la sangre [°C]
x_a = 0.0       # Límite inferior del dominio
x_b = 1.0       # Límite superior del dominio
N = 5           # Número de nodos

# Definición de la función de base radial Gaussiana
def gaussian_rbf(r, epsilon):
    return np.exp(- (r / epsilon) ** 2)

# Definición de la función de base radial multicuádrica
def multiquadric_rbf(r, c):
    return np.sqrt(r**2 + c**2)

# Función para calcular la solución usando RBF
def rbf_solution(method='gaussian', epsilon=0.1, c=0.1):
    # Generar nodos
    x_nodes = np.linspace(x_a, x_b, N)
    
    # Crear una malla de puntos para evaluar la solución
    x_eval = np.linspace(x_a, x_b, 100)
    T_eval = np.zeros_like(x_eval)
    
    # Cálculo de la solución RBF
    for i, x in enumerate(x_eval):
        r = np.abs(x - x_nodes)
        if method == 'gaussian':
            phi = gaussian_rbf(r, epsilon)
        elif method == 'multiquadric':
            phi = multiquadric_rbf(r, c)
        else:
            raise ValueError("Método no reconocido")
        
        # Calcular la temperatura aproximada
        T_eval[i] = np.dot(phi, np.ones(N))  # Coeficientes dummy para simplicidad
    
    return x_eval, T_eval

# Definición de la función de perfusión
def c(x):
    return a * np.tanh(b * (x - x_a)**2)

# Definición de la ecuación de Biocalor para solve_bvp
def biocalor_equation(x, T):
    return np.vstack((T[1], -(c(x) * (T_s - T[0]) + (m * T[0] + p)) / k))

# Condiciones de frontera
def boundary_conditions(T_a, T_b):
    return np.array([T_a[0] - T_s, T_b[1] - omega])

# Solución del problema utilizando solve_bvp
x = np.linspace(x_a, x_b, 100)
T_initial = np.zeros((2, x.size))  # Inicialización

# Resolver el problema de valores de frontera
solution = solve_bvp(biocalor_equation, boundary_conditions, x, T_initial)

# Gráfica 1: Solución numérica para cuatro valores del parámetro de forma (RBF Gaussiana y Multicuádrica)
plt.figure(figsize=(12, 6))
for epsilon in [0.05, 0.1, 0.2, 0.3]:
    x_eval, T_eval = rbf_solution(method='gaussian', epsilon=epsilon)
    plt.plot(x_eval, T_eval, label=f'RBF Gaussiana (Epsilon = {epsilon})')

for c in [0.05, 0.1, 0.2, 0.3]:
    x_eval, T_eval = rbf_solution(method='multiquadric', c=c)
    plt.plot(x_eval, T_eval, label=f'RBF Multicuádrica (c = {c})')

# Añadir la solución de solve_bvp
plt.plot(solution.x, solution.y[0], label='Solución de Biocalor (solve_bvp)', linestyle='--', color='black')

plt.title('Solución numérica usando RBF Gaussiana y Multicuádrica, y solve_bvp')
plt.xlabel('Posición (x)')
plt.ylabel('Temperatura (T)')
plt.legend()
plt.grid()
plt.savefig('solucion_numerica.png')  # Guardar figura
plt.show()

# Gráfica 2: Error local de la solución con colocación
true_solution = solution.y[0]  # Usamos la solución de solve_bvp como la verdadera
error_gaussian = np.zeros_like(true_solution)
error_multiquadric = np.zeros_like(true_solution)

# Calcular el error para RBF Gaussiana
for epsilon in [0.05, 0.1, 0.2, 0.3]:
    x_eval, T_eval = rbf_solution(method='gaussian', epsilon=epsilon)
    error_gaussian += np.abs(T_eval - true_solution)  # Acumulando errores

# Calcular el error para RBF Multicuádrica
for c in [0.05, 0.1, 0.2, 0.3]:
    x_eval, T_eval = rbf_solution(method='multiquadric', c=c)
    error_multiquadric += np.abs(T_eval - true_solution)  # Acumulando errores

# Calcular el error de solve_bvp
error_bvp = np.abs(solution.y[0] - true_solution)

# Gráfica de error
plt.figure(figsize=(12, 6))
plt.plot(solution.x, error_gaussian / 4, label='Error local RBF Gaussiana', color='blue')  # Dividir por 4 para el promedio
plt.plot(solution.x, error_multiquadric / 4, label='Error local RBF Multicuádrica', linestyle='--', color='orange')
plt.plot(solution.x, error_bvp, label='Error local solve_bvp', linestyle=':', color='black')

plt.title('Error Local de la Solución con Colocación')
plt.xlabel('Posición (x)')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.savefig('error_local.png')  # Guardar figura
plt.show()

# Gráfica 3: Norma del error L2 para RBF Gaussiana y Multicuádrica
L2_norm_gaussian = np.sqrt(np.sum((error_gaussian / 4) ** 2))  # Error promedio
L2_norm_multiquadric = np.sqrt(np.sum((error_multiquadric / 4) ** 2))  # Error promedio
L2_norm_bvp = np.sqrt(np.sum(error_bvp**2))  # Error para solve_bvp

plt.figure(figsize=(12, 6))
plt.bar(['RBF Gaussiana', 'RBF Multicuádrica', 'solve_bvp'], [L2_norm_gaussian, L2_norm_multiquadric, L2_norm_bvp])
plt.title('Norma del Error L2 para RBF Gaussiana, Multicuádrica y solve_bvp')
plt.ylabel('Norma L2 del Error')
plt.grid()
plt.savefig('norma_error_L2.png')  # Guardar figura
plt.show()
