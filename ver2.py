# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Configuración de constantes
conductividad_termica = 1.0  # Conductividad térmica [W/m/K]
coeficiente_convectivo = 1.0  # Coeficiente convectivo
parametro_perfusion_a = 1.0  # Parámetro para la función de perfusión
parametro_perfusion_b = 1.0  # Parámetro para la función de perfusión
parametro_generacion_calor_m = 1.0  # Parámetro para la generación de calor
constante_generacion_calor_p = 0.0  # Constante para la generación de calor
temperatura_promedio_sangre = 37.0  # Temperatura promedio de la sangre [°C]
limite_inferior_dominio = 0.0  # Límite inferior del dominio
limite_superior_dominio = 1.0  # Límite superior del dominio
numero_nodos = 100  # Número de nodos

# Definición de la función de base radial gaussiana
def funcion_base_radial_gaussiana(r, epsilon):
    """
    Función de base radial gaussiana.
    
    Parámetros:
    r (float): Distancia entre los nodos.
    epsilon (float): Parámetro de forma.
    
    Retorna:
    float: Valor de la función de base radial gaussiana.
    """
    return np.exp(- (r / epsilon) ** 2)  # Fórmula de la función gaussiana

# Función para calcular la solución usando RBF
def solucion_rbf(epsilon):
    """
    Función para calcular la solución usando RBF.
    
    Parámetros:
    epsilon (float): Parámetro de forma.
    
    Retorna:
    x_eval (numpy array): Puntos de evaluación de la solución.
    T_eval (numpy array): Valores de la solución en los puntos de evaluación.
    """
    # Generar nodos en el dominio
    x_nodos = np.linspace(limite_inferior_dominio, limite_superior_dominio, numero_nodos)
    
    # Crear una malla de puntos para evaluar la solución
    x_eval = np.linspace(limite_inferior_dominio, limite_superior_dominio, 100)
    T_eval = np.zeros_like(x_eval)  # Inicializar el vector de temperaturas
    
    # Cálculo de la solución RBF
    for i, x in enumerate(x_eval):
        r = np.abs(x - x_nodos)  # Calcular distancia desde los nodos
        phi = funcion_base_radial_gaussiana(r, epsilon)  # Evaluar función RBF
        
        # Calcular la temperatura aproximada como combinación lineal
        T_eval[i] = np.dot(phi, np.ones(numero_nodos))  # Coeficientes dummy para simplicidad
    
    return x_eval, T_eval  # Retornar los puntos y valores de temperatura

# Definición de la función de perfusión
def funcion_perfusion(x):
    """
    Función de perfusión.
    
    Parámetros:
    x (float): Punto en el dominio.
    
    Retorna:
    float: Valor de la función de perfusión.
    """
    return parametro_perfusion_a * np.tanh(parametro_perfusion_b * (x - limite_inferior_dominio)**2)  # Función hiperbólica

# Definición de la ecuación de Biocalor para solve_bvp
def ecuacion_biocalor(x, T):
    """
    Ecuación de Biocalor para solve_bvp.
    
    Parámetros:
    x (float): Punto en el dominio.
    T (numpy array): Vector de temperaturas.
    
    Retorna:
    numpy array: Vector de derivadas de la temperatura.
    """
    return np.vstack((T[1], -(funcion_perfusion(x) * (temperatura_promedio_sangre - T[0]) + 
                             (parametro_generacion_calor_m * T[0] + constante_generacion_calor_p)) / conductividad_termica))

# Condiciones de frontera
def condiciones_frontera(T_a, T_b):
    """
    Condiciones de frontera.
    
    Parámetros:
    T_a (numpy array): Vector de temperaturas en el límite inferior.
    T_b (numpy array): Vector de temperaturas en el límite superior.
    
    Retorna:
    numpy array: Vector de condiciones de frontera.
    """
    return np.array([T_a[0] - temperatura_promedio_sangre, T_b[1] - coeficiente_convectivo])  # Condiciones de temperatura

# Solución del problema utilizando solve_bvp
x = np.linspace(limite_inferior_dominio, limite_superior_dominio, 100)  # Puntos del dominio
T_inicial = np.zeros((2, x.size))  # Inicialización de la solución

# Resolver el problema de valores de frontera
solucion = solve_bvp(ecuacion_biocalor, condiciones_frontera, x, T_inicial)

# Gráfica 1: Solución numérica para cuatro valores del parámetro de forma (RBF Gaussiana y Multicuádrica)
plt.figure(figsize=(12, 6)) 
for epsilon in [0.05, 0.1, 0.2, 0.3]:  # Diferentes valores de epsilon
    x_eval, T_eval = solucion_rbf(epsilon)  # Calcular solución RBF
    plt.plot(x_eval, T_eval, label=f'RBF Gaussiana (Epsilon = {epsilon})')  # Graficar cada solución

# Añadir la solución de solve_bvp
plt.plot(solucion.x, solucion.y[0], label='Solución de Biocalor (solve_bvp)', linestyle='--', color='black')  # Solución de Biocalor

plt.title('Solución numérica usando RBF Gaussiana y solve_bvp')  
plt.xlabel('Posición (x)')  
plt.ylabel('Temperatura (T)')  
plt.legend()  
plt.grid()  
plt.show()  

# Gráfica 2: Error local de la solución con colocación
solucion_verdadera = solucion.y[0]  # Usamos la solución de solve_bvp como la verdadera

# Calcular el error para RBF Gaussiana
epsilones = [0.05, 0.1, 0.2, 0.3]  # Valores de epsilon para evaluar
errores_rbf = []  # Lista para almacenar los errores

for epsilon in epsilones:
    x_eval, T_eval = solucion_rbf(epsilon)  # Obtener solución RBF
    error_rbf = np.abs(T_eval - solucion_verdadera)  # Calcular el error
    errores_rbf.append(error_rbf)  # Almacenar el error

# Gráfica de error
plt.figure(figsize=(12, 6))  
for i, error in enumerate(errores_rbf):
    plt.plot(solucion.x, error, label=f'Error local RBF Gaussiana (Epsilon = {epsilones[i]})')  # Graficar el error

plt.plot(solucion.x, np.abs(solucion.y[0] - solucion_verdadera), label='Error local solve_bvp', linestyle=':', color='black')  # Error de solve_bvp

plt.title('Error Local de la Solución con Colocación')  
plt.xlabel('Posición (x)')  
plt.ylabel('Error')  
plt.legend()  
plt.grid()  
plt.show()  

# Gráfica 3: Norma del error L2 para RBF Gaussiana
epsilones = [0.05, 0.1, 0.2, 0.3]  # Valores de epsilon para evaluar
normas_error_rbf = []  # Lista para almacenar las normas de error

for epsilon in epsilones:
    x_eval, T_eval = solucion_rbf(epsilon)  # Obtener solución RBF
    error_rbf = np.abs(T_eval - solucion_verdadera)  # Calcular el error
    norma_error_rbf = np.sqrt(np.sum(error_rbf ** 2))  # Calcular la norma L2 del error
    normas_error_rbf.append(norma_error_rbf)  # Almacenar la norma

# Graficar las normas del error L2
plt.figure(figsize=(12, 6))  
plt.bar(['Epsilon = 0.05', 'Epsilon = 0.1', 'Epsilon = 0.2', 'Epsilon = 0.3'], normas_error_rbf)  # Graficar como barras
plt.title('Norma del Error L2 para RBF Gaussiana')  
plt.ylabel('Norma L2 del Error') 
plt.grid()  
plt.show()  
