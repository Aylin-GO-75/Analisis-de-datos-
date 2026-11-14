import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el archivo de texto
try:
    data = np.loadtxt('datos_ADN.txt')
except OSError:
    print("No se encontró el archivo 'datos_ADN.txt'. Asegúrate de que esté en la misma carpeta.")
    exit()

x = data[:, 0]
y = data[:, 1]

# 2. Configurar el Binning (Medias y Desviaciones Estándar en X e Y)
num_bins = 12  
bins = np.linspace(x.min(), x.max(), num_bins + 1)
bin_indices = np.digitize(x, bins) - 1

bin_centers_x = []
std_x = []
mean_y = []
std_y = []

for i in range(num_bins):
    mask = (bin_indices == i)
    if np.sum(mask) > 1:
        bin_centers_x.append(np.mean(x[mask]))
        std_x.append(np.std(x[mask], ddof=1))
        mean_y.append(np.mean(y[mask]))
        std_y.append(np.std(y[mask], ddof=1))

bin_centers_x = np.array(bin_centers_x)
std_x = np.array(std_x)
mean_y = np.array(mean_y)
std_y = np.array(std_y)

# Evitar divisiones por cero
std_x[std_x == 0] = np.mean(std_x[std_x > 0]) if np.any(std_x > 0) else 1.0
std_y[std_y == 0] = np.mean(std_y[std_y > 0]) if np.any(std_y > 0) else 1.0

# 3. Ajuste Lineal Ponderado y obtención de la Matriz de Covarianza (V)
weights = 1.0 / (std_y ** 2)
p, V = np.polyfit(bin_centers_x, mean_y, 1, w=np.sqrt(weights), cov=True)
m, b = p  # m = pendiente, b = ordenada al origen

# 4. Cálculo de la Chi-cuadrada y Chi-cuadrada reducida (chi2/ndf <= 1)
y_fit = m * bin_centers_x + b
chi2 = np.sum(((mean_y - y_fit) / std_y) ** 2)
ndf = len(bin_centers_x) - 2  # Grados de libertad (N_bins - 2 parámetros)
chi2_reducida = chi2 / ndf

# 5. SECCIÓN 6: Propagación analítica de errores para un valor específico de X
# Puedes cambiar este valor de x por el que te pida tu práctica o guía
x_evaluar = 15.0  

# Cálculo analítico de y y su incertidumbre sigma_y usando la matriz de covarianza (V)
# V[0,0] = var(m), V[1,1] = var(b), V[0,1] = cov(m,b)
y_evaluado = m * x_evaluar + b
sigma_y_evaluado = np.sqrt(V[0,0] * (x_evaluar ** 2) + V[1,1] + 2 * V[0,1] * x_evaluar)

# Imprimir resultados detallados en consola
print("--- RESULTADOS DEL ANÁLISIS ---")
print(f"Pendiente (m): {m:.4f} ± {np.sqrt(V[0,0]):.4f}")
print(f"Ordenada al origen (b): {b:.4f} ± {np.sqrt(V[1,1]):.4f}")
print(f"Chi-cuadrada reducida (chi2/ndf): {chi2_reducida:.4f}")
print("\n--- SECCIÓN 6: PROPAGACIÓN DE ERRORES ---")
print(f"Para el valor de X = {x_evaluar} Å:")
print(f"  Valor evaluado Y = {y_evaluado:.4f} ± {sigma_y_evaluado:.4f} Å")

# 6. Gráfica Profesional con Cruz de Errores y Banda de Confianza
x_line = np.linspace(bin_centers_x.min(), bin_centers_x.max(), 200)
y_line = m * x_line + b
sigma_y_line = np.sqrt(V[0,0] * (x_line ** 2) + V[1,1] + 2 * V[0,1] * x_line)

plt.figure(figsize=(9, 6))
plt.errorbar(bin_centers_x, mean_y, xerr=std_x, yerr=std_y, fmt='ro', capsize=4, elinewidth=1.2, 
             label='Medias por Bin (con $\\sigma_x$ y $\\sigma_y$)')
plt.plot(x_line, y_line, 'b-', linewidth=2, 
         label=f'Ajuste Lineal Ponderado\n$\\chi^2/ndf = {chi2_reducida:.3f}$')
plt.fill_between(x_line, y_line - sigma_y_line, y_line + sigma_y_line, color='blue', alpha=0.15,
                 label='Propagación $y(x) \pm \\sigma_y$')

# Resaltar el punto evaluado de la Sección 6 en la gráfica
plt.plot(x_evaluar, y_evaluado, 'go', markersize=8, label=f'Punto evaluado X={x_evaluar}')

plt.xlabel('Posición X (Ångströms)', fontsize=12)
plt.ylabel('Posición Y (Ångströms)', fontsize=12)
plt.title('Análisis Estadístico, Chi-cuadrada y Propagación de Errores (PDB 1BNA)', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()