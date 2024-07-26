import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import time

# Define parameters
Lx = 1  # Length in X-direction (m)
nx = 10  # Number of nodes in X-direction
dx = Lx / (nx - 1)  # Step size in X-direction
Ly = 1  # Length in Y-direction (m)
ny = 10  # Number of nodes in Y-direction
dy = Ly / (ny - 1)  # Step size in Y-direction
z = 0.001 #thickness(m)
Ta = 300  # Ambient temperature (K)
K = 200 # Thermal conductivity of aluminum (W/mK)
e = 0.9  # Emissivity
s = 5.67 * 10**(-8)  # Stefan-Boltzmann constant (W/m²K⁴)
q1 = 600 # Heat flux at the center (W/m²) - increased for more noticeable effect
q2 = 1000 # Heat flux at the center (W/m²) - increased for more noticeable effect
relaxation_factor = 0.9  # Relaxation factor for convergence
convergence_criterion = 1e-5  # Convergence criterion

# Initialize temperature field
T = np.zeros([nx, ny]) + Ta

# Maximum number of iterations
max_iterations = 2000

# Initialize the progress bar
pbar = tqdm(total=max_iterations, desc="Iterating", unit="iteration")

# Start the timer
start_time = time.time()

# Iterate to solve the temperature field
converged = False
for l in range(max_iterations):
    T_old = T.copy()
    # Define coefficients
    aw = ae = K * z*dy / dx
    an = asn = K * z*dx / dy

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            b = 2 * e * s * (3 * T_old[i, j]**4 + Ta**4) * dx * dy
            # Edge boundary conditions
            T[0, j] = (ae * T[1, j] + an * T[0, j + 1] * 0.5 + asn * T[0, j - 1] * 0.5 + b / 2) / (asn * 0.5 + an * 0.5 + ae + 4 * e * s * T_old[i, j]**3 * dx * dy)
            T[i, 0] = (ae * T[i + 1, 0] * 0.5 + aw * T[i - 1, 0] * 0.5 + an * T[i, 1] + b / 2) / (aw * 0.5 + an + ae * 0.5 + 4 * e * s * T_old[i, j]**3 * dx * dy)
            T[nx - 1, j] = (aw * T[nx - 2, j] + an * T[nx - 1, j + 1] * 0.5 + asn * T[nx - 1, j - 1] * 0.5 + b / 2) / (aw + an * 0.5 + asn * 0.5 + 4 * e * s * T_old[i, j]**3 * dx * dy)
            T[i, ny - 1] = (ae * T[i + 1, ny - 1] * 0.5 + aw * T[i - 1, ny - 1] * 0.5 + asn * T[i, ny - 2] + b / 2) / (aw * 0.5 + ae * 0.5 + asn + 4 * e * s * T_old[i, j]**3 * dx * dy)

            # Corner boundary conditions
            T[0, 0] = (ae * T[1, 0] * 0.5 + asn * T[0, 1] * 0.5 + b / 4) / (ae * 0.5 + asn * 0.5 + 2 * e * s * T_old[i, j]**3 * dx * dy)
            T[0, ny - 1] = (ae * T[1, ny - 1] * 0.5 + asn * T[0, ny - 2] * 0.5 + b / 4) / (ae * 0.5 + asn * 0.5 + 2 * e * s * T_old[i, j]**3 * dx * dy)
            T[nx - 1, 0] = (aw * T[nx - 2, 0] * 0.5 + an * T[nx - 1, 1] * 0.5 + b / 4) / (aw * 0.5 + an * 0.5 + 2 * e * s * T_old[i, j]**3 * dx * dy)
            T[nx - 1, ny - 1] = (aw * T[nx - 2, ny - 1] * 0.5 + asn * T[nx - 1, ny - 2] * 0.5 + b / 4) / (aw * 0.5 + asn * 0.5 + 2 * e * s * T_old[i, j]**3 * dx * dy)

            # Bulk equation
            ap = aw + ae + an + asn + 8 * e * s * T_old[i, j]**3 * dx * dy

            # Apply heat flux at the center
            if i == nx // 2 and j == ny // 2:
                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b + q1* dx * dy) / ap
            elif i == nx // 3 and j == ny // 3:
                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b + q2 * dx * dy) / ap
##
##            elif i == 2*nx // 3 and j == 2*ny // 3:
##                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b + q2 * dx * dy) / ap
##            elif i == nx // 3 and j == 2*ny // 3:
##                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b + q2 * dx * dy) / ap
####            elif i == 5 and j == 4:
####                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b + q1 * dx * dy) / ap
            else:
                T[i, j] = (ae * T[i + 1, j] + aw * T[i - 1, j] + an * T[i, j + 1] + asn * T[i, j - 1] + b) / ap

            # Apply relaxation factor
            T[i, j] = relaxation_factor * T[i, j] + (1 - relaxation_factor) * T_old[i, j]

    # Update the progress bar
    pbar.update(1)

    # Check for convergence
    if np.max(np.abs(T - T_old)) < convergence_criterion:
        converged = True
        print(f"Converged after {l} iterations.")
        break

# Close the progress bar
pbar.close()

# Check if the solution converged
if not converged:
    print("Error: Solution did not converge within the maximum number of iterations.")

# Print the final temperature field
print(T)

# Create a meshgrid for plotting
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Plot the temperature field
plt.figure(figsize=(10, 10))
contour = plt.contourf(X, Y, T, levels=100, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
plt.xlabel("Length in X-direction (m)")
plt.ylabel("Length in Y-direction (m)")
plt.title("2D Steady-State Temperature Distribution with Central Heat Flux")
plt.grid(False)
plt.show()
