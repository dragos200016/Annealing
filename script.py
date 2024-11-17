import numpy as np
import matplotlib.pyplot as plt

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def simulated_annealing(func, initial_state, max_iter=1000, initial_temp=100.0, cooling_rate=0.99):
    current_state = np.array(initial_state)
    current_value = func(current_state)
    best_state = np.copy(current_state)
    best_value = current_value
    temperature = initial_temp

    for iteration in range(max_iter):
        
        candidate_state = current_state + np.random.uniform(-1, 1, size=current_state.shape)
        candidate_value = func(candidate_state)

        delta_value = candidate_value - current_value

        
        if delta_value < 0 or np.exp(-delta_value / temperature) > np.random.rand():
            current_state = candidate_state
            current_value = candidate_value

            
            if current_value < best_value:
                best_state = np.copy(current_state)
                best_value = current_value

        
        temperature *= cooling_rate

        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best Value = {best_value:.4f}, Best State = {best_state}")

    return best_state, best_value


initial_state = [0.0, 0.0]


best_state, best_value = simulated_annealing(himmelblau, initial_state)

print("\nBest State:", best_state)
print("Best Value:", best_value)


x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
plt.plot(best_state[0], best_state[1], 'r*', markersize=15)
plt.title("Simulated Annealing on Himmelblau's Function")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
