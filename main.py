import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Визначення функції диференціального рівняння
def dydt(t, y, alpha, beta):
    return -alpha * beta * t * y ** 2


# Початкові умови
alpha = 1
beta = 1
y0 = beta
t_span = (alpha, alpha + 3)

# Розв'язок методом Метьюса (метод Рунге-Кутта)
sol_rk = solve_ivp(dydt, t_span, [y0], args=(alpha, beta), method='RK45', atol=1e-4, rtol=1e-4)


# Метод Мілна-Симпсона
def milne_simpson(dydt, t_span, y0, alpha, beta, h=0.1):
    N = int((t_span[1] - t_span[0]) / h)
    t_values = np.linspace(t_span[0], t_span[1], N + 1)
    y_values = np.zeros(N + 1)
    y_values[0] = y0

    # Метод Ейлера для знаходження початкових значень
    for i in range(1, 4):
        y_values[i] = y_values[i - 1] + h * dydt(t_values[i - 1], y_values[i - 1], alpha, beta)

    # Метод Мілна-Симпсона
    for i in range(3, N):

        y_pred = y_values[i - 3] + 4 * h / 3 * (
                    2 * dydt(t_values[i - 2], y_values[i - 2], alpha, beta) - dydt(t_values[i - 1], y_values[i - 1],
                                                                                   alpha, beta) + 2 * dydt(t_values[i],
                                                                                                           y_values[i],
                                                                                                           alpha, beta))

        y_values[i + 1] = y_values[i - 1] + h / 3 * (
                    dydt(t_values[i - 1], y_values[i - 1], alpha, beta) + 4 * dydt(t_values[i], y_values[i], alpha,
                                                                                   beta) + dydt(t_values[i + 1], y_pred,
                                                                                                alpha, beta))

    return t_values, y_values


# Використання методу Мілна-Симпсона
t_milne, y_milne = milne_simpson(dydt, t_span, y0, alpha, beta)

# Побудова графіків
plt.plot(sol_rk.t, sol_rk.y[0], label='Метод Метьюса (Рунге-Кутта)')
plt.plot(t_milne, y_milne, label='Метод Мілна-Симпсона')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

