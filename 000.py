# %% import few packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def q_logarithm(x, q = 1):
    """
    Compute the q-logarithm of x.

    Parameters:
        x (float or array-like): Input value(s), must be positive.
        q (float): The entropic index, determines the nature of the q-logarithm.

    Returns:
        float or ndarray: The q-logarithm of x.
    """
    if np.any(x <= 0):
        raise ValueError("x must be positive.")
    
    if q == 1:
        return np.log(x)
    else:
        return (x**(1-q) - 1) / (1-q)

def box_muller(n, q=1):
    assert q<5/3
    J = -q_logarithm(np.random.rand(n), q)
    J = J/np.std(J)
    phase = np.random.rand(n)
    rho = np.sqrt(2*J)
    x = rho * np.cos(2*np.pi*phase)
    px = rho * np.sin(2*np.pi*phase)
    beta = 1/np.std(x)**2/(5-3*q)
    return x, px, J, rho, phase, beta


# Define the q-exponential function
def q_exponential(x, q):
    if q == 1:
        return np.exp(x)  # Standard exponential when q=1
    return (1 + (1 - q) * x)**(1 / (1 - q)) if (1 + (1 - q) * x) > 0 else 0

# Define the normalization factor C_q
def C_q(q):
    if q < 1:
        return (2 * np.sqrt(np.pi) * gamma(1 / (1 - q))) / (
            (3 - q) * np.sqrt(1 - q) * gamma((3 - q) / (2 * (1 - q)))
        )
    elif q == 1:
        return np.sqrt(np.pi)
    elif 1 < q < 3:
        return (np.sqrt(np.pi) * gamma((3 - q) / (2 * (q - 1)))) / (
            np.sqrt(q - 1) * gamma(1 / (q - 1))
        )
    else:
        raise ValueError("q must be in the range (-∞, 3) excluding q ≥ 3.")

# Define the main function f(x)
def q_gaussian(x, beta, q):
    Cq = C_q(q)
    return np.sqrt(beta) / Cq * q_exponential(-beta * x**2, q)
# %% generate random numbers

n = 1000000
my_q = 1
x, px, J, rho, phase, beta = box_muller(n, q=my_q)
plt.hist(x, bins=100, density=True)
plt.hist(px, bins=100, density=True)
my_x = np.linspace(-6, 6, 100)
# map the q-gaussian on my_x
print(np.std(x))    

plt.plot(my_x, np.vectorize(q_gaussian)(my_x, q=my_q, beta= beta), 'r')
plt.xlim( [-6, 6])
plt.show()
# %%
plt.plot(x[0:100000], px[0:100000], '.',alpha=0.01)
plt.xlabel('x')
plt.ylabel('px')
plt.axis('equal')
# %%
plt.hist(J, bins=1000, density=True)
plt.hist(rho, bins=1000, density=True)

# %%
# select rho < 3.5
idx = np.where(rho < 3.5)
print(len(idx[0])/n)
plt.hist(rho[idx], bins=1000, density=False)
plt.hist(J[idx], bins=1000, density=False)
plt.hist(x[idx], bins=1000, density=False)
# %%
n = 10000000
my_q = 1.
x, px, J, rho, phase, beta = box_muller(n, q=my_q)

sigma_cut = 1
idx = np.where(rho > sigma_cut)
print(len(idx[0])/n)
idx = np.where( (x > sigma_cut) | (x < -sigma_cut))
print(len(idx[0])/n)
# %%
np.std(px)
# %%
J = -q_logarithm(np.random.rand(n), 1)
J = J/np.std(J)
np.std(J)
# %%
