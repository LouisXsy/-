import numpy as np
import math
import matplotlib.pyplot as plt

w = np.array([0.3, 0.7])
mu = np.array([4.0, 7.0])
var = np.array([0.3, 2.0])
sig = np.sqrt(var)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

def norm_logpdf(x, m, v):
    return -LOG_SQRT_2PI - 0.5 * math.log(v) - 0.5 * (x - m) ** 2 / v

def logsumexp2(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))

def log_pi(x):
    a0 = math.log(w[0]) + norm_logpdf(x, mu[0], var[0])
    a1 = math.log(w[1]) + norm_logpdf(x, mu[1], var[1])
    return logsumexp2(a0, a1)

def pi_pdf(x):
    return math.exp(log_pi(x))

def U(x):
    return -log_pi(x)

def U_grad(x):
    logc0 = math.log(w[0]) + norm_logpdf(x, mu[0], var[0])
    logc1 = math.log(w[1]) + norm_logpdf(x, mu[1], var[1])
    m = max(logc0, logc1)
    c0 = math.exp(logc0 - m)
    c1 = math.exp(logc1 - m)
    s = c0 + c1
    r0, r1 = c0 / s, c1 / s
    return r0 * (x - mu[0]) / var[0] + r1 * (x - mu[1]) / var[1]

def norm_cdf(x, m, s):
    z = (x - m) / (s * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def pi_cdf(x):
    return float(w[0] * norm_cdf(x, mu[0], sig[0]) + w[1] * norm_cdf(x, mu[1], sig[1]))

# Leapfrog 
def leapfrog(x, p, eps, L):
    # half momentum step
    p = p - 0.5 * eps * U_grad(x)
    for i in range(L):
        # full position step
        x = x + eps * p
        if i != L - 1:
            p = p - eps * U_grad(x)
    # last half momentum step
    p = p - 0.5 * eps * U_grad(x)
    return x, p

def hmc(n_samples=30000, burn=3000, x0=6.0, eps=0.22, L=30, seed=7):
    rng = np.random.default_rng(seed)
    x = float(x0)
    samples = []
    acc = 0

    for t in range(n_samples + burn):
        p0 = rng.normal(0.0, 1.0)
        # current Hamiltonian H = U + 0.5 p^2
        H0 = U(x) + 0.5 * p0 * p0
        x_prop, p_prop = leapfrog(x, p0, eps, L)
        H1 = U(x_prop) + 0.5 * p_prop * p_prop
        # MH 
        log_alpha = -(H1 - H0)
        if math.log(rng.random()) < log_alpha:
            x = x_prop
            accepted = True
        else:
            accepted = False

        if t >= burn:
            samples.append(x)
            acc += int(accepted)

    return np.array(samples), acc / n_samples

#  Error
def true_moments():
    mu_star = float(np.sum(w * mu))
    Ex2 = float(np.sum(w * (var + mu ** 2)))
    var_star = Ex2 - mu_star ** 2
    return mu_star, var_star

def ks_stat(samples):
    x = np.sort(samples)
    n = len(x)
    Fn = np.arange(1, n + 1) / n
    F = np.array([pi_cdf(xi) for xi in x])
    return float(np.max(np.abs(Fn - F)))

if __name__ == "__main__":
    samples, acc_rate = hmc(n_samples=30000, burn=3000, x0=6.0, eps=0.22, L=30, seed=7)

    mu_star, var_star = true_moments()
    mu_hat = float(np.mean(samples))
    var_hat = float(np.var(samples, ddof=1))
    ks = ks_stat(samples)

    print(f"Acceptance rate: {acc_rate:.3f}")
    print(f"True mean={mu_star:.4f}, Sample mean={mu_hat:.4f}, abs err={abs(mu_hat-mu_star):.3e}")
    print(f"True var ={var_star:.4f}, Sample var ={var_hat:.4f}, abs err={abs(var_hat-var_star):.3e}")
    print(f"KS statistic: {ks:.6f}")

    # Plot histogram vs true pdf
    grid = np.linspace(0, 12, 2000)
    true_pdf = np.array([pi_pdf(t) for t in grid])

    plt.figure(figsize=(9, 4.8))
    plt.hist(samples, bins=140, density=True, alpha=0.45, label="HMC samples")
    plt.plot(grid, true_pdf, linewidth=2, label="Target pdf")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("density")
    plt.tight_layout()
    plt.show()

    # Trace 
    plt.figure(figsize=(9, 3.2))
    plt.plot(samples[:2000], linewidth=0.8)
    plt.title("Trace (first 2000 post-burn samples)")
    plt.xlabel("iteration")
    plt.ylabel("x")
    plt.tight_layout()
    plt.show()
