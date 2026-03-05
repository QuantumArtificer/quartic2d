import numpy as np
import cProfile
import pstats
import io
import time
import pandas as pd
import matplotlib.pyplot as plt

import interactions2D as inter
import orbitalspectrum as orb
from interactions2D import HankelofHarmonics, Interact

# -----------------------------
# USER PARAMETERS
# -----------------------------
Nr = 256
Ntheta = 512
N_hankel = 2048 * 5
N_q = 1024
N_interact = 2048
rmax = 10.0

deltas = np.array([
    [0,0],
    [1,0],
    [2,0],
    [3,0],
    [4,0]
])

def U_q(q):
    return 2*np.pi/q


# -----------------------------
# Example Gaussian
# -----------------------------
def gaussian_example():
    x = np.linspace(-10, 10, Nr)
    y = np.linspace(-10, 10, Nr)
    X, Y = np.meshgrid(x, y, indexing='ij')
    rho = np.exp(-(X**2 + Y**2))
    return x, y, rho


# -----------------------------
# Profiling Utility
# -----------------------------
def profile_block(name, func, *args, **kwargs):
    print(f"\n--- Profiling {name} ---")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats('cumulative')
    ps.print_stats(25)
    print(s.getvalue())

    return result, elapsed, ps


# -----------------------------
# 1. Polar Decomposition
# -----------------------------
def run_decomposition():
    x, y, rho = gaussian_example()
    f = orb.PolarDecomposition(
        f_xy=rho,
        x=x,
        y=y,
        Nr=Nr,
        Ntheta=Ntheta,
        rmax=rmax,
        recon_err_tol=1.0,
        recon_power_tol=1e-12,
        m_abs_max=None,
        interp_method='cubic'
    )
    return f


# -----------------------------
# 2. HankelofHarmonics
# -----------------------------
def run_hankel(f):
    F = HankelofHarmonics(
        cdh=f,
        q_max=20*np.pi/f.r_cutoff[0],
        N=N_hankel,
        N_q=N_q,
        tol_power=1e-12,
        interp_method = 'linear'
    )
    return F


# -----------------------------
# 3. Interaction
# -----------------------------
def run_interact(F):
    V = Interact(deltas, F, F, U_q, N=N_interact)
    return V


# -----------------------------
# MAIN BENCHMARK
# -----------------------------
if __name__ == "__main__":

    times = {}

    # Decomposition
    f, t_decomp, stats_decomp = profile_block(
        "PolarDecomposition",
        run_decomposition
    )
    times["PolarDecomposition"] = t_decomp

    # Hankel
    F, t_hankel, stats_hankel = profile_block(
        "HankelofHarmonics",
        run_hankel,
        f
    )
    times["HankelofHarmonics"] = t_hankel

    # Interaction
    V, t_interact, stats_interact = profile_block(
        "Interact",
        run_interact,
        F
    )
    times["Interact"] = t_interact

    # -----------------------------
    # PRINT TABLE
    # -----------------------------
    df = pd.DataFrame.from_dict(times, orient='index', columns=["Time (s)"])
    df["Percent"] = 100 * df["Time (s)"] / df["Time (s)"].sum()

    print("\n============================")
    print("Benchmark Summary")
    print("============================")
    print(df)

    # -----------------------------
    # BAR CHART OF STAGE TIMES
    # -----------------------------
    plt.figure()
    plt.bar(df.index, df["Time (s)"])
    plt.ylabel("Time (seconds)")
    plt.title("Stage-wise Runtime")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Detailed breakdown inside HankelofHarmonics
    # -----------------------------
    print("\n--- Top cumulative time inside HankelofHarmonics ---")
    stats_hankel.sort_stats('cumulative').print_stats(20)

    # Extract function times for plotting
    func_data = []
    for func, stat in stats_hankel.stats.items():
        cumulative = stat[3]
        name = f"{func[2]}"
        func_data.append((name, cumulative))

    func_df = pd.DataFrame(func_data, columns=["Function", "Cumulative"])
    func_df = func_df.sort_values("Cumulative", ascending=False).head(10)

    plt.figure()
    plt.barh(func_df["Function"], func_df["Cumulative"])
    plt.gca().invert_yaxis()
    plt.xlabel("Cumulative Time (s)")
    plt.title("Top 10 Functions in HankelofHarmonics")
    plt.tight_layout()
    plt.show()
