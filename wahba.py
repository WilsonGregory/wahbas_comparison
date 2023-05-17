import numpy as np
from functools import partial
import time
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import sys
sys.path.insert(1, '../bcg') #hacks all the way down
import os
import contextlib

import jax.numpy as jnp
import jax.random as random
from jax import value_and_grad, jit, vmap, grad

import optax
import scipy.optimize
import bcg
from bcg.run_BCG import BCG
from examples.example_model_class import ModelBirkhoffPolytope

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting and Printing functions

def plot_heatmap(arr, ax, xlabels, ylabels, title, vmin=None, vmax=None):
    ax.imshow(arr, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)

    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            text = ax.text(j, i, f'{arr[i, j]:.3f}', ha="center", va="center", color="w")

    ax.set_title(title)

def table_print(arr, model, N_range):
    for N, row in zip(N_range, arr):
        # print('N', row)
        if (N == N_range[0]):
            print(f'{model} ', end='')

        print(f'& {N} ', end='')

        for val in row:
            print(f' & {val:.4f}', end='')

        print(' \\\\') #each slash is escaped

def plot_dots(V, labels, color, ax, annot_mult):
    # V is an d x n matrix
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)

    ax.scatter(V[0], V[1], color=color)

    for i, label in enumerate(labels):
        ax.annotate(label, xy=annot_mult*V[:,i])

def plot_dots_with_arrows(V, Vtilde, show_vtilde_labels=True, show_arrows=True):
    D, N = V.shape
    fig, ax = plt.subplots()
    plot_dots(V, np.arange(N), 'blue', ax, annot_mult=1.1)

    if (show_vtilde_labels):
        vtilde_labels = np.arange(N)
    else:
        vtilde_labels = []

    plot_dots(Vtilde, vtilde_labels, 'red', ax, annot_mult=0.9)

    if (show_arrows):
        for col in range(N):
            direction = (Vtilde[:,col] - V[:,col]) / jnp.linalg.norm(Vtilde[:,col] - V[:,col])
            start = V[:,col] + 0.05 * direction
            end = (Vtilde[:,col] - V[:,col]) - 0.1*direction

            ax.arrow(
                start[0],
                start[1],
                end[0],
                end[1],
                length_includes_head=True,
                head_width=0.05,
                head_length=0.05,
            )

    fig.tight_layout()

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Naive and brute force methods

def mul_svd(V, Vtilde, W, P):
    # Multiply the relevant matrices, then peform the svd
    return jnp.linalg.svd(Vtilde @ P @ W @ W.T @ V.T, full_matrices=False)

@jit
def svd_over_permutations(V, Vtilde, W):
    # For given V, Vtilde, W, multiply them accordingly then do svd for all permutations.
    _, N = V.shape
    # generate the permutation matrices
    Ps = jnp.stack(list(jnp.eye(N)[jnp.array(sigma)] for sigma in it.permutations(range(N), N)))

    vmap_svd = vmap(mul_svd, in_axes=(None, None, None, 0))
    return vmap_svd(V, Vtilde, W, Ps) + (Ps,)

def procrustes_all_permutations(V, Vtilde, W):
    # Brute force solution to finding the the best permutation.
    u_s, s_s, vh_s, P_s = svd_over_permutations(V, Vtilde, W) # for each permutation

    opt_idx = jnp.argmax(jnp.linalg.norm(jnp.sqrt(s_s), axis=1)**2)

    return vh_s[opt_idx].T @ u_s[opt_idx].T, P_s[opt_idx]

@jit
def procrustes(V, Vtilde, W):
    D, N = V.shape
    u, s, vh = mul_svd(V, Vtilde, W, jnp.eye(N)) # same, but use identity as the permutation
    return vh.T @ u.T

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wasserstein Procrustes Relaxation https://arxiv.org/pdf/1805.11222.pdf

def wasserstein_procrustes(key, V, Vtilde, W, T, b, alpha=0.01, Qnot=None):
    D, N = V.shape

    assert (N % b) == 0, 'b must evenly divide N, the number of vectors'

    X = W @ W @ V.T # use the same variables used in the referenced paper.
    Y = Vtilde.T

    if (Qnot is None):
        Q_t = get_initial_Q(X, Y)
    else:
        Q_t = Qnot

    train_loss_best = jnp.inf
    patience = 30
    Q_t_best = Q_t

    for t in range(T):

        # pick a random subset of the vectors
        key, subkey = random.split(key)
        vector_permutation = random.permutation(subkey, N)

        for i in range(N // b):
            col_subset = vector_permutation[i*b:(i+1)*b]

            X_t = X[col_subset]
            Y_t = Y[col_subset]

            # find the best permutation for the given Q_t
            sigma = scipy.optimize.linear_sum_assignment(-1*Y_t @ Q_t.T @ X_t.T)[1]
            P_t = jnp.eye(b)[sigma].T #why is this the transpose?

            # update the Q_t given the permutation
            G_t = -2 * (X_t.T @ P_t @ Y_t) #the gradient step, it could be the transpose
            u, s, vh = jnp.linalg.svd(Q_t - alpha * G_t, full_matrices=False)
            Q_t = u @ vh #update this

        # Calculate the full training loss
        temp_sigma = scipy.optimize.linear_sum_assignment(-1*Y @ Q_t.T @ X.T)[1]
        temp_P = jnp.eye(N)[temp_sigma].T
        train_loss = jnp.linalg.norm(X @ Q_t - temp_P @ Y)**2


        if (train_loss < train_loss_best):
            train_loss_best = train_loss
            iters_since_improvement = 0
            Q_t_best = Q_t
        elif(iters_since_improvement > patience):
            break
        else:
            iters_since_improvement += 1

    final_sigma = scipy.optimize.linear_sum_assignment(-1*Y @ Q_t_best.T @ X.T)[1]
    P = jnp.eye(N)[final_sigma].T

    return Q_t_best, P.T

def fluba(P, X, Y):
    N, _ = X.shape
    P = P.reshape((N,N))
    return jnp.linalg.norm(X @ X.T @ P -  P @ Y @ Y.T)**2

def get_initial_Q(X, Y):
    N, _ = X.shape

    f_partial = partial(fluba, X=X, Y=Y)
    f_grad = grad(f_partial)

    birkhoff_polytope = ModelBirkhoffPolytope(N)

    run_config = {
        'solution_only': True,
        'verbosity': 'quiet', #hide table
        'dual_gap_acc': 1e-06,
        'runningTimeLimit': 180, #max time in seconds
        'use_LPSep_oracle': True,
        'max_lsFW': 30,
        'strict_dropSteps': True,
        'max_stepsSub': 200,
        'max_lsSub': 30,
        'LPsolver_timelimit': 100,
        'K': 1,
    }

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): #silence printing
        res = BCG(f_partial, f_grad=f_grad, model=birkhoff_polytope, run_config=run_config)

    opt_P = res[0].reshape((N,N))

    u, s, vh = jnp.linalg.svd(X.T @ opt_P @ Y, full_matrices=False)
    Q_not = u @ vh

    return Q_not

def get_optimal_permutation(V, Vtilde):
    D, N = V.shape

    min_error = jnp.inf
    Ps = jnp.stack(list(jnp.eye(N)[jnp.array(sigma)] for sigma in it.permutations(range(N), N)))
    for P in Ps:

        error = jnp.linalg.norm(V - Vtilde @ P)**2

        if (error < min_error):
            min_error = error
            P_opt = P

    return P_opt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data generation and core functions

def get_loss(V, Vtilde, W):
    return jnp.linalg.norm((V - Vtilde) @ W)**2

def getV(subkey, shape):
    raw_V = random.normal(subkey, shape=shape)
    V = raw_V / jnp.linalg.norm(raw_V, axis=0).reshape((1,-1))
    assert jnp.allclose(np.linalg.norm(V, axis=0), jnp.ones(shape[1]))
    return V

def getW(subkey, shape, weights='exp'):
    if (weights == 'exp'):
        lmbda = 0.5
        uniform_samples = random.uniform(subkey, shape=shape)
        weights = (-1 / lmbda)*jnp.log(1 - uniform_samples)
    elif (weights == 'id'):
        weights = jnp.ones(shape)
    else:
        raise Exception(f'Weights not recognized: {weights}, must be exp or id')

    # print('WARNING! Weights currently identity, not normalized')
    weights = weights / jnp.linalg.norm(weights, ord=1)
    assert jnp.isclose(jnp.sum(weights),1)
    return jnp.diag(weights)

def get_data(key, D, N, Vtilde_permuted=False, Vtilde_rotated=False, weights='exp'):
    """
    Generate random data. If either Vtilde_permuted or Vtilde_rotated is true, then Vtilde
    is not random, but V left multiplied by Omega.T and right multiplied by P.T.
    args:
        key (int): random key
        D (int): dimension of the vectors (rows)
        N (int): number of vectors (columns)
        Vtilde_permuted (bool): whether Vtilde is a permuted version of V
        Vtilde_rotated (bool): whether Vtilde is a rotated version of V
    returns:
        V: (D,N) matrix
        Vtilde: (D,N) matrix
        W: (N,N) matrix
        P: (N,N) permutation matrix if Vtilde_permuted is True, Identity otherwise
        Omega: (D,D) random orthogonal matrix if Vtilde_rotated is True, Identity otherwise
    """
    key, subkey = random.split(key)
    V = getV(subkey, (D, N)) # get V where the columns have 2-norm equal to 1
    assert jnp.allclose(np.linalg.norm(V, axis=0), jnp.ones(N))

    P = jnp.eye(N)
    Omega = jnp.eye(D)

    if (Vtilde_permuted):
        key, subkey = random.split(key)
        P = P[random.permutation(subkey, N)]

    if (Vtilde_rotated):
        Omega = ortho_group.rvs(D)

    if (Vtilde_permuted or Vtilde_rotated):
        Vtilde = Omega.T @ V @ P.T
    else:
        key, subkey = random.split(key)
        Vtilde = getV(subkey, (D, N))

    key, subkey = random.split(key)
    W = getW(subkey, (N,), weights)

    return V, Vtilde, W, P, Omega

def compare_wahba_permutations_small(key, epochs, D_range, num_vectors_range, weights='exp'):
    original_loss = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    loss_with_P = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    loss_without_P = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    loss_wasserstein_procrustes = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    for epoch in range(epochs):
        if ((epoch % (epochs // 10)) == 0):
            print(epoch)

        for j, D in enumerate(D_range):
            for k, num_vectors in enumerate(num_vectors_range):
                key, subkey = random.split(key)
                V, Vtilde, W, _, _ = get_data(subkey, D, num_vectors, weights=weights)

                original_loss[epoch,j,k] = get_loss(V, Vtilde, W)

                Omega, P_opt = procrustes_all_permutations(V, Vtilde, W)
                loss_with_P[epoch,j,k] = get_loss(V, Omega @ Vtilde @ P_opt, W)

                Omega2 = procrustes(V, Vtilde, W)
                loss_without_P[epoch,j,k] = get_loss(V, Omega2 @ Vtilde, W)

                key, subkey = random.split(key)
                Omega3, P_opt3 = wasserstein_procrustes(
                    subkey,
                    V,
                    Vtilde,
                    W,
                    T=1000,
                    b=num_vectors, #for this small count, use all the vectors always
                    alpha=0.1,
                )
                loss_wasserstein_procrustes[epoch,j,k] = get_loss(V, Omega3 @ Vtilde @ P_opt3, W)



    return loss_with_P, loss_without_P, original_loss, loss_wasserstein_procrustes

def compare_wahba_permutations_large(key, epochs, D_range, num_vectors_range):
    original_loss = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    loss_without_P = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    loss_wasserstein_procrustes = np.zeros((epochs, len(D_range), len(num_vectors_range)))
    for epoch in range(epochs):
        print(epoch)

        for j, D in enumerate(D_range):
            for k, num_vectors in enumerate(num_vectors_range):
                key, subkey = random.split(key)
                V, Vtilde, W, _, _ = get_data(subkey, D, num_vectors, weights='id')

                original_loss[epoch,j,k] = get_loss(V, Vtilde, W)

                Omega2 = procrustes(V, Vtilde, W)
                loss_without_P[epoch,j,k] = get_loss(V, Omega2 @ Vtilde, W)

                key, subkey = random.split(key)
                Omega3, P_opt3 = wasserstein_procrustes(
                    subkey,
                    V,
                    Vtilde,
                    W,
                    T=1000,
                    b=num_vectors // 5, #for this small count, use all the vectors always
                    alpha=0.1,
                )
                loss_wasserstein_procrustes[epoch,j,k] = get_loss(V, Omega3 @ Vtilde @ P_opt3, W)

        jnp.save(f'loss_without_P_l.npy', loss_without_P)
        jnp.save(f'original_loss_l.npy', original_loss)
        jnp.save(f'loss_wass_p_l.npy', loss_wasserstein_procrustes)

    return loss_without_P, original_loss, loss_wasserstein_procrustes

def compare_small_sample():
    load = True
    epochs = 20
    D_range = [2,3,5,10,50]
    num_vectors_range = [3,5,7]
    weights = 'id'

    # seed = time.time_ns()
    seed = 1729
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    if load:
        loss_with_P = np.load(f'loss_with_P_{weights}_{seed}.npy')
        loss_without_P = np.load(f'loss_without_P_{weights}_{seed}.npy')
        original_loss = np.load(f'original_loss_{weights}_{seed}.npy')
        loss_wass_p = np.load(f'loss_wass_p_{weights}_{seed}.npy')
    else:
        loss_with_P, loss_without_P, original_loss, loss_wass_p = compare_wahba_permutations_small(
            subkey,
            epochs,
            D_range,
            num_vectors_range,
            weights,
        )

        jnp.save(f'loss_with_P_{weights}_{seed}.npy', loss_with_P)
        jnp.save(f'loss_without_P_{weights}_{seed}.npy', loss_without_P)
        jnp.save(f'original_loss_{weights}_{seed}.npy', original_loss)
        jnp.save(f'loss_wass_p_{weights}_{seed}.npy', loss_wass_p)

    table_print(jnp.mean(loss_without_P, axis=0).T, 'procrustes', num_vectors_range)

    print('~~~~~~~~')
    table_print(jnp.mean(loss_with_P, axis=0).T, 'procrustes w/ permutations', num_vectors_range)

    print('~~~~~~~~')
    table_print(jnp.mean(loss_wass_p, axis=0).T, 'wasserstein procrustes', num_vectors_range)

    loss_with_P_improvement = jnp.mean(loss_without_P, axis=0).T / jnp.mean(loss_with_P, axis=0).T
    loss_wass_p_improvement = jnp.mean(loss_without_P, axis=0).T / jnp.mean(loss_wass_p, axis=0).T

    vmin = np.min([np.min(loss_with_P_improvement), np.min(loss_wass_p_improvement)])
    vmax = np.max([np.max(loss_with_P_improvement), np.max(loss_wass_p_improvement)])

    fig, axs = plt.subplots(ncols=2, figsize=(12,4))
    plot_heatmap(
        loss_with_P_improvement,
        axs[0],
        D_range,
        num_vectors_range,
        'Improvement Using Permutations',
        vmin=vmin,
        vmax=vmax,
    )

    # fig, ax = plt.subplots()
    plot_heatmap(
        loss_wass_p_improvement,
        axs[1],
        D_range,
        num_vectors_range,
        'Improvement Using Wasserstein Relaxation',
        vmin=vmin,
        vmax=vmax,
    )
    plt.tight_layout()
    plt.show()

def compare_large_sample():
    load = True
    epochs = 20
    D_range = [2,3,5,10,50]
    num_vectors_range = [50,100]
    weights = 'id'

    # seed = time.time_ns()
    seed = 1729
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    if load:
        loss_without_P = np.load(f'loss_without_P_{weights}_{seed}_l.npy')
        original_loss = np.load(f'original_loss_{weights}_{seed}_l.npy')
        loss_wass_p = np.load(f'loss_wass_p_{weights}_{seed}_l.npy')
    else:
        loss_without_P, original_loss, loss_wass_p = compare_wahba_permutations_large(
            subkey,
            epochs,
            D_range,
            num_vectors_range,
        )

        jnp.save(f'loss_without_P_{weights}_{seed}_l.npy', loss_without_P)
        jnp.save(f'original_loss_{weights}_{seed}_l.npy', original_loss)
        jnp.save(f'loss_wass_p_{weights}_{seed}_l.npy', loss_wass_p)

    print('~~~~~~~~')
    table_print(jnp.mean(loss_without_P, axis=0).T, 'procrustes', num_vectors_range)

    print('~~~~~~~~')
    table_print(jnp.mean(loss_wass_p, axis=0).T, 'wasserstein procrustes', num_vectors_range)

    loss_wass_p_improvement = jnp.mean(loss_without_P, axis=0).T / jnp.mean(loss_wass_p, axis=0).T

    fig, ax = plt.subplots(figsize=(6,4))
    plot_heatmap(
        loss_wass_p_improvement,
        ax,
        D_range,
        num_vectors_range,
        'Improvement Using Wasserstein Relaxation',
    )
    plt.tight_layout()
    plt.show()

def arrow_plots():
    # seed = time.time_ns()
    seed = 123
    key = random.PRNGKey(seed)

    D = 2
    N = 4

    key, subkey = random.split(key)
    V, Vtilde, W, _, _ = get_data(subkey, D, N, weights='id')

    # usual procrustes
    Omega = procrustes(V, Vtilde, W)
    updated_Vtilde = Omega @ Vtilde

    plot_dots_with_arrows(V, Vtilde)
    plot_dots_with_arrows(V, updated_Vtilde)

    # procrustes with all permutations
    Omega, P_opt = procrustes_all_permutations(V, Vtilde, W)
    updated_Vtilde = Omega @ Vtilde @ P_opt

    plot_dots_with_arrows(V, Vtilde, show_vtilde_labels=False)
    plot_dots_with_arrows(V, updated_Vtilde, show_vtilde_labels=False)

    # wasserstein distance only
    P_opt = get_optimal_permutation(V, Vtilde)
    updated_Vtilde = Vtilde @ P_opt

    # plot_dots_with_arrows(V, Vtilde, show_vtilde_labels=True)
    plot_dots_with_arrows(V, updated_Vtilde, show_vtilde_labels=False)

    plt.show()

# MAIN
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'

arrow_plots()
# compare_small_sample()
# compare_large_sample()

