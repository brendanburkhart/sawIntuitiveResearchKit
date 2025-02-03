#!/usr/bin/env python3

# Author(s):  Brendan Burkhart
# Created on: 2025-01-24
#
# (C) Copyright 2025 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

import argparse
import numpy as np
import scipy.optimize as opt

def solve(A, b, lambda1 = 0.01, lambda2 = 0.1):
    removed_links = 1
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23])
    mass_indices = np.array([i for i in range(len(indices)) if i % 4 == 0])
    estimated_lengths = [ 0.15, 0.15, 0.16, 0.03, 0.02 ]

    A = A[:, indices]
    b = b[:]

    n = A.shape[1]

    def objective(x):
        lstsq = 0.5 * np.linalg.norm(A @ x - b, 2)**2
        l2 = 0.5 * np.linalg.norm(x, 2)**2
        l1 = np.linalg.norm(x, 1)

        return lstsq + lambda1 * l2 + lambda2 * l1

    def gradient(x):
        return A.T @ (A @ x - b)  + lambda1 * x + lambda2 * np.sign(x)

    def hessian(x):
        return A.T @ A + lambda1 * np.eye(n)

    # Use simple L2-regularized solutions as initial guess
    x0 = np.linalg.lstsq(A.T @ A + 0.01 * np.eye(n), A.T @ b, rcond=None)[0]

    msk = np.zeros(x0.shape, dtype=bool)
    msk[mass_indices] = True
    x0[np.logical_and(x0 < 0, msk)] = 0.0 # Make sure initial guess is feasible

    lb, ub = np.zeros((n,)), np.zeros((n,))
    lb.fill(-np.inf)
    ub.fill(np.inf)

    for k in range(6-removed_links):
        lb[4 * k] = 0.0

    for k in [0, 3, 6, 7, 11, 14, 15, 17]:
        lb[k] = 0.0
        ub[k] = 0.0

    bounds = [(lower, upper) for lower, upper in zip(lb, ub)]

    result = opt.minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds)
    if not result.success:
        print("!!! Failed to solve !!!")
        print("Details: ")
        print(result)
    else:
        print("Success!")

    predicted_b = A @ result.x
    # print("Residual l-inf norm: ", predicted_b - b)

    solution = result.x

    print(solution)

    pstar = [np.array([-0.0712, 0, -0.2913]), np.array([0.203130, 0.0, 0.0]), np.array([0.346122, 0.0, 0.0])]

    # Re-arrange mass to achieve more realistic center of masses
    for i in range(6-removed_links-1):
        k = (6 - removed_links) - 2 - i
        mass = solution[4*k]
        com = solution[4*k+1 : 4*k+4]
        length = estimated_lengths[k]
        adjusted_mass = np.linalg.norm(com, 2) / length
        solution[4*k] = adjusted_mass
        delta = mass - adjusted_mass
        if k > 0:
            solution[4*(k-1)] += delta
            solution[4*(k-1)+1:4*k] += delta * pstar[k-1]

    solution[0] = 1.0
    solution[3] = 0.0
    
    solution[6] = 0.0
    solution[7] = 0.0

    solution[11] = 0.0
    solution[14] = 0.0
    solution[15] = 0.0

    solution[17] = 0.0

    print("Re-arrangement error: ", np.linalg.norm(predicted_b - A @ solution, np.inf))

    link_masses = []

    for i in range(6-removed_links):
        mass = solution[4 * i]
        if mass < 1e-5:
            mass = 1.0
        com = solution[4*i + 1:4 * i + 4]
        com = com / mass
        link_index = (indices[4 * i] // 4) + 1
        link_mass = f'link {link_index} | "mass": {mass:.3f}, "cx": {com[0]: 7.3f}, "cy": {com[1]: 7.3f}, "cz": {com[2]: 7.3f}'
        link_masses.append(link_mass)
        print(link_mass)

    return result.success, link_masses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help = "input file containing mass determination equation")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = "output file name")
    args = parser.parse_args()

    Eq = np.loadtxt(args.input)
    A = Eq[:, 0:-1]
    b = Eq[:, -1]

    ok, link_masses = solve(A, b)
    with open(args.output, 'w') as f:
        for mass in link_masses:
            f.write(f"{mass}\n")

    print(f"\nLink masses saved to {args.output}")
