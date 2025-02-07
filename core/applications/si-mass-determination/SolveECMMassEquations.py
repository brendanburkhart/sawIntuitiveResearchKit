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

def solve(A, b, lambda1 = 1, lambda2 = 0.1):
    full_n = A.shape[1]
    mass_indices = np.array([i for i in range(full_n) if i % 4 == 0])
    force_to_zero = np.array([3, 7, 11, 15, 18])
    n = full_n - len(force_to_zero)

    def idx_map(i):
        skip = 0
        for idx in force_to_zero:
            if i > idx:
                skip += 1
            elif i == idx:
                assert(False)

        return i - skip

    def inv_idx_map(i):
        for idx in force_to_zero:
            if i >= idx:
                i += 1

        return i

    def xmap(x):
        full_x = np.zeros((full_n,))
        for i in range(n):
            full_x[inv_idx_map(i)] = x[i]

        return full_x

    def yvec(x):
        x = xmap(x)
        y = np.zeros_like(x)
        for i in range(5):
            m = x[4 * i + 0]
            y[4 * i + 0] = 1 * x[4 * i + 0]
            y[4 * i + 1] = m * x[4 * i + 1]
            y[4 * i + 2] = m * x[4 * i + 2]
            y[4 * i + 3] = m * x[4 * i + 3]

        return y

    def Hmat(x):
        x = xmap(x)
        H = np.zeros((full_n, full_n))
        for i in range(n):
            j = i % 4
            if j == 0:
                H[i:i+4, i] = np.array([1.0, x[i+1], x[i+2], x[i+3]])
            else:
                m = x[i - j]
                H[i, i] = m

        return H

    def objective(x):
        y = yvec(x)
        lstsq = 0.5 * np.linalg.norm(A @ y - b, 2)**2
        l2 = 0.5 * np.linalg.norm(x, 2)**2
        l1 = np.linalg.norm(x, 1)

        return lstsq + lambda1 * l2 + lambda2 * l1

    def gradient(x):
        y, H = yvec(x), Hmat(x)
        g = H.T @ A.T @ (A @ y - b)
        g = np.delete(g, force_to_zero, 0)

        return g + lambda1 * x + lambda2 * np.sign(x)

    def hessian(x):
        y, H = yvec(x), Hmat(x)
        D = A @ H
        D = np.delete(D, force_to_zero, 1)

        return D.T @ D + lambda1 * np.eye(n)

    # Use simple L2-regularized solutions as initial guess
    # x0 = np.linalg.lstsq(A.T @ A + 10 * np.eye(n), A.T @ b, rcond=None)[0]

    # msk = np.zeros(x0.shape, dtype=bool)
    # msk[mass_indices] = True
    # x0[np.logical_and(x0 < 0, msk)] = 0.0 # Make sure initial guess is feasible

    lb, ub = np.zeros((n,)), np.zeros((n,))
    lb.fill(-np.inf)
    ub.fill(np.inf)

    mass_indices = np.array([idx_map(i) for i in mass_indices if i not in force_to_zero])

    lb[mass_indices] = 0.0

    # for k in force_to_zero:
    #     lb[k] = 0.0
    #     ub[k] = 0.0

    # x0 = np.array([1.0, -0.05,  0.05,  0.00,
    #                1.0,  0.10,  0.00,  0.00,
    #                1.0,  0.15, -0.07,  0.00,
    #                1.0,  0.01,  0.15,  0.00,
    #                1.0, -0.03,  0.00, -0.10])
    # x0 = np.delete(x0, force_to_zero, 0)
    x0 = np.zeros_like(lb)
    x0[mass_indices] = 1.0

    print(np.linalg.norm(A @ yvec(x0) - b, np.inf))

    lb[idx_map(4 * 0 + 1)] = -0.12
    ub[idx_map(4 * 0 + 1)] =  0.02
    lb[idx_map(4 * 0 + 2)] =  0.00
    ub[idx_map(4 * 0 + 2)] =  0.12

    lb[idx_map(4 * 1 + 1)] =  0.0
    ub[idx_map(4 * 1 + 1)] =  0.25
    lb[idx_map(4 * 1 + 2)] = -0.10
    ub[idx_map(4 * 1 + 2)] =  0.10

    lb[idx_map(4 * 2 + 1)] =  0.0
    ub[idx_map(4 * 2 + 1)] =  0.35
    lb[idx_map(4 * 2 + 2)] = -0.15
    ub[idx_map(4 * 2 + 2)] =  0.0

    lb[idx_map(4 * 3 + 1)] =  0.00
    ub[idx_map(4 * 3 + 1)] =  0.03
    lb[idx_map(4 * 3 + 2)] = -0.10
    ub[idx_map(4 * 3 + 2)] =  0.40

    lb[idx_map(4 * 4 + 1)] = -0.06
    ub[idx_map(4 * 4 + 1)] =  0.00
    lb[idx_map(4 * 4 + 3)] = -0.20
    ub[idx_map(4 * 4 + 3)] =  0.00

    for i, v in enumerate(x0):
        x0[i] = max(lb[i], min(ub[i], v))

    for i in range(5):
        for j in range(4):
            if i*4 + j in force_to_zero:
                print(0, ", ", end='')
            else:
                print(x0[idx_map(i*4 + j)], ", ", end='')
        print()

    bounds = [(lower, upper) for lower, upper in zip(lb, ub)]

    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, options={ "maxiter": 100000})
    if not result.success:
        print("!!! Failed to solve !!!")
        print("Details: ")
        print(result)
    else:
        print("Success!")

    predicted_b = A @ yvec(result.x)
    print("L-inf norm of residuals: ", np.linalg.norm(predicted_b - b, np.inf))

    solution = np.zeros((full_n,))
    for i, v in enumerate(result.x):
        solution[inv_idx_map(i)] = v

    link_masses = []
    for i in range(5):
        mass = solution[4 * i]
        com = solution[4*i + 1:4 * i + 4]
        link_mass = f'link {i + 1} | "mass": {mass:.3f}, "cx": {com[0]: 7.3f}, "cy": {com[1]: 7.3f}, "cz": {com[2]: 7.3f}'
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
