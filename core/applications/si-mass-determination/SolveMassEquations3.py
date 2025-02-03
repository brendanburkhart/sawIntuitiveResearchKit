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

force_to_zero = [0, 3, 6, 7, 11, 14, 15, 17]

# post-processing to keep centers of mass realistic
#     by re-arranging masses without changing model outputs
def post_process(solution):
    print(solution)
    link_translations = [
        np.array([-0.0712, 0, -0.2913]),
        np.array([0.203130, 0.0, 0.0]),
        np.array([0.346122, 0.0, 0.0])
    ]
    # rough estimates of distance from link frame to center of mass
    estimated_lengths = [ 0.15, 0.15, 0.20, 0.03, 0.02 ]

    # Re-arrange mass to achieve more realistic center of masses
    for k in reversed(range(4)):
        mass = solution[4*k]
        com = solution[4*k+1 : 4*k+4]
        length = estimated_lengths[k]
        print(k, length, np.linalg.norm(com, 2), mass)
        adjusted_mass = np.linalg.norm(com, 2) / estimated_lengths[k]
        solution[4*k] = adjusted_mass
        delta = mass - adjusted_mass
        print(mass, adjusted_mass, delta)
        # if we removed mass from the proximal end of link k, we need to add
        # the same mass at distal end of link k-1
        if k > 0:
            solution[4*(k-1)] += delta
            # adjust center of mass of link k-1 correspondingly
            solution[4*(k-1)+1:4*k] += delta * link_translations[k-1]

    # ensure rearrangement hasn't made anything non-zero that should stay zero
    for idx in force_to_zero:
        solution[idx] = 0.0 if (idx % 4 != 0) else solution[idx]

    return solution

def mass_data(solution):
    link_masses = []
    n = solution.size // 4

    for i in range(n):
        mass = solution[4 * i]
        com = solution[4*i + 1:4 * i + 4]
        if mass < 1e-5:
            mass = 1.0
        com = com / mass

        link_mass = f'"mass": {mass:.3f}, "cx": {com[0]: 7.3f}, "cy": {com[1]: 7.3f}, "cz": {com[2]: 7.3f}'
        link_masses.append(link_mass)
        print(link_mass)

    return link_masses

def solve(A, b, lambda1 = 0.01, lambda2 = 0.1):
    n = A.shape[1]
    mass_indices = np.array([i for i in range(n) if i % 4 == 0])

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

    mass_mask = np.zeros(x0.shape, dtype=bool)
    mass_mask[mass_indices] = True
    x0[np.logical_and(x0 < 0, mass_mask)] = 0.0 # Make sure initial guess is feasible

    lb, ub = np.zeros((n,)), np.zeros((n,))
    lb.fill(-np.inf)
    ub.fill(np.inf)

    lb[mass_mask] = 0.0

    for k in force_to_zero:
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

    return result.success, result.x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help = "input file containing mass determination equation")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = "output file name")
    args = parser.parse_args()

    Eq = np.loadtxt(args.input)
    A = Eq[:, 0:-1]
    A = np.delete(A, [16, 17, 18, 19], 1)
    b = Eq[:, -1]

    ok, solution = solve(A, b)
    if not ok:
        sys.exit(1)

    Ax = A @ solution
    solution = post_process(solution)
    print("Post-processing induced error: ", np.linalg.norm(Ax - A @ solution, np.inf))

    link_masses = mass_data(solution)

    with open(args.output, 'w') as f:
        for mass in link_masses:
            f.write(f"{mass}\n")

    print(f"\nLink masses saved to {args.output}")
