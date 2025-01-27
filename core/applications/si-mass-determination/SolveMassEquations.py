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
    x0[x0 < 0] = 0.0 # Make sure initial guess is feasible

    lb, ub = np.zeros((n,)), np.zeros((n,))
    lb.fill(-np.inf)
    ub.fill(np.inf)

    for k in range(6):
        lb[4 * k] = 0.0

    bounds = [(lower, upper) for lower, upper in zip(lb, ub)]

    result = opt.minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds)
    if not result.success:
        print("!!! Failed to solve !!!")
        print("Details: ")
        print(result)
    else:
        print("Success!")
        print(result.x)

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
    b = Eq[:, -1]

    ok, x = solve(A, b)
    np.savetxt(args.output, x)
