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
import enum
import numpy as np
import scipy.optimize as opt
import sys

class ArmType(enum.Enum):
    PSM = 0
    ECM = 1

def solve(A, b, arm_type, lambda1 = 0.01, lambda2 = 0.1):
    n = A.shape[1]
    links = n // 4 # four variables per link (mass, cx, cy, cz)
    mass_indices = np.array([i for i in range(n) if i % 4 == 0])

    def bounds(n):
        lower = np.full((n,), -np.inf)
        upper = np.full((n,), +np.inf)

        lower[mass_indices] = 0.0 # link masses must be non-negative

        def set_com_bounds(k, x, y, z):
            lower[4 * k + 1] = x[0]
            upper[4 * k + 1] = x[1]

            lower[4 * k + 2] = y[0]
            upper[4 * k + 2] = y[1]

            lower[4 * k + 3] = z[0]
            upper[4 * k + 3] = z[1]

        set_com_bounds(0, (-0.12,  0.02), ( 0.00,  0.12), ( 0.00,  0.00))
        set_com_bounds(1, ( 0.00,  0.20), (-0.10,  0.10), ( 0.00,  0.00))
        set_com_bounds(2, ( 0.00,  0.35), (-0.15,  0.00), ( 0.00,  0.00))
        set_com_bounds(3, ( 0.00,  0.03), (-0.10,  0.40), ( 0.00,  0.00))

        if arm_type == ArmType.PSM:
            # completely ignore first insertion stage
            lower[4 * 4 + 0], uppper[4 * 4 + 0] = 0.0, 0.0
            set_com_bounds(4, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

            set_com_bounds(5, ( 0.00,  0.02), ( 0.00,  0.00), (-0.10,  0.10))
        elif arm_type == ArmType.ECM:
            set_com_bounds(4, (-0.06,  0.00), ( 0.00,  0.00), (-0.20,  0.00))

        return lower, upper

    def yvec(x):
        y = np.zeros_like(x)
        for i in range(links):
            m = x[4 * i + 0]
            y[4 * i + 0] = 1 * x[4 * i + 0]
            y[4 * i + 1] = m * x[4 * i + 1]
            y[4 * i + 2] = m * x[4 * i + 2]
            y[4 * i + 3] = m * x[4 * i + 3]

        return y

    def Hmat(x):
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
        l1_norm = np.linalg.norm(x, 1)
        l2_norm = 0.5 * np.linalg.norm(x, 2)**2

        return lstsq + lambda1 * l1_norm + lambda2 * l2_norm

    def gradient(x):
        y, H = yvec(x), Hmat(x)
        g = H.T @ A.T @ (A @ y - b)

        return g + lambda1 * np.sign(x) + lambda2 * x

    def hessian(x):
        H = Hmat(x)
        D = A @ H

        return D.T @ D + lambda2 * np.eye(n)

    lower, upper = bounds(n)

    x0 = np.zeros((n,))
    x0[mass_indices] = 1.0

    # clamp initial guess within bounds so it is feasible
    for i, v in enumerate(x0):
        x0[i] = max(lower[i], min(upper[i], v))

    print("L-inf norm of initial guess residuals: ", np.linalg.norm(A @ yvec(x0) - b, np.inf))

    result = opt.minimize(objective, x0,
                          method='L-BFGS-B',
                          bounds=list(zip(lower, upper)),
                          options={ "maxiter": 1e5})
    if not result.success:
        print("!!! Failed to solve !!!")
        print("Details: ")
        print(result)
    else:
        print("Success!")

    solution = result.x
    predicted_b = A @ yvec(result.x)
    print("L-inf norm of residuals: ", np.linalg.norm(predicted_b - b, np.inf))
    print("Normalized L2 norm of residuals: ", np.linalg.norm(predicted_b - b, 1) / b.shape[0])

    # serialize solution
    link_masses = []
    for i in range(links):
        if arm_type == ArmType.PSM and i == 4:
            continue

        mass = solution[4 * i]
        com = solution[4*i + 1:4 * i + 4]
        link_mass = f'link {i + 1} | "mass": {mass:.3f}, "cx": {com[0]: 7.3f}, "cy": {com[1]: 7.3f}, "cz": {com[2]: 7.3f}'
        link_masses.append(link_mass)
        print(link_mass)

    return result.success, link_masses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=["PSM", "ECM"], required=True,
                        help = "type of arm")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help = "input file containing mass determination equations")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = "output file name")
    args = parser.parse_args()

    Eq = np.loadtxt(args.input)
    A = Eq[:, 0:-1]
    b = Eq[:, -1]

    arm_type = ArmType.PSM if args.type == "PSM" else ArmType.ECM

    if arm_type == ArmType.PSM:
        assert(A.shape[1] == 6 * 4)
    elif arm_type == ArmType.ECM:
        assert(A.shape[1] == 5 * 4)
    else:
        assert(False)

    ok, link_masses = solve(A, b, arm_type)
    if not ok:
        sys.exit(-1)

    with open(args.output, 'w') as f:
        for mass in link_masses:
            f.write(f"{mass}\n")

    print(f"\nLink masses saved to {args.output}")
