#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import matplotlib.backends.backend_tkagg
import matplotlib.pyplot as plt


class FiniteDifference:

    def __init__(self, a, b, c, d, f, alpha, beta, N):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.grid = np.linspace(self.a, self.b, N+1)
        self.c_ev = self.c(self.grid[1:self.N])
        self.d_ev = self.d(self.grid[1:self.N])

        # if not np.all(self.d_ev >= 0):
        #    raise ValueError("d(x) >= 0")

        if not np.all(np.abs(self.c_ev)/N < 2):
            raise ValueError("konvektionsdominantes Problem")

    def get_coefficient_matrix(self):
        A = np.diagflat(2*self.N**2+self.d_ev)
        A += np.diagflat(-self.N**2+0.5*self.N*self.c_ev[:-1], 1)
        A += np.diagflat(-self.N**2-0.5*self.N*self.c_ev[1:], -1)
        return A

    def get_rhs(self):
        rhs = self.f(self.grid[1:self.N])

        # boundary values
        rhs[0] += (self.N**2 + 0.5*self.N*self.c_ev[0])*self.alpha
        rhs[-1] += (self.N**2 - 0.5*self.N*self.c_ev[-1])*self.beta

        return rhs

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def calculate_solution(self):
        A = self.get_coefficient_matrix()
        b = self.get_rhs()
        return self.solve(A, b)

    def calculate_full_solution(self):
        u = np.zeros(self.N+1)
        u[1:self.N] = self.calculate_solution()

        # insert boundary values
        u[0] = self.alpha
        u[-1] = self.beta

        return u

    def plot(self, analytical_solution=None):

        U = self.calculate_full_solution()
        plt.plot(self.grid, U, label="approx. solution")

        if analytical_solution is not None:
            u = analytical_solution(self.grid)
            plt.plot(self.grid, u, label="analytical solution")

        plt.legend()
        plt.show()


class FiniteDifferenceSparse(FiniteDifference):

    def get_coefficient_matrix(self):
        diags = [2*self.N**2+self.d_ev,
                 -self.N**2+0.5*self.N*self.c_ev[:-1],
                 -self.N**2-0.5*self.N*self.c_ev[1:]]
        return sp.diags(diags, [0, 1, -1], format="csc")

    def solve(self, A, b):
        return sp_linalg.spsolve(A, b)


if __name__ == '__main__':

    def f(x):
        return np.full_like(x, 1.0)

    def c(x):
        return np.full_like(x, 10.0)

    def d(x):
        return np.full_like(x, 0.0)

    N = 100

    fd = FiniteDifference(0, 1, c, d, f, 0, 1, N)
    fd.plot()
