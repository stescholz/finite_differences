#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import matplotlib.backends.backend_tkagg
import matplotlib.pyplot as plt


class FiniteDifference:
    """Finite difference method to solve a differential equation in 1D

    Implementation of the finite difference method to solve the differential
    equation -u'' + c u' + d u = f on the square [a,b] with boundary
    condition u(a)=alpha and u(b)=beta numerically.

    Parameters
    ----------
    a : float
        Left boundary of the domain.
    b : float
        Right boundary of the domain.
    c : function
        The function in front of the first derivative.
    d : function
        The function in front of u.
    f : function
        The right hand side of the equation.
    alpha : float
        Left boundary value u(a).
    beta : float
        Right boundary value u(b).
    N: int
        Number of gridpoints minus 1, step size h=1/N.

    Attributes
    ----------
    a : float
        Left boundary of the domain.
    b : float
        Right boundary of the domain.
    c : function
        The function in front of the first derivative.
    d : function
        The function in front of u.
    f : function
        The right hand side of the equation.
    alpha : float
        Left boundary value u(a).
    beta : float
        Right boundary value u(b).
    N: int
        Number of gridpoints minus 1, step size h=1/N.
    grid: array, shape=(N+1,)
        Grid points.
    c_ev: array, shape(N-1,)
        Evaluation of the function c at the inner grid points.
    d_ev: array, shape(N-1,)
        Evaluation of the function d at the inner grid points.
    """
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

        if not np.all(np.abs(self.c_ev)/N < 2):
            raise ValueError("convection dominated problem")

    def get_coefficient_matrix(self):
        """Build the coefficient matrix of the problem

        Returns
        -------
        array, shape(N-1, N-1)
            The coefficient matrix.
        """

        A = np.diagflat(2*self.N**2+self.d_ev)
        A += np.diagflat(-self.N**2+0.5*self.N*self.c_ev[:-1], 1)
        A += np.diagflat(-self.N**2-0.5*self.N*self.c_ev[1:], -1)
        return A

    def get_rhs(self):
        """Build the right hand side of the problem

        Returns
        -------
        array, shape(N-1, )
            The right hand side.
        """

        rhs = self.f(self.grid[1:self.N])

        # boundary values
        rhs[0] += (self.N**2 + 0.5*self.N*self.c_ev[0])*self.alpha
        rhs[-1] += (self.N**2 - 0.5*self.N*self.c_ev[-1])*self.beta

        return rhs

    def solve(self, A, b):
        """Solve the system of linear equations Ax=b

        Parameters
        ----------
        A: array, shape(n, n)
            The coefficient matrix
        b: array, shape(n, )
            The right hand side.

        Returns
        -------
        array, shape(n,)
            The solution x
        """
        return np.linalg.solve(A, b)

    def calculate_solution(self):
        """Solve the system of linear equations for the finite differnces

        Returns
        -------
        array, shape(N-1,)
            The solution u at the inner grid points
        """
        A = self.get_coefficient_matrix()
        b = self.get_rhs()
        return self.solve(A, b)

    def calculate_full_solution(self):
        """Get the full solution u

        Returns
        -------
        array, shape(N+1,)
            The solution u at all grid points
        """

        u = np.zeros(self.N+1)
        u[1:self.N] = self.calculate_solution()

        # insert boundary values
        u[0] = self.alpha
        u[-1] = self.beta

        return u

    def plot(self, analytical_solution=None):
        """Create a plot of the approximative solution

        Parameters
        ----------
        analytical_solution: function, optional
            Add a second plot of the analytical solution of the problem.
        """
        U = self.calculate_full_solution()
        plt.plot(self.grid, U, label="approx. solution")

        if analytical_solution is not None:
            u = analytical_solution(self.grid)
            plt.plot(self.grid, u, label="analytical solution")

        plt.legend()
        plt.show()


class FiniteDifferenceSparse(FiniteDifference):
    """Finite difference method with sparse matrices"""
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
