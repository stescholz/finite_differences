#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import exp, sqrt, pi
import fd


def example_1(N=100):
    C = 20.
    beta = 10.
    c = lambda x: np.full_like(x, C)
    d = lambda x: np.full_like(x, 0.0)
    f = lambda x: np.full_like(x, 0.0)

    def sol(x):
        return beta*(1-np.exp(C*x))/(1-exp(C))

    p = fd.FiniteDifference(0, 1, c, d, f, 0, beta, N)
    p.plot(sol)


def example_2(N=100):

    c = lambda x: np.full_like(x, 2.)
    d = lambda x: np.full_like(x, 1.)
    f = lambda x: x**2

    lambda_1 = 1+sqrt(2)
    lambda_2 = 1-sqrt(2)
    eta_2 = (10*exp(lambda_1)-7)/(exp(lambda_2)-exp(lambda_1))
    eta_1 = -eta_2-10

    def sol(x):
        return eta_1*np.exp(lambda_1*x)+eta_2*np.exp(lambda_2*x)+x**2-4*x+10

    p = fd.FiniteDifference(0, 1, c, d, f, 0, 0, N)
    p.plot(sol)


def example_3(N=100):

    Lambda = 2.
    Lambda = 4*pi**2

    c = lambda x: np.full_like(x, 0.)
    d = lambda x: np.full_like(x, -Lambda)
    f = lambda x: np.full_like(x, 0.)

    eta_2 = 1

    def sol(x):
        if Lambda % pi**2 == 0:
            return eta_2*np.sin(sqrt(Lambda)*x)
        else:
            return np.full_like(x, 0.)

    p = fd.FiniteDifference(0, 1, c, d, f, 0, 0, N)
    p.plot(sol)


def example_4(N=100):

    c = lambda x: -x/(1-x)
    d = lambda x: 1/(1-x)
    f = lambda x: np.full_like(x, 0.)

    def sol(x):
        return np.exp(x)+(1-exp(1))*x

    p = fd.FiniteDifference(0., 1., c, d, f, 1., 1., N)
    p.plot(sol)


if __name__ == '__main__':

    # example_1()
    # example_2()
    # example_3()
    example_4()
