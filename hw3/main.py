import objectives as obj
import leastsquares as ls
import numpy as np


# A total of six problems
problems = [["x = np.array([.5, 1.5, -1, .01, .02])",
             "fun = obj.Osborne1()"],
            ["x = np.array([1.3, .65, .65, .7, .6, 3, 5, 7, 2, 4.5, 5.5])",
             "fun = obj.Osborne2()"],
            ["n = 8",
             "x = np.arange(1, n + 1) / (n + 1)",
             "fun = obj.ChebyQuad(n)"],
            ["x = np.array([.3, .4])",
             "fun = obj.JennrichSampson(m=10)"],
            ["x = np.array([.02, 4000, 250])",
             "fun = obj.Meyer()"],
            ["x = np.array([25, 5, -5, -1])",
             "fun = obj.BrownDennis(m=20)"]]

# A total of eight algorithms
algorithms = ["res = ls.GaussNewton(fun, x)",
              "res = ls.LevenbergMarquardt(fun, x, method='lmf')",
              "res = ls.LevenbergMarquardt(fun, x, method='lmn')",
              "res = ls.Dogleg(fun, x, method='single')",
              "res = ls.Dogleg(fun, x, method='double')",
              "res = ls.ScipyMethod(fun, x)",
              "res = ls.DennisGayWelsch(fun, x, method='ls')",
              "res = ls.DennisGayWelsch(fun, x, method='tr')"]

# Solve each problem with every algorithm
for problem in problems:

    for setting in problem:
        exec(setting)
    for algorithm in algorithms:
        exec(algorithm)
        print('& %.3e & %.1f & %d & %d \\\\'%(
            res[1], -np.log(res[2]) / np.log(10), res[3], res[4]))
    print(' ')
