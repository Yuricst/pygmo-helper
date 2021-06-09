"""
Test pygmo & snopt
"""

import os
from sys import platform
import numpy as np
from numba import jit, float64
import pygmo as pg
import pygmo_plugins_nonfree as ppnf


# PATH TO SNOPT LIBRARY --------------------------------- #
if platform == "linux" or platform == "linux2":
    # linux
    # provide path to snopt7.so as environment variable 'SNOPT_SO'
    path_to_snopt7 = os.getenv('SNOPT_SO')

elif platform == "darwin":
    # OS X
    print("Not yet worked on...")

elif platform == "win32":
    # Windows...
	# provide path to snopt7.dll as environment variable 'SNOPT_DLL'
	path_to_snopt7 = os.getenv('SNOPT_DLL')
print(f"Using path to snopt7: {path_to_snopt7}")
# ------------------------------------------------------- #

class Rosenbrock:
    def __init__(self,dim):
        self.dim = dim


    def fitness(self,x):
        return Rosenbrock._fitness(x)

    # jit-ted fitness-computation for faster computation
    @jit(float64[:](float64[:]),nopython=True)
    def _fitness(x):
        retval = np.zeros((1,))
        for i in range(len(x) - 1):
            tmp1 = (x[i + 1]-x[i]*x[i])
            tmp2 = (1.-x[i])
            retval[0] += 100.*tmp1*tmp1+tmp2*tmp2
        return retval

    def get_bounds(self):
        return (np.full((self.dim,),-5.),np.full((self.dim,),10.))

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


def main():
    # define udp
    udp = Rosenbrock(dim=100)
    prob = pg.problem(udp)
    # c_tol = 1E-6
    # prob.c_tol = [c_tol]*(nec + nic)   # number of nonlinear constraints

    # set population
    pop = pg.population(prob, size=1)


    # use snopt
    pygmoSnopt = ppnf.snopt7(screen_output=False, library=path_to_snopt7,  minor_version=7)
    pygmoSnopt.set_numeric_option('Major feasibility tolerance', 1e-6)
    pygmoSnopt.set_numeric_option('Minor feasibility tolerance', 1e-6)
    pygmoSnopt.set_numeric_option('Major optimality tolerance', 1e-6)
    pygmoSnopt.set_numeric_option('Major step limit', 2)
    pygmoSnopt.set_integer_option('Iterations limit', 2000)  # shoul be 4e3

    # # pure SNOPT
    algo = pg.algorithm(pygmoSnopt)
    algo.set_verbosity(2)
    print(algo)

    # solve
    pop = algo.evolve(pop)

    # print result
    xbest = pop.get_x()[pop.best_idx()]
    fbest = pop.get_f()[pop.best_idx()]
    print(f"fbest: {fbest}")
    print(f"xbest: {xbest}")
    print("Done!")


if __name__=="__main__":
    main()