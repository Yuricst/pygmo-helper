"""
Run snopt on linux via pygmo and ppnf
Make sure the user environment variable `SNOPT_LICENSE` points to `snopt7.lic`. 
The `library` input in `ppnf.snopt7()` should point to the `libsnopt7.so` file. 
python 3.8.12
Pygmo: version 2.16.1
ppng: version 0.22
"""

import os
import numpy as np
import pygmo as pg
import pygmo_plugins_nonfree as ppnf


class Rosenbrock:
    def __init__(self,dim):
        self.dim = dim

    def fitness(self,x):
        return Rosenbrock._fitness(x)

    # jit-ted fitness-computation for faster computation
    #@jit(float64[:](float64[:]),nopython=True)
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


if __name__=="__main__":
	pygmoSnopt = ppnf.snopt7(
	    screen_output=True,
	    library=os.getenv('SNOPT_SO'),
	    minor_version=7
	)
	pygmoSnopt.set_numeric_option('Major feasibility tolerance', 1e-6)
	pygmoSnopt.set_numeric_option('Minor feasibility tolerance', 1e-6)
	pygmoSnopt.set_numeric_option('Major optimality tolerance', 1e-6)
	pygmoSnopt.set_numeric_option('Major step limit', 2)
	pygmoSnopt.set_integer_option('Iterations limit', 2000)
	algo = pg.algorithm(pygmoSnopt)

	udp = Rosenbrock(dim=10)
	prob = pg.problem(udp)

	pop = pg.population(prob, size=1)
	pop = algo.evolve(pop)

	print(pop.champion_x)
	print(pop.champion_f)
