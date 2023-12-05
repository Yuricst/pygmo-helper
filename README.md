# pygmo-helper
Helper functions for [pygmo](https://esa.github.io/pygmo2/index.html)

Requirements: pygmo, pygmo_plugins_nonfree

Installation via conda requires channel configurations:

```shell
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pygmo
```

## Generic problem template

The generic shape of a pygmo UDP (user-defined problem) is as follows:

```python
import pygmo as pg

class MyUDP:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def fitness(self, x):
	"""Compute fitness for given decision vector x

	Args:
	    x (np.array-like): decision vector
	"""
        # compute fitness 
        # in order: objective, equality constraints, inequality constraints
	ceqs = [x[1]**3 - 5, ]
	cineqs = [x[0]**2 - 2x[1], ]
        return [obj,] + ceqs + cineqs
    
    # Number of equality Constraints
    def get_nec(self):
        return 4

    # Number inequality Constraints
    def get_nic(self):
        return 2

    # Lower and Upper bounds on x
    def get_bounds(self):
        return (self.lb, self.ub)
		
    # provide gradients
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
```

Methods inside the UDP may be jit-ed for speeds; for example, see the n-dimension Rosenbrock [example from the official doc](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html#notes-on-computational-speed):

```python
import numpy as np
import pygmo as pg
from numba import jit, float64

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
```


## Using pygmo with SNOPT
Using SNOPT7 requires `pygmo_plugins_nonfree` to be installed as well; see [official pygmo docs using on SNOPT](https://esa.github.io/pagmo_plugins_nonfree/py_snopt7.html). 

### Linux

The files to be downloaded are the **Fortran/C Libraries** (not C++). 
Then, provide path to 

```
export SNOPT_LICENSE=/home/path/to/snopt7.lic
export LD_LIBRARY_PATH=$HOME/path/to/libsnopt7   # ONLY FOR WINDOWS
export SNOPT_SO=$HOME/path/to/libsnopt7/libsnopt7.so   # optional but useful
```

### Windows
On Windows, provide the path to the `snopt7.dll` file. 

```python
import pygmo_plugins_nonfree as ppnf
path_to_snopt7 = "C:\path\to\snopt7.dll"
pygmoSnopt = ppnf.snopt7(screen_output=False, library=path_to_snopt7,  minor_version=7)  # MAKE SURE MINOR_VERSION IS CORRECT
```

**Note: SNOPT7 has its own gradient estimator which is better than `pg.estimate_gradient_h(lambda x: self.fitness(x), x)`, so we can choose to not implement the `self.gradient()` method in the UDP.**

Optionally, add to system environment variables `C:\path\to\snopt7.dll` as `SNOPT_DLL`:

1. From Start, search "Edit environment variables for your account" and open
2. Go to Environment Variables
3. Under "User Variables", click on New, and add entries for "SNOPT_LICENSE" (must) & "SNOPT_DLL" (optional)

Then, this may be accessed in Python as 

```python
import os
path_to_snopt7 = os.getenv('SNOPT_DLL')
```

### Common issues

- If the `minor_version` is not correct, a segmentation fault is thrown!!
