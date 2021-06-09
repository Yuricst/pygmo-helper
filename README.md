# pygmo-helper
Helper functions for [pygmo](https://esa.github.io/pygmo2/index.html)

Requirements: pygmo, pygmo_plugins_nonfree

### Using with SNOPT
#### Windows
To use SNOPT7 as the algorithm, provide the path to the `snopt7.dll` file. 

```python
import pygmo_plugins_nonfree as ppnf
path_to_snopt7 = "C:\path\to\snopt7.dll"
pygmoSnopt = ppnf.snopt7(screen_output=False, library=path_to_snopt7,  minor_version=7)
```

Optionally, add to system environment variables `C:\path\to\snopt7.dll` as `SNOPT_DLL`, then this may be accessed in Python as 

```python
import os
path_to_snopt7 = os.getenv('SNOPT_DLL')
```
