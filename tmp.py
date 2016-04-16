
# Burkert
import numpy as np
def func(x,y,z,alpha=1.):
    r=np.sqrt(x*x+y*y+z*z)
    return 1.424/(1+(r**2/5**2))
    
    
