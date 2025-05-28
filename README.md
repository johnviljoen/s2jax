# NOTICE: 

This repo is incomplete and will likely stay that way. This is because I came to the
conclusion that integrating PyCUTEst in a black box manner was sufficient for my purposes.
I leave this here in case I (or anyone else) ever want to come back to transpiling the 1103 problems from
s2mpj into JAX properly. Do be aware PyCUTEst has all ~1500 problems of CUTEst I believe.


# S2JAX

This repo recieves the Python created problems from s2mpj and then converts them
to be JAX/Equinox compatible. The original s2mpj code worked as follows:

1. regenerate.m would decode the SIF files of CUTEst into Python/NumPy problems
2. These Python/NumPy problems would be interpreted by an additional library s2mpjlib.py (there were equivalents for Julia and MATLAB).
3. These interpreters would expose Objectives, Jacobians, Hessians to the end user

This ended up being incompatible with JAX for numerous reasons. The NumPy code was written in such a way that almost the same logic can be used in Julia/MATLAB. This was so that regenerate.m could simply have switch statements and the overall logic be roughly the same. This led to very non-Pythonic, but more importantly, unjittable, undifferentiable JAX code if directly copied. Furthermore there were numerous edge cases of NumPy that were used that did not have a 1-to-1 mapping in JAX. (numpy arrays of strings/objects, weird down broadcasting, heavy usage of in-place updates)

This repo contains a "transpiler" in convert.py (using that word liberally), which takes in the Python/NumPy problems and adjusts them through heavy usage of regex to be able to be JAXified. Then we also rewrite the problem interpreter to be JAXified such that we return jittable, vmappable, differentiable functions which represent what we need from the optimization problem.


