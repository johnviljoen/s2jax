import jax
import jax.numpy as jnp
import numpy as np

def append(arr: jnp.array, values: list):
    return jnp.append(arr, jnp.array(values))

# An emulation of the Matlab find() function for everything that can ne enumerated
def find( lst, condition ):
    return np.array([i for i, elem in enumerate(lst) if condition(elem)])

# Set the (i,j)-th element of a list of arrays (loa) to value.
def loaset( loa, i, j, value ):
    if len(loa) <= i:
       loa.extend( [None] * ( i - len( loa ) + 1 ) )
    if loa[i] is None:
       loa[i]= np.full( j + 1, None )
    if len(loa[i]) <= j:
       loa[i]= np.append(loa[i],np.full( j - len( loa[i] ) + 1, None ) )
    loa[i][j] = value
    return loa


# used by LEVYM.py, LEVYMONT8C.py
class structtype():
    def __str__(self):
        pprint(vars(self))
        return ''
    pass
    def __repr__(self):
        pprint(vars(self))
        return ''
    
# Computes the effective index name in List, or add name to List if not in there already. Return
# the index of name in List and new = 1 if the List has been enlarged or 0 if name was already
# present in List at the call.

def s2mpj_ii( name, List ):
    
    if name in List:
       idx = List[ name ]
       new = 0
    else:
       idx = len( List)
       List[ name ] = idx
       new = 1
    return idx, List, new

# Get the index of a nonlinear variable.  This implies adding it to the variables' dictionary ix_
# if it is a new one, and adjusting the bounds, start point and types according to their default
# setting.
def s2mpj_nlx( self, name, List, getxnames=None, xlowdef=None, xuppdef=None, x0def=None ):

    iv, List, newvar = s2mpj_ii( name, List );
    if( newvar ):
        self.n = self.n + 1;
        if getxnames:
            self.xnames = arrset( self.xnames, iv, name )
        if hasattr( self, "xlower" ):
            thelen = len( self.xlower )
            if ( iv <= thelen ):
                self.xlower = np.append( self.xlower, np.full( (iv-thelen+1,1), float(0.0) ), 0 )
            if not xlowdef is None:
                self.xlower[iv] 
        if hasattr( self, "xupper" ):
            thelen = len( self.xupper )
            if ( iv <= thelen ):
                self.xupper = np.append( self.xupper, np.full( (iv-thelen+1,1), float('Inf') ), 0 )
            if not xuppdef is None:
                self.xupper[iv] 
        try:
            self.xtype  = arrset( self.xtype, iv, 'r' )
        except:
            pass
        thelen = len( self.x0 )
        if ( iv <= thelen ):
            self.x0 = np.append( self.x0, np.full( (iv-thelen+1,1), 0.0 ), 0 )
        if not x0def is None:
            self.x0[iv] =  x0def
    return iv, List


# This is used for numpy arrays that carry strings for names (should be lists),
# but keeping it numpy as jnp doesnt have this functionality, and its not numerical
def arrset( arr, index, value ):

    if isinstance(value, float):
        if isinstance( index, jnp.ndarray):
            maxind = jnp.max( index )
        else:
            maxind = index
        if len(arr) <= maxind:
            arr= jnp.append( arr, jnp.full( maxind - len( arr ) + 1, None ) )
        arr = arr.at[index].set(jnp.array(value))
        return arr
    else:
        if isinstance( index, np.ndarray):
            maxind = np.max( index )
        else:
            maxind = index
        if len(arr) <= maxind:
            arr= np.append( arr, np.full( maxind - len( arr ) + 1, None ) )
        arr[index] = value
        return arr

# jittable lets goooooo
# def np_like_set(arr: jnp.ndarray, idx, val):
#     """Mimic NumPy's broadcasting rules for x[idx] = val."""
#     val = jnp.asarray(val)

#     # Get the *static* shape of the slice without materialising data
#     slice_shape = jax.eval_shape(lambda a: a[idx], arr).shape

#     # Broadcast if necessary
#     if val.shape != slice_shape:
#         val = jnp.broadcast_to(val, slice_shape)

#     return arr.at[idx].set(val)

def np_like_set(arr: jnp.ndarray, idx, val):
    """
    NumPy‑style slice assignment that works inside JAX‑traced code.

    Parameters
    ----------
    arr : jnp.ndarray
        The array to update.
    idx : Any
        Index expression (int, slice, tuple, Fancy, …).
    val : Any
        Value to write; will be broadcast to the slice shape
        following *NumPy* semantics (leading/trailing 1‑dims may be dropped).
    """
    val = jnp.asarray(val)

    # Shape of the target slice, but *static* (compile‑time) not run‑time
    slice_shape: Tuple[int, ...] = jax.eval_shape(lambda a: a[idx], arr).shape

    # ── Make `val` broadcast‑compatible with `slice_shape` ───────────
    if val.shape != slice_shape:

        # 1) Drop superfluous length‑1 axes until ndim ≤ target ndim
        while val.ndim > len(slice_shape) and val.shape[0] == 1:
            val = jnp.squeeze(val, axis=0)

        # 2) If slice is a scalar () ensure val is scalar too
        if slice_shape == () and val.shape == (1,):
            val = jnp.squeeze(val, axis=0)

        # 3) Final broadcast (now guaranteed to work or raise a clear error)
        val = jnp.broadcast_to(val, slice_shape)

    return arr.at[idx].set(val)

if __name__ == "__main__":
    
    test = jnp.array([1,2,3])
    test2 = jax.jit(np_like_set)(test, jnp.array([1]), 3)
    test2 = jax.jit(np_like_set)(test, jnp.array([1, 2]), 3)



    pass
