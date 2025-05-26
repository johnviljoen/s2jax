import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp

def stack(bcoo_list):
    """Stack BCOO matrices along a new axis"""
    bcoo_list = [A[None,...] for A in bcoo_list]
    return jsp.bcoo_concatenate(bcoo_list, dimension=0)

def hstack(bcoo_list):
    """Horizontally concatenate BCOO matrices along columns (axis=1)."""
    return jsp.bcoo_concatenate(bcoo_list, dimension=1)

def vstack(bcoo_list):
    """Vertically concatenate BCOO matrices along rows (axis=0)."""
    return jsp.bcoo_concatenate(bcoo_list, dimension=0)

def bcoo_diagflat(data):
    n    = data.size
    ind  = jnp.arange(n, dtype=jnp.int32)
    indices = jnp.stack([ind, ind], axis=1)        # shape (n, 2)
    return jsp.BCOO((data, indices), shape=(n, n))

def bcoo_zeros(shape, dtype=jnp.int32):
    """Return an all-zeros BCOO sparse matrix with the requested shape."""
    ndim = len(shape)
    data    = jnp.empty((0,))          # no stored values
    indices = jnp.empty((0, ndim), dtype=dtype) # no coordinates
    return jsp.BCOO((data, indices), shape=shape)

def bcoo_eye(n):
    """n×n identity matrix in BCOO format."""
    inds  = jnp.arange(n, dtype=jnp.int32)
    data  = jnp.ones(n)
    # indices must be shape (nse, ndim) → here (n, 2) for 2-D
    indices = jnp.stack([inds, inds], axis=1)
    return jsp.BCOO((data, indices), shape=(n, n))

def bcoo_add(A, B, start_indices=[0,0]):
    # check that B fits entirely in A so we cannot place data in an illegal location
    start_indices = jnp.array(start_indices, dtype=A.indices.dtype)
    shifted_indices = B.indices + start_indices
    B = jsp.BCOO((B.data, shifted_indices), shape=jnp.array(B.shape) + start_indices)
    assert all([dimA >= dimB for dimA, dimB in zip(A.shape, B.shape)])
    return A + jsp.BCOO([B.data, B.indices], shape=A.shape)

# def bcoo_solve(b, csr_data, csr_indptr, csr_indices, device_id=0, mtype_id=0, mview_id=0, solve_id=0):
#     return solve(b, csr_data, csr_indptr, csr_indices, device_id, mtype_id, mview_id, solve_id)

# def bcoo_solve_LS(b: jnp.array, csr_data, csr_indptr, csr_indices, device_id=0, solve_id=0):
#     LHS = M.T @ M
#     RHS = M.T @ b
#     LHS_csr = jsp.BCSR.from_bcoo(LHS)
#     _csr_offsets, _csr_columns, _csr_values = LHS_csr.indptr, LHS_csr.indices, LHS_csr.data
#     return solve(RHS, _csr_values, _csr_offsets, _csr_columns, device_id, 1, 0, solve_id)

if __name__ == "__main__":
    zeros = bcoo_zeros([10,10])
    smol_eye = bcoo_eye(3)
    result = bcoo_add(zeros, smol_eye, start_indices=jnp.array([2,3]))

    result = jax.jit(bcoo_add)(zeros, smol_eye, start_indices=jnp.array([2,3]))

    print('fin')