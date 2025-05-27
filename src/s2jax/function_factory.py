"""
This file aims to create the desired jax/equinox functions
for each problem in our problem set. Previously s2mpj would inherit from a class
that would then search for the class which inherited from its available functions.
That method was carried over likely from their original MATLAB implementation, and
isnt very pythonic. We aim to partially correct that here.
"""

import numpy as np

# make_fx(x)

# def evalgrsum(isobj, glist, x, nargout):

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # dynamically allocate memory like pytorch does
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_log_compiles", True) # will print out recompilations
    jax.config.update('jax_default_matmul_precision', "default")
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


    import jax.numpy as jnp

    if jax.config.jax_enable_x64 is True:
        dtype = jnp.int64
    else:
        dtype = jnp.int32

    from jax.experimental.sparse import BCOO, BCSR

    import s2jax.sparse_utils as spu
    # from python_problems.DIAMON2D import DIAMON2D as PROBLEM
    # from python_problems.ACOPP14 import ACOPP14 as PROBLEM
    # from python_problems.test_ACOPP14_jax import ACOPP14 as PROBLEM
    from conversion_testing.HS107 import HS107 as PROBLEM

    problem = PROBLEM()

    # Extract the values of the global element's and group's parameters, if any.
    # Effectively a __post_init__().
    try: problem.e_globs(problem)
    except: pass
    try: problem.g_globs(problem)
    except: pass
    
    def make_fx(p):

        x = jnp.array(problem.x0)

        if (hasattr(p, "objgrps") and len(p.objgrps)) or hasattr(p, "H"):
            x = x.reshape(-1,1)

            # conditionals
            c = {
                "isobj": True,
                "nargout": 2,
                "has_conderlvl": hasattr(p, "conderlvl"), # constraint derivative level
                "has_objderlvl": hasattr(p, "objderlvl"), # objective derivative level
                "has_A": hasattr(p, "A"), # check presence of linear term
                "has_H": hasattr(p, "H"), # check presence of quadratic term
                
                # group conditionals
                "has_gscale": hasattr(p, "gscale"), # check presence of group scaling
                "has_gconst": hasattr(p, "gconst"), # check presence of linear term
                "has_grelt": hasattr(p, "grelt"), #
                "has_grelw": hasattr(p, "grelw"), #
                "has_grftype": hasattr(p,"grftype"), #      

                # efnames
                "name2efunc": {
                    name: getattr(p, name) for name in np.unique(p.elftype)
                }   
            }

            # do some preprocessing on the groups to evaluate things that do not depend on runtime variable parameters
            nouts = [] # level of derivatives
            gscs = [] # group scalings
            derlvls = [] # derivative levels on individual groups
            for ig in p.objgrps:
                # level of available derivatives for the group
                if  c["isobj"]:
                    if c["has_objderlvl"]:
                        derlvl = p.objderlvl
                    else:
                        derlvl = 2
                else:
                    if c["has_conderlvl"]:
                        if len(p.conderlvl) == 1:
                            derlvl = p.conderlvl[ 0 ]
                        else:
                            derlvl = p.conderlvl[np.where( p.congrps == ig )[0][0]]
                    else:
                        derlvl = 2
                nout = min(c["nargout"], derlvl + 1 )

                # group scaling
                if c["has_gscale"]:
                    if ig < len(p.gscale) and not p.gscale[ig] is None and abs(p.gscale[ig]) > 1.0e-15:
                        gsc = p.gscale[ig]
                    else:
                        gsc = 1.0
                else:
                    gsc = 1.0

                nouts.append(nout)
                gscs.append(gsc)
                derlvls.append(derlvl)

            c["nouts"] = nouts
            c["gscs"] = gscs
            c["derlvls"] = derlvls

            fx = lambda x: evalgrsum(p=p, x=x, c=c)

            return fx
        
        else:
            print( " ")
            print( "ERROR: problem "+p.name+" has no objective function!" )
            print( "       Please refer to the problem classification for checking a problem's type." )
            print( " ")

    def evalgrsum(p, x, c):

        glist = p.objgrps
        n = len(x)
        m = len(glist)

        # initializations
        fx = jnp.zeros([1]) # 0.
        cx = jnp.zeros([m, 1])
        ic = -1 # some constraint interator
        if c["has_conderlvl"]: lder = len(p.conderlvl)
        gx = jnp.zeros([n, 1])
        Jx = BCSR.from_bcoo(spu.bcoo_zeros([m, n], dtype=dtype)) # jnp.zeros([m, n]) # Sparse normally
        Hx = BCSR.from_bcoo(spu.bcoo_zeros([n, n], dtype=dtype)) # jnp.zeros([n, n]) # Sparse normally
        if c["has_A"]: sA1, sA2 = p.A.shape

        # Evaluate the quadratic term if any
        if c["isobj"] and c["has_H"]:
            Htimesx = p.H @ x
            gx += Htimesx
            fx += 0.5 * x.T @ (Htimesx)
            Hx = BCSR.from_bcoo(spu.bcoo_add(Hx.to_bcoo(), p.H.to_bcoo())) # += p.H

        # loop on the groups list
        for ig, nout, gsc, derlvl in zip(glist, c["nouts"], c["gscs"], c["derlvls"]):

            # evaluate the linear term if any
            if c["has_gconst"] and ig < len(p.gconst) and not p.gconst[ig] is None: fin = -p.gconst[ig]
            else: fin = 0
            if c["has_A"] and ig < sA1:
                gin           = jnp.zeros( (n, 1) )
                gin = gin.at[:sA2, :1].set(p.A.to_bcoo()[ ig, :sA2 ].T.todense()[:,None])
                fin           = fin + gin.T .dot(x)
            elif c["nargout"] >= 2: gin = jnp.zeros(( n, 1 ))

            if c["nargout"] > 2: Hin = BCSR.from_bcoo(spu.bcoo_zeros([n, n])) # np.zeros((n, n)) # Sparse normally

            if c["has_grelt"] and ig < len(p.grelt) and not p.grelt[ig] is None:
                for iiel in range(len(p.grelt[ig])): # loop on elements
                    iel    = p.grelt[ ig ][ iiel ]         #  the element's index
                    efname = p.elftype[ iel ]              #  the element's ftype
                    irange = [iv for iv in p.elvar[ iel ]] #  the elemental variable's indeces 
                    xiel   = x[ np.array(irange) ]            #  the elemental variable's values

                    if  c['has_grelw'] and ig <= len( p.grelw ) and not p.grelw[ig] is None :
                        has_weights = True
                        wiel        = p.grelw[ ig ][ iiel ]
                    else:
                        has_weights = False

                    # Only the value is requested.
                    if nout == 1:
                        fiel = c["name2efunc"][efname](p, 1, xiel, iel) # eval('self.'+efname +'( self, 1, xiel, iel )')
                        if ( has_weights ):
                            fin += wiel * fiel
                        else:
                            fin += fiel

                    elif nout == 2:
                        print(" NOT VALIDATED ")
                        fiel, giel = c["name2efunc"][efname](p, 2, xiel, iel) # eval('self.'+efname +'( self, 2, xiel, iel)')
                        if  has_weights:
                            fin += wiel * fiel
                            for ir in range(len(irange)):
                                ii = irange[ ir ]
                                gin = gin.at[ ii ].add(wiel * giel[ ir ])
                        else:
                            raise NotImplementedError("nargout == 2 not implemented yet")
                            fin = fin + fiel;
                            for ir in range(len(irange)):
                                ii = irange[ ir ]
                                gin[ ii ] += giel[ ir ]

                    elif nout == 3:
                        print(" NOT VALIDATED ")
                        fiel, giel, Hiel = c["name2efunc"][efname](p, 3, xiel, iel) # eval('self.'+efname +'( self, 3, xiel, iel )')
                        if has_weights:
                            fin += wiel * fiel
                            for ir in range(len(irange)):
                                ii = irange[ ir ]
                                gin[ ii ] += wiel * giel[ ir ]
                                for jr in range(len( irange )):
                                    jj  = irange[ jr ]
                                    Hin[ ii, jj ] += wiel * Hiel[ ir, jr ]
                        else:
                            fin = fin + fiel;
                            for ir in range(len(irange)):
                                ii = irange[ ir ]
                                gin[ ii ] += giel[ ir ]
                                for jr in range(len( irange )):
                                    jj  = irange[ jr ]
                                    Hin[ ii, jj ] += Hiel[ ir, jr ]

                    # raise NotImplementedError()
                    # pass

                # pass

            #  Evaluate the group function.
            #  1) the non-TRIVIAL case
            if c["has_grftype"] and ig < len(p.grftype) and not p.grftype[ig] is None: egname = p.grftype[ig]
            else: egname = "TRIVIAL"

            if egname!='TRIVIAL' and egname is not None:
                raise NotImplementedError()
            

            #  2) the TRIVIAL case: the group function is the identity
            else:

                if c["isobj"]:
                    if c["nargout"] == 1:
                        fx += fin / gsc 
                    if c["nargout"] == 2:
                        fx += fin / gsc
                        if derlvl >= 1:
                            gx += gin / gsc
                        else:
                            gx = jnp.nan * jnp.ones(( n, 1 ))
                    # if c["nargout"] == 3:
                    #     fx += fin / gsc
                    #     if derlvl >= 1:
                    #         gx += gin / gsc
                    #     else:
                    #         gx = jnp.zeros(( n, 1 ))
                    #         gx[0] = jnp.nan
                    #     if derlvl >= 2:
                    #         Hx += Hin / gsc
                    #     else:
                    #         Hx = BCSR.from_bcoo(spu.bcoo_zeros([n, n])) # np.zeros(( n, n )) # Sparse normally
                    #         Hx[0,0] = jnp.nan
                # pass

            # pass
            
        if c["isobj"]:
            if c["nargout"] == 1:
                return fx
            elif c["nargout"] == 2:
                return fx, gx.reshape(-1,1)
            elif c["nargout"] == 3:
                return fx, gx.reshape(-1,1), Hx
        else:
            if c["nargout"] == 1:
                return cx
            elif c["nargout"] == 2:
                return cx, Jx
            elif c["nargout"] == 3:
                return cx, Jx, Hx
        pass



    _fx = make_fx(problem)
    fx = lambda x: _fx(x)[0].flatten() # reshape(-1)  # ensure output is a 1-D array
    x = jnp.array(problem.x0)
    objective_value = fx(x)

    objective_grad = jax.jacobian(fx)(x)

    # objective_value = jax.jit(fx)(x)


    pass


