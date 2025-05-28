import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class FERRISDC:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : FERRISDC
#    *********
# 
#    A QP suggested by Michael Ferris
#    classification = "C-C"
#    SIF input: Nick Gould, November 2001.
# 
#    classification = "C-CQLR2-AN-V-V"
# 
#           Alternative values for the SIF file parameters:
# IE n                   4              $-PARAMETER
# IE n                   100            $-PARAMETER
# IE n                   200            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'FERRISDC'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['n'] = int(4);  #  SIF file default value
        else:
            v_['n'] = int(args[0])
# IE n                   300            $-PARAMETER
# IE k                   3              $-PARAMETER
# IE k                   10             $-PARAMETER
        if nargin<2:
            v_['k'] = int(3);  #  SIF file default value
        else:
            v_['k'] = int(args[1])
# IE k                   20             $-PARAMETER
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['12'] = 12.0
        v_['24'] = 24.0
        v_['240'] = 240.0
        v_['k-1'] = -1+v_['k']
        v_['k'] = float(v_['k'])
        v_['k-1'] = float(v_['k-1'])
        v_['-1/k-1'] = -1.0/v_['k-1']
        v_['-1/k'] = -1.0/v_['k']
        v_['n'] = float(v_['n'])
        v_['1'] = 1.0
        v_['2'] = 2.0
        v_['1/12'] = v_['1']/v_['12']
        v_['1/24'] = v_['1']/v_['24']
        v_['1/240'] = v_['1']/v_['240']
        v_['7/240'] = 7.0*v_['1/240']
        v_['2**2'] = v_['2']*v_['2']
        v_['2**4'] = v_['2**2']*v_['2**2']
        v_['2**8'] = v_['2**4']*v_['2**4']
        v_['2**10'] = v_['2**8']*v_['2**2']
        v_['2**16'] = v_['2**8']*v_['2**8']
        v_['2**26'] = v_['2**16']*v_['2**10']
        v_['2**-26'] = v_['1']/v_['2**26']
        v_['nlambda'] = v_['n']*v_['2**-26']
        v_['-1/k-1*nl'] = v_['nlambda']*v_['-1/k-1']
        v_['ix'] = 1
        v_['ax'] = 16807
        v_['b15'] = 32768
        v_['b16'] = 65536
        v_['pp'] = 2147483647
        v_['pp'] = float(v_['pp'])
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['xhi'] = int(jnp.fix(v_['ix']/v_['b16']))
            v_['xalo'] = v_['xhi']*v_['b16']
            v_['xalo'] = v_['ix']-v_['xalo']
            v_['xalo'] = v_['xalo']*v_['ax']
            v_['leftlo'] = int(jnp.fix(v_['xalo']/v_['b16']))
            v_['fhi'] = v_['xhi']*v_['ax']
            v_['fhi'] = v_['fhi']+v_['leftlo']
            v_['kk'] = int(jnp.fix(v_['fhi']/v_['b15']))
            v_['dum'] = v_['leftlo']*v_['b16']
            v_['dum'] = v_['xalo']-v_['dum']
            v_['ix'] = v_['dum']-v_['pp']
            v_['dum'] = v_['kk']*v_['b15']
            v_['dum'] = v_['fhi']-v_['dum']
            v_['dum'] = v_['dum']*v_['b16']
            v_['ix'] = v_['ix']+v_['dum']
            v_['ix'] = v_['ix']+v_['kk']
            v_['a'] = float(v_['ix'])
            v_['a'] = -1.0*v_['a']
            v_['b'] = 0.0
            v_['absa'] = jnp.absolute(v_['a'])
            v_['absb'] = jnp.absolute(v_['b'])
            v_['absa+b'] = v_['absa']+v_['absb']
            v_['absa+b+2'] = 2.0+v_['absa+b']
            v_['a'] = v_['a']+v_['absa+b+2']
            v_['b'] = v_['b']+v_['absa+b+2']
            v_['a/b'] = v_['a']/v_['b']
            v_['b/a'] = v_['b']/v_['a']
            v_['a/b'] = int(jnp.fix(v_['a/b']))
            v_['b/a'] = int(jnp.fix(v_['b/a']))
            v_['a/b'] = float(v_['a/b'])
            v_['b/a'] = float(v_['b/a'])
            v_['sum'] = v_['a/b']+v_['b/a']
            v_['a'] = v_['a']*v_['a/b']
            v_['b'] = v_['b']*v_['b/a']
            v_['maxa,b'] = v_['a']+v_['b']
            v_['maxa,b'] = v_['maxa,b']/v_['sum']
            v_['c'] = v_['absa+b+2']-v_['maxa,b']
            v_['a'] = v_['absa+b+2']-v_['a']
            v_['absc'] = jnp.absolute(v_['c'])
            v_['absc+1'] = 1.0+v_['absc']
            v_['absc+2'] = 2.0+v_['absc']
            v_['f'] = v_['absc+2']/v_['absc+1']
            v_['f'] = int(jnp.fix(v_['f']))
            v_['g'] = 2-v_['f']
            for l in range(int(v_['1']),int(v_['g'])+1):
                v_['ix'] = v_['ix']+v_['pp']
            v_['randp'] = float(v_['ix'])
            v_['X'+str(j)] = v_['randp']/v_['pp']
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['xhi'] = int(jnp.fix(v_['ix']/v_['b16']))
            v_['xalo'] = v_['xhi']*v_['b16']
            v_['xalo'] = v_['ix']-v_['xalo']
            v_['xalo'] = v_['xalo']*v_['ax']
            v_['leftlo'] = int(jnp.fix(v_['xalo']/v_['b16']))
            v_['fhi'] = v_['xhi']*v_['ax']
            v_['fhi'] = v_['fhi']+v_['leftlo']
            v_['kk'] = int(jnp.fix(v_['fhi']/v_['b15']))
            v_['dum'] = v_['leftlo']*v_['b16']
            v_['dum'] = v_['xalo']-v_['dum']
            v_['ix'] = v_['dum']-v_['pp']
            v_['dum'] = v_['kk']*v_['b15']
            v_['dum'] = v_['fhi']-v_['dum']
            v_['dum'] = v_['dum']*v_['b16']
            v_['ix'] = v_['ix']+v_['dum']
            v_['ix'] = v_['ix']+v_['kk']
            v_['a'] = float(v_['ix'])
            v_['a'] = -1.0*v_['a']
            v_['b'] = 0.0
            v_['absa'] = jnp.absolute(v_['a'])
            v_['absb'] = jnp.absolute(v_['b'])
            v_['absa+b'] = v_['absa']+v_['absb']
            v_['absa+b+2'] = 2.0+v_['absa+b']
            v_['a'] = v_['a']+v_['absa+b+2']
            v_['b'] = v_['b']+v_['absa+b+2']
            v_['a/b'] = v_['a']/v_['b']
            v_['b/a'] = v_['b']/v_['a']
            v_['a/b'] = int(jnp.fix(v_['a/b']))
            v_['b/a'] = int(jnp.fix(v_['b/a']))
            v_['a/b'] = float(v_['a/b'])
            v_['b/a'] = float(v_['b/a'])
            v_['sum'] = v_['a/b']+v_['b/a']
            v_['a'] = v_['a']*v_['a/b']
            v_['b'] = v_['b']*v_['b/a']
            v_['maxa,b'] = v_['a']+v_['b']
            v_['maxa,b'] = v_['maxa,b']/v_['sum']
            v_['c'] = v_['absa+b+2']-v_['maxa,b']
            v_['a'] = v_['absa+b+2']-v_['a']
            v_['absc'] = jnp.absolute(v_['c'])
            v_['absc+1'] = 1.0+v_['absc']
            v_['absc+2'] = 2.0+v_['absc']
            v_['f'] = v_['absc+2']/v_['absc+1']
            v_['f'] = int(jnp.fix(v_['f']))
            v_['g'] = 2-v_['f']
            for l in range(int(v_['1']),int(v_['g'])+1):
                v_['ix'] = v_['ix']+v_['pp']
            v_['randp'] = float(v_['ix'])
            v_['R'+str(j)] = v_['randp']/v_['pp']
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['arg'] = -3.0*v_['X'+str(j)]
            v_['arg'] = jnp.exp(v_['arg'])
            v_['P'+str(j)+','+str(int(v_['1']))] = 0.97*v_['arg']
            v_['arg'] = -1.2+v_['X'+str(j)]
            v_['arg'] = v_['arg']*v_['arg']
            v_['arg'] = -2.5*v_['arg']
            v_['P'+str(j)+','+str(int(v_['3']))] = jnp.exp(v_['arg'])
            v_['arg'] = v_['1']-v_['P'+str(j)+','+str(int(v_['1']))]
            v_['P'+str(j)+','+str(int(v_['2']))] = (v_['arg']-v_['P'+str(j)+','+                  str(int(v_['3']))])
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['a'] = v_['P'+str(j)+','+str(int(v_['1']))]-v_['R'+str(j)]
            v_['a'] = -1.0*v_['a']
            v_['b'] = 0.0
            v_['absa'] = jnp.absolute(v_['a'])
            v_['absb'] = jnp.absolute(v_['b'])
            v_['absa+b'] = v_['absa']+v_['absb']
            v_['absa+b+2'] = 2.0+v_['absa+b']
            v_['a'] = v_['a']+v_['absa+b+2']
            v_['b'] = v_['b']+v_['absa+b+2']
            v_['a/b'] = v_['a']/v_['b']
            v_['b/a'] = v_['b']/v_['a']
            v_['a/b'] = int(jnp.fix(v_['a/b']))
            v_['b/a'] = int(jnp.fix(v_['b/a']))
            v_['a/b'] = float(v_['a/b'])
            v_['b/a'] = float(v_['b/a'])
            v_['sum'] = v_['a/b']+v_['b/a']
            v_['a'] = v_['a']*v_['a/b']
            v_['b'] = v_['b']*v_['b/a']
            v_['maxa,b'] = v_['a']+v_['b']
            v_['maxa,b'] = v_['maxa,b']/v_['sum']
            v_['c'] = v_['absa+b+2']-v_['maxa,b']
            v_['a'] = v_['absa+b+2']-v_['a']
            v_['absc'] = jnp.absolute(v_['c'])
            v_['absc+1'] = 1.0+v_['absc']
            v_['absc+2'] = 2.0+v_['absc']
            v_['f'] = v_['absc+2']/v_['absc+1']
            v_['f'] = int(jnp.fix(v_['f']))
            v_['g'] = 2-v_['f']
            for l1 in range(int(v_['g']),int(v_['0'])+1):
                v_['y'+str(j)] = 1.0
            for l1 in range(int(v_['1']),int(v_['g'])+1):
                v_['a'] = v_['1']-v_['P'+str(j)+','+str(int(v_['3']))]
                v_['a'] = v_['a']-v_['R'+str(j)]
                v_['a'] = -1.0*v_['a']
                v_['b'] = 0.0
                v_['absa'] = jnp.absolute(v_['a'])
                v_['absb'] = jnp.absolute(v_['b'])
                v_['absa+b'] = v_['absa']+v_['absb']
                v_['absa+b+2'] = 2.0+v_['absa+b']
                v_['a'] = v_['a']+v_['absa+b+2']
                v_['b'] = v_['b']+v_['absa+b+2']
                v_['a/b'] = v_['a']/v_['b']
                v_['b/a'] = v_['b']/v_['a']
                v_['a/b'] = int(jnp.fix(v_['a/b']))
                v_['b/a'] = int(jnp.fix(v_['b/a']))
                v_['a/b'] = float(v_['a/b'])
                v_['b/a'] = float(v_['b/a'])
                v_['sum'] = v_['a/b']+v_['b/a']
                v_['a'] = v_['a']*v_['a/b']
                v_['b'] = v_['b']*v_['b/a']
                v_['maxa,b'] = v_['a']+v_['b']
                v_['maxa,b'] = v_['maxa,b']/v_['sum']
                v_['c'] = v_['absa+b+2']-v_['maxa,b']
                v_['a'] = v_['absa+b+2']-v_['a']
                v_['absc'] = jnp.absolute(v_['c'])
                v_['absc+1'] = 1.0+v_['absc']
                v_['absc+2'] = 2.0+v_['absc']
                v_['f'] = v_['absc+2']/v_['absc+1']
                v_['f'] = int(jnp.fix(v_['f']))
                v_['g'] = 2-v_['f']
                for l2 in range(int(v_['g']),int(v_['0'])+1):
                    v_['y'+str(j)] = 2.0
                for l2 in range(int(v_['1']),int(v_['g'])+1):
                    v_['y'+str(j)] = 3.0
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['yj'] = v_['y'+str(j)]
            v_['yj'] = int(jnp.fix(v_['yj']))
            for i in range(int(v_['1']),int(v_['k'])+1):
                v_['c'] = v_['yj']-i
                v_['c'] = float(v_['c'])
                v_['absc'] = jnp.absolute(v_['c'])
                v_['absc+1'] = 1.0+v_['absc']
                v_['absc+2'] = 2.0+v_['absc']
                v_['f'] = v_['absc+2']/v_['absc+1']
                v_['f'] = int(jnp.fix(v_['f']))
                v_['g'] = 2-v_['f']
                for l in range(int(v_['g']),int(v_['0'])+1):
                    v_['Y'+str(i)+','+str(j)] = v_['nlambda']
                for l in range(int(v_['1']),int(v_['g'])+1):
                    v_['Y'+str(i)+','+str(j)] = v_['-1/k-1*nl']
        for i in range(int(v_['1']),int(v_['n'])+1):
            v_['di'] = -0.5+v_['X'+str(i)]
            v_['di2'] = v_['di']*v_['di']
            v_['di2'] = v_['di2']-v_['1/12']
            for j in range(int(i),int(v_['n'])+1):
                v_['Xi-Xj'] = v_['X'+str(i)]-v_['X'+str(j)]
                v_['bij'] = jnp.absolute(v_['Xi-Xj'])
                v_['dj'] = -0.5+v_['X'+str(j)]
                v_['dj2'] = v_['dj']*v_['dj']
                v_['dj2'] = v_['dj2']-v_['1/12']
                v_['c'] = -0.5+v_['bij']
                v_['c2'] = v_['c']*v_['c']
                v_['c4'] = v_['c2']*v_['c2']
                v_['c2'] = -0.5*v_['c2']
                v_['arg'] = v_['7/240']+v_['c2']
                v_['arg'] = v_['arg']+v_['c4']
                v_['arg'] = v_['arg']*v_['1/24']
                v_['dij'] = v_['di']*v_['dj']
                v_['dij2'] = v_['di2']*v_['dj2']
                v_['dij2'] = 0.25*v_['dij2']
                v_['arg'] = v_['dij2']-v_['arg']
                v_['K'+str(i)+','+str(j)] = v_['dij']+v_['arg']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for i in range(int(v_['1']),int(v_['k'])+1):
            for j in range(int(v_['1']),int(v_['n'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('A'+str(i)+','+str(j),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'A'+str(i)+','+str(j))
        for i in range(int(v_['1']),int(v_['n'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('W'+str(i),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'W'+str(i))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for j in range(int(v_['1']),int(v_['n'])+1):
            for i in range(int(v_['1']),int(v_['k'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['A'+str(i)+','+str(j)]])
                valA = jtu.append(valA,float(v_['Y'+str(i)+','+str(j)]))
        for i in range(int(v_['1']),int(v_['k'])+1):
            for j in range(int(v_['1']),int(v_['n'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(i),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'C'+str(i))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['A'+str(i)+','+str(j)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['W'+str(j)]])
                valA = jtu.append(valA,float(v_['-1/k']))
        for j in range(int(v_['1']),int(v_['n'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('A'+str(j),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'A'+str(j))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['W'+str(j)]])
            valA = jtu.append(valA,float(-1.0))
            for i in range(int(v_['1']),int(v_['k'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('A'+str(j),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'A'+str(j))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['A'+str(i)+','+str(j)]])
                valA = jtu.append(valA,float(1.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        legrps = jnp.where(gtype=='<=')[0]
        eqgrps = jnp.where(gtype=='==')[0]
        gegrps = jnp.where(gtype=='>=')[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m   = self.nle+self.neq+self.nge
        self.congrps = jnp.concatenate((legrps,eqgrps,gegrps))
        self.cnames = cnames[self.congrps]
        self.nob = ngrp-self.m
        self.objgrps = jnp.where(gtype=='<>')[0]
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),0.0)
        self.xupper = jnp.full((self.n,1),1.0)
        for i in range(int(v_['1']),int(v_['n'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['W'+str(i)], -float('Inf'))
            self.xupper = jtu.np_like_set(self.xupper, ix_['W'+str(i)], +float('Inf'))
        for j in range(int(v_['1']),int(v_['n'])+1):
            v_['yj'] = v_['y'+str(j)]
            v_['yj'] = int(jnp.fix(v_['yj']))
            for i in range(int(v_['1']),int(v_['k'])+1):
                v_['c'] = v_['yj']-i
                v_['c'] = float(v_['c'])
                v_['absc'] = jnp.absolute(v_['c'])
                v_['absc+1'] = 1.0+v_['absc']
                v_['absc+2'] = 2.0+v_['absc']
                v_['f'] = v_['absc+2']/v_['absc+1']
                v_['f'] = int(jnp.fix(v_['f']))
                v_['g'] = 2-v_['f']
                for l in range(int(v_['g']),int(v_['0'])+1):
                    self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['A'+str(i)+','+str(j)]]), 0.0)
                    self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['A'+str(i)+','+str(j)]]), 0.0)
        #%%%%%%%%%%%%%%%%%%%% QUADRATIC %%%%%%%%%%%%%%%%%%%
        irH  = jnp.array([],dtype=int)
        icH  = jnp.array([],dtype=int)
        valH = jnp.array([],dtype=float)
        for i in range(int(v_['1']),int(v_['k'])+1):
            for l in range(int(v_['1']),int(v_['n'])+1):
                for j in range(int(v_['1']),int(l)+1):
                    irH  = jtu.append(irH,[ix_['A'+str(i)+','+str(j)]])
                    icH  = jtu.append(icH,[ix_['A'+str(i)+','+str(l)]])
                    valH = jtu.append(valH,float(v_['K'+str(j)+','+str(l)]))
                    irH  = jtu.append(irH,[ix_['A'+str(i)+','+str(l)]])
                    icH  = jtu.append(icH,[ix_['A'+str(i)+','+str(j)]])
                    valH = jtu.append(valH,float(v_['K'+str(j)+','+str(l)]))
        for l in range(int(v_['1']),int(v_['n'])+1):
            for j in range(int(v_['1']),int(l)+1):
                v_['coef'] = v_['-1/k']*v_['K'+str(j)+','+str(l)]
                irH  = jtu.append(irH,[ix_['W'+str(j)]])
                icH  = jtu.append(icH,[ix_['W'+str(l)]])
                valH = jtu.append(valH,float(v_['coef']))
                irH  = jtu.append(irH,[ix_['W'+str(l)]])
                icH  = jtu.append(icH,[ix_['W'+str(j)]])
                valH = jtu.append(valH,float(v_['coef']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# XL SOLUTION            -1.131846D+2   $ nlambda = 1.5625
# XL SOLUTION            -8.032841E-5   $ nlambda = 1.4901E-06
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        self.H = BCSR.from_bcoo(BCOO((valH, jnp.array((irH,icH)).T), shape=(self.n,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CQLR2-AN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

