from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DALLASS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DALLASS
#    *********
# 
#    The small Dallas water distribution problem
#    The problem is also named "W30" in some references.
#    This is a nonlinear network problem with conditioning of
#    the order of 10**4.
# 
#    Source:
#    R. Dembo,
#    private communication, 1986.
# 
#    SIF input: Ph. Toint, June 1990.
# 
#    classification = "C-CONR2-MN-46-31"
# 
#    Number of arcs
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DALLASS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 46
        v_['NODES'] = 31
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X42']])
        valA = jtu.append(valA,float(-6.38400e+02))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X43']])
        valA = jtu.append(valA,float(-6.33000e+02))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X44']])
        valA = jtu.append(valA,float(-5.54500e+02))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X45']])
        valA = jtu.append(valA,float(-5.05000e+02))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X46']])
        valA = jtu.append(valA,float(-4.36900e+02))
        [ig,ig_,_] = jtu.s2mpj_ii('N1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X46']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X41']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X45']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X44']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X16']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X7']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X6']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N6')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X7']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X9']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X8']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N7',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N7')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X9']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X10']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N8',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N8')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X10']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X12']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X11']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N9',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N9')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X12']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X13']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N10',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N10')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X16']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X15']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X14']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N11',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N11')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X15']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X13']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X17']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N12',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N12')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X20']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X19']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X8']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X6']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X18']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N13',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N13')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X42']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X18']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X19']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N14',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N14')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X21']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X20']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N15',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N15')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X43']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X21']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N16',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N16')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X14']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X11']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X23']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X22']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N17',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N17')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X23']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X25']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X24']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N18',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X31']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X25']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X22']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X26']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N19',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X26']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X17']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X28']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X27']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N20',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X28']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N21',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N21')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X31']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X30']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X29']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N22',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N22')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X30']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X27']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N23',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N23')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X24']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X32']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N24',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N24')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X38']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X29']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X34']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X33']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N25',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N25')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X32']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X35']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N26',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N26')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X35']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X37']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X36']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N27',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N27')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X37']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X34']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N28',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N28')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X36']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X40']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X39']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X38']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N29',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N29')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X39']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X33']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N30',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N30')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X40']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X41']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('N31',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'N31')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X46']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X45']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X44']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X43']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X42']])
        valA = jtu.append(valA,float(-1.0))
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
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        self.gconst = jtu.arrset(self.gconst,ig_['N5'],float(2.80000))
        self.gconst = jtu.arrset(self.gconst,ig_['N7'],float(4.03000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N8'],float(5.92000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N9'],float(1.15600))
        self.gconst = jtu.arrset(self.gconst,ig_['N10'],float(2.00000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N11'],float(4.95000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N16'],float(3.13000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N17'],float(8.44000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N18'],float(3.31000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N19'],float(5.30000e-02))
        self.gconst = jtu.arrset(self.gconst,ig_['N21'],float(2.72000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N22'],float(8.83000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N23'],float(5.71000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N24'],float(7.55000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N26'],float(5.27000e-01))
        self.gconst = jtu.arrset(self.gconst,ig_['N29'],float(1.00000e-03))
        self.gconst = jtu.arrset(self.gconst,ig_['N31'],float(-1.01960e+01))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-2.00000e+02)
        self.xupper = jnp.full((self.n,1),2.00000e+02)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X1'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X1'], 2.11673e+01)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X2'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X2'], 4.37635e+01)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X3'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X3'], 3.28255e+01)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X19'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X19'], 2.20120e+01)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X21'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X21'], 1.36703e+01)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(-2.00000e+02))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(2.11673e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(2.11673e+01)))
        if('X2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(4.37635e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X2'])[0],float(4.37635e+01)))
        if('X3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(3.28255e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3']),float(3.28255e+01)))
        if('X4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(1.42109e-14))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X4'])[0],float(1.42109e-14)))
        if('X5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X5'], float(1.68826e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5']),float(1.68826e+02)))
        if('X7' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X7'], float(2.81745e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X7'])[0],float(2.81745e+01)))
        if('X8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X8'], float(8.75603e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X8']),float(8.75603e+01)))
        if('X9' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X9'], float(-5.93858e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X9'])[0],float(-5.93858e+01)))
        if('X10' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X10'], float(-5.97888e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X10']),float(-5.97888e+01)))
        if('X11' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X11'], float(1.83383e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X11'])[0],float(1.83383e+02)))
        if('X13' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X13'], float(-1.68331e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X13']),float(-1.68331e+02)))
        if('X15' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X15'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X15'])[0],float(2.00000e+02)))
        if('X16' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X16'], float(2.00000e-01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X16']),float(2.00000e-01)))
        if('X17' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X17'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X17'])[0],float(2.00000e+02)))
        if('X18' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X18'], float(-7.67574e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X18']),float(-7.67574e+01)))
        if('X19' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X19'], float(2.20120e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X19'])[0],float(2.20120e+01)))
        if('X20' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X20'], float(1.36703e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X20']),float(1.36703e+01)))
        if('X21' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X21'], float(1.36703e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X21'])[0],float(1.36703e+01)))
        if('X22' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X22'], float(-1.98461e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X22']),float(-1.98461e+02)))
        if('X23' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X23'], float(1.81531e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X23'])[0],float(1.81531e+02)))
        if('X24' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X24'], float(-1.93133e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X24']),float(-1.93133e+01)))
        if('X25' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X25'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X25'])[0],float(2.00000e+02)))
        if('X26' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X26'], float(-1.98792e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X26']),float(-1.98792e+02)))
        if('X27' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X27'], float(1.15500))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X27'])[0],float(1.15500)))
        if('X28' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X28'], float(0.00000))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X28']),float(0.00000)))
        if('X29' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X29'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X29'])[0],float(2.00000e+02)))
        if('X30' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X30'], float(2.72000e-01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X30']),float(2.72000e-01)))
        if('X32' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X32'], float(-1.98843e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X32'])[0],float(-1.98843e+01)))
        if('X33' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X33'], float(1.78834e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X33']),float(1.78834e+02)))
        if('X34' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X34'], float(-1.79589e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X34'])[0],float(-1.79589e+02)))
        if('X35' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X35'], float(-1.98843e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X35']),float(-1.98843e+01)))
        if('X37' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X37'], float(1.79589e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X37'])[0],float(1.79589e+02)))
        if('X40' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X40'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X40']),float(2.00000e+02)))
        if('X41' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X41'], float(2.00000e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X41'])[0],float(2.00000e+02)))
        if('X42' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X42'], float(9.87694e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X42']),float(9.87694e+01)))
        if('X43' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X43'], float(1.36703e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X43'])[0],float(1.36703e+01)))
        if('X44' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X44'], float(3.28255e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X44']),float(3.28255e+01)))
        if('X45' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X45'], float(4.37635e+01))
        else:
            self.y0  = (                   jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X45'])[0],float(4.37635e+01)))
        if('X46' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X46'], float(-1.78833e+02))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X46']),float(-1.78833e+02)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eT1', iet_)
        elftv = jtu.loaset(elftv,it,0,'ARC')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'C1')
        elftp = jtu.loaset(elftp,it,1,'C2')
        elftp = jtu.loaset(elftp,it,2,'C3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eT4', iet_)
        elftv = jtu.loaset(elftv,it,0,'ARC')
        elftp = jtu.loaset(elftp,it,0,'C1')
        elftp = jtu.loaset(elftp,it,1,'C2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'E1'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eT4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eT4"])
        vname = 'X1'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.48060e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.51200e+02))
        ename = 'E2'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eT4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eT4"])
        vname = 'X2'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.91526e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.46300e+01))
        ename = 'E3'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eT4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eT4"])
        vname = 'X3'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.07752e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.81400e+01))
        ename = 'E4'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X4'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.90000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.22000e+02))
        ename = 'E5'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X5'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        ename = 'E6'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X6'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.63000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+02))
        ename = 'E7'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X7'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.10000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.22000e+02))
        ename = 'E8'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X8'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.45000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+02))
        ename = 'E9'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X9'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(7.40000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.22000e+02))
        ename = 'E10'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X10'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.00000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.50000e+01))
        ename = 'E11'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X11'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(8.00000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.07000e+02))
        ename = 'E12'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X12'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.20000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.80000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E13'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X13'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.00000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.80000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E14'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X14'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.00000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        ename = 'E15'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X15'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.12200e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.30000e+02))
        ename = 'E16'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X16'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.50000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.22000e+02))
        ename = 'E17'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X17'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.10000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E18'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X18'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.80000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.18000e+02))
        ename = 'E19'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eT4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eT4"])
        vname = 'X19'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.84530e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.12970e+02))
        ename = 'E20'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X20'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.60000e+04))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.80000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E21'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eT4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eT4"])
        vname = 'X21'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.86880e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.60610e+02))
        ename = 'E22'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X22'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.20000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.36100e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.30000e+02))
        ename = 'E23'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X23'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.60000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.50000e+01))
        ename = 'E24'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X24'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.50000e+01))
        ename = 'E25'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X25'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.60000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E26'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X26'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.30000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E27'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X27'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.20000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.24000e+02))
        ename = 'E28'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X28'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E29'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X29'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.90000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.13000e+02))
        ename = 'E30'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X30'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.80000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.13000e+02))
        ename = 'E31'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X31'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.70000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E32'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X32'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.10000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.50000e+01))
        ename = 'E33'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X33'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        ename = 'E34'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X34'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(4.30000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.13000e+02))
        ename = 'E35'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X35'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.20000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.50000e+01))
        ename = 'E36'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X36'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.80000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(5.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E37'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X37'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.40000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+02))
        ename = 'E38'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X38'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.31000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        ename = 'E39'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X39'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(6.65000e+02))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+02))
        ename = 'E40'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X40'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.10000e+03))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.60000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.20000e+02))
        ename = 'E41'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eT1')
            ielftype = jtu.arrset(ielftype,ie,iet_['eT1'])
        vname = 'X41'
        [iv,ix_]  = (               jtu.s2mpj_nlx(self,vname,ix_,1,float(-2.00000e+02),float(2.00000e+02),float(-2.00000e+02)))
        posev = jnp.where(elftv[ielftype[ie]]=='ARC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='C1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.23000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+01))
        posep = jnp.where(elftp[ielftype[ie]]=='C3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.00000e+02))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E4'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E5'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E7'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E8'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E9'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E10'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E11'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E12'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E13'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E14'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E15'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E16'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E17'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E18'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E19'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E20'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E21'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E22'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E23'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E24'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E25'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E26'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E27'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E28'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E29'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E30'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E31'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E32'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E33'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E34'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E35'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E36'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E37'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E38'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E39'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E40'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E41'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               -3.2393D+04
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CONR2-MN-46-31"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eT1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        TMP = 850559.0e0/2.85*self.elpar[iel_][0]
        TMP = TMP/(self.elpar[iel_][2]**1.85)
        TMP = TMP/(self.elpar[iel_][1]**4.87)
        X = jnp.absolute(EV_[0])
        XEXP = X**0.85
        f_   = TMP*X**2*XEXP
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 2.85*TMP*EV_[0]*XEXP)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 5.2725*TMP*XEXP)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eT4(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EPS2 = 1.0e-14
        SQC1 = jnp.sqrt(self.elpar[iel_][0])
        X = min(EV_[0],SQC1)
        TMP = self.elpar[iel_][1]*(self.elpar[iel_][0]-X*X)
        TMP = jnp.sqrt(max(TMP,EPS2))
        TMP2 = jnp.sqrt(self.elpar[iel_][1])*jnp.arcsin(X/SQC1)
        f_   = 0.5*(-X*TMP-self.elpar[iel_][0]*TMP2)
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -TMP)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), self.elpar[iel_][1]*X/TMP)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

