import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class WATER:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A small nonlinear network problem.
#    The problem is to compute the flows in a water distribution network
#    with 7 nodes and 8 links, subject to known supply/demand at the nodes 
#    and a unique reservoir at node 1.
# 
#    The problem is convex.
# 
#    Source:
#    an exercize for L. Watson course on LANCELOT in the Spring 1993.
# 
#    SIF input: E. P. Smith, Virginia Tech., Spring 1993.
# 
#    classification = "C-CONR2-MN-31-10"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'WATER'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [ig,ig_,_] = jtu.s2mpj_ii('obj0102',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(105665.6))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0203',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(3613.412))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0204',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(105665.6))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0305',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(890.1553))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0405',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(76.66088))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0406',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(55145.82))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0607',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(26030.46))
        [ig,ig_,_] = jtu.s2mpj_ii('obj0705',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(890.1553))
        [ig,ig_,_] = jtu.s2mpj_ii('obj',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('c1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c1')
        [ig,ig_,_] = jtu.s2mpj_ii('c2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c2')
        [ig,ig_,_] = jtu.s2mpj_ii('c3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c3')
        [ig,ig_,_] = jtu.s2mpj_ii('c4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c4')
        [ig,ig_,_] = jtu.s2mpj_ii('c5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c5')
        [ig,ig_,_] = jtu.s2mpj_ii('c6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c6')
        [ig,ig_,_] = jtu.s2mpj_ii('c7',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c7')
        [ig,ig_,_] = jtu.s2mpj_ii('c8',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c8')
        [ig,ig_,_] = jtu.s2mpj_ii('c9',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c9')
        [ig,ig_,_] = jtu.s2mpj_ii('c10',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'c10')
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        ngrp   = len(ig_)
        [iv,ix_,_] = jtu.s2mpj_ii('Q0102',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0102')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0102']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c1']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0102',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0102')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0203',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0203')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0203']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0203',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0203')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0204',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0204')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0204']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0204',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0204')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0305',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0305')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0305']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0305',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0305')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0405',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0405')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0405']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0405',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0405')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0406',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0406')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0406']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0406',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0406')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0607',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0607')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0607']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0607',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0607')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0705',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0705')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj0705']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0705',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0705')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q01u0',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q01u0')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c1']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q01u0',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q01u0')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c8']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y02up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y02up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y02up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y02up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y03up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y03up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y03up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y03up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y04up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y04up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y04up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y04up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y05up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y05up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y05up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y05up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y06up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y06up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y06up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y06up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('y07up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y07up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(210))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('y07up',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'y07up')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu02',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu02')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-175))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu02',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu02')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu03',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu03')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-190))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu03',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu03')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu04',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu04')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-185))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu04',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu04')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu05',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu05')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-180))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu05',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu05')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu06',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu06')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-195))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu06',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu06')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu07',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu07')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['obj']])
        valA = jtu.append(valA,float(-190))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('yqu07',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yqu07')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0201',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0201')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c1']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0302',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0302')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0402',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0402')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c2']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0503',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0503')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c3']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0504',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0504')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0604',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0604')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c4']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0507',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0507')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c5']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(-1))
        [iv,ix_,_] = jtu.s2mpj_ii('Q0706',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q0706')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c6']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c7']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yupu0',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yupu0')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c8']])
        valA = jtu.append(valA,float(-1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c9']])
        valA = jtu.append(valA,float(1))
        [iv,ix_,_] = jtu.s2mpj_ii('yu0uq',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'yu0uq')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c8']])
        valA = jtu.append(valA,float(1))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['c10']])
        valA = jtu.append(valA,float(-1))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
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
        self.gconst = jtu.arrset(self.gconst,ig_['c1'],float(1120))
        self.gconst = jtu.arrset(self.gconst,ig_['c2'],float(-100))
        self.gconst = jtu.arrset(self.gconst,ig_['c3'],float(-100))
        self.gconst = jtu.arrset(self.gconst,ig_['c4'],float(-120))
        self.gconst = jtu.arrset(self.gconst,ig_['c5'],float(-270))
        self.gconst = jtu.arrset(self.gconst,ig_['c6'],float(-330))
        self.gconst = jtu.arrset(self.gconst,ig_['c7'],float(-200))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0102'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0203'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0204'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0305'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0405'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0406'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0607'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0705'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q01u0'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y02up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y03up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y04up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y05up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y06up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['y07up'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu02'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu03'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu04'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu05'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu06'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yqu07'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0201'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0302'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0402'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0503'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0504'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0604'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0507'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q0706'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yupu0'], 1200)
        self.xupper = jtu.np_like_set(self.xupper, ix_['yu0uq'], 1200)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gPOWER',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['obj0102']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0203']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0204']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0305']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0405']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0406']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0607']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        ig = ig_['obj0705']
        self.grftype = jtu.arrset(self.grftype,ig,'gPOWER')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SOLUTION           1.054938D+04
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CONR2-MN-31-10"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gPOWER(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_**2.852
        if nargout>1:
            g_ = 2.852*GVAR_**1.852
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 5.282*GVAR_**.852
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

