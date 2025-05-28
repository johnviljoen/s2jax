import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LOADBAL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    The problem arises in the field of computer networks and parallel
#    computation.  It deals with the static load balancing in a tree
#    computer network with two-way traffic.  A set of heterogeneous host
#    computers are interconnected, in which each node processes jobs (the 
#    jobs arriving at each node according to a time invariant Poisson process) 
#    locally or sends it to a remote node,.  In the latter case, there is a
#    communication delay of forwarding the job and getting a response back.
#    The problem is then to minimize the mean response time of a job.
# 
#    The example considered here features 11 computers arranged as follows:
# 
#          1      6      9
#           \     |     /
#            \    |    /
#         2---4---5---8---10
#            /    |    \
#           /     |     \
#          3      7      11
# 
#    Source:
#    J. Li and H. Kameda,
#    "Optimal load balancing in tree network with two-way traffic",
#    Computer networks and ISDN systems, vol. 25, pp. 1335-1348, 1993.
# 
#    SIF input: Masha Sosonkina, Virginia Tech., 1995.
# 
#    classification = "C-COLR2-MN-31-31"
# 
#  Parameter assignment.
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LOADBAL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['P1'] = 3
        v_['N'] = 11
        v_['NLINK'] = 20
        v_['NLINK-3'] = 17
        v_['NLINK-4'] = 16
        v_['4C'] = 4
        v_['5C'] = 5
        v_['6C'] = 6
        v_['7C'] = 7
        v_['8C'] = 8
        v_['FI'] = 514.0
        v_['0.2*FI'] = 0.2*v_['FI']
        v_['0.0125*FI'] = 0.0125*v_['FI']
        v_['0.05*FI'] = 0.05*v_['FI']
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['0.2*FI']))
        for I in range(int(v_['1']),int(v_['NLINK'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CNST'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'CNST'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('GA'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['0.0125*FI']))
            [ig,ig_,_] = jtu.s2mpj_ii('GB'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['0.05*FI']))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('N'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'N'+str(I))
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        ngrp   = len(ig_)
        [iv,ix_,_] = jtu.s2mpj_ii('X4,1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,1')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N1']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X1,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X1,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N1']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,2')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N2']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X2,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X2,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N2']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,3')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N3']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X3,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X3,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N3']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N4']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,6',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,6')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N6']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X6,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X6,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N6']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,7',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,7')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N7']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X7,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X7,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N7']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N5']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,9',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,9')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N9']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X9,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X9,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N9']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,10',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,10')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N10']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X10,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X10,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N10']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,11',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,11')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N11']])
        valA = jtu.append(valA,float(1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(-1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X11,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X11,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N11']])
        valA = jtu.append(valA,float(-1.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['N8']])
        valA = jtu.append(valA,float(1.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,1')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST1']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST2']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X1,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X1,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST1']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST2']])
        valA = jtu.append(valA,float(20.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,2')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST3']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST4']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X2,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X2,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST3']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST4']])
        valA = jtu.append(valA,float(20.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,3')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST5']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST6']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X3,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X3,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST5']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST6']])
        valA = jtu.append(valA,float(20.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,6',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,6')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST7']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST8']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X6,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X6,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST7']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST8']])
        valA = jtu.append(valA,float(20.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,7',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,7')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST9']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST10']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X7,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X7,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST9']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST10']])
        valA = jtu.append(valA,float(20.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,9',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,9')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST11']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST12']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X9,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X9,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST11']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST12']])
        valA = jtu.append(valA,float(20.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,10',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,10')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST13']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST14']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X10,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X10,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST13']])
        valA = jtu.append(valA,float(80.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST14']])
        valA = jtu.append(valA,float(20.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,11',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,11')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST15']])
        valA = jtu.append(valA,float(20.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST16']])
        valA = jtu.append(valA,float(80.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X11,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X11,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST15']])
        valA = jtu.append(valA,float(80.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST16']])
        valA = jtu.append(valA,float(20.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X4,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST17']])
        valA = jtu.append(valA,float(20.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST18']])
        valA = jtu.append(valA,float(80.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,4')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST17']])
        valA = jtu.append(valA,float(80.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST18']])
        valA = jtu.append(valA,float(20.00))
        [iv,ix_,_] = jtu.s2mpj_ii('X5,8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5,8')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST19']])
        valA = jtu.append(valA,float(20.00))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST20']])
        valA = jtu.append(valA,float(80.0))
        [iv,ix_,_] = jtu.s2mpj_ii('X8,5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X8,5')
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST19']])
        valA = jtu.append(valA,float(80.0))
        icA  = jtu.append(icA,[iv])
        irA  = jtu.append(irA,[ig_['CNST20']])
        valA = jtu.append(valA,float(20.00))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('B'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'B'+str(I))
            icA  = jtu.append(icA,[iv])
            irA  = jtu.append(irA,[ig_['N'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
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
        self.gconst = jtu.arrset(self.gconst,ig_['N1'],float(-95.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N2'],float(-95.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N3'],float(-19.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N4'],float(-70.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N5'],float(-70.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N6'],float(-19.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N7'],float(-19.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N8'],float(-70.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N9'],float(-19.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N10'],float(-19.0))
        self.gconst = jtu.arrset(self.gconst,ig_['N11'],float(-19.0))
        v_['CIJE'] = 999.99
        for I in range(int(v_['1']),int(v_['NLINK-4'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['CNST'+str(I)],float(v_['CIJE']))
        v_['CIJE'] = 9999.99
        for I in range(int(v_['NLINK-3']),int(v_['NLINK'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['CNST'+str(I)],float(v_['CIJE']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['B1'], 99.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B2'], 99.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B4'], 99.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B5'], 99.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B8'], 99.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B3'], 19.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B6'], 19.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B7'], 19.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B9'], 19.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B10'], 19.99)
        self.xupper = jtu.np_like_set(self.xupper, ix_['B11'], 19.99)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('X1,4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X1,4']]), float(00.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1,4']),float(00.0)))
        if('X4,1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,1']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X4,1']),float(0.0)))
        if('B1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B1'], float(95.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B1']),float(95.0)))
        if('X2,4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X2,4']]), float(0.00))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X2,4']),float(0.00)))
        if('X4,2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,2']]), float(0.00))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X4,2']),float(0.00)))
        if('B2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B2'], float(95.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B2']),float(95.0)))
        if('X3,4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X3,4']]), float(0.00))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3,4']),float(0.00)))
        if('X4,3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,3']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X4,3']),float(0.0)))
        if('B3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B3'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B3']),float(19.0)))
        if('B4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B4'], float(70.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B4']),float(70.0)))
        if('X5,4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,4']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5,4']),float(0.0)))
        if('X4,5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,5']]), float(00.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X4,5']),float(00.0)))
        if('X6,5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X6,5']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X6,5']),float(0.0)))
        if('X5,6' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,6']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5,6']),float(0.0)))
        if('X7,5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X7,5']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X7,5']),float(0.0)))
        if('X5,7' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,7']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5,7']),float(0.0)))
        if('B5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B5'], float(70.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B5']),float(70.0)))
        if('B6' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B6'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B6']),float(19.0)))
        if('B7' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B7'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B7']),float(19.0)))
        if('X8,5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,5']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X8,5']),float(0.0)))
        if('X5,8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,8']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5,8']),float(0.0)))
        if('X9,8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X9,8']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X9,8']),float(0.0)))
        if('X8,9' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,9']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X8,9']),float(0.0)))
        if('X10,8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X10,8']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X10,8']),float(0.0)))
        if('X8,10' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,10']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X8,10']),float(0.0)))
        if('X11,8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X11,8']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X11,8']),float(0.0)))
        if('X8,11' in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,11']]), float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X8,11']),float(0.0)))
        if('B8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B8'], float(70.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B8']),float(70.0)))
        if('B9' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B9'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B9']),float(19.0)))
        if('B10' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B10'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B10']),float(19.0)))
        if('B11' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['B11'], float(19.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['B11']),float(19.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eBETA1', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eBETA2', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCOMA1', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCOMA2', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCOMB1', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCOMB2', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'EB1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA1"])
        vname = 'B1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA1"])
        vname = 'B2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB4'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA1"])
        vname = 'B4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB5'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA1"])
        vname = 'B5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB6'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB7'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB8'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA1"])
        vname = 'B8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB9'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB10'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB11'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBETA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBETA2"])
        vname = 'B11'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['P1'])+1):
            ename = 'EGA'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            vname = 'X'+str(int(v_['4C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(I)+','+str(int(v_['4C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            vname = 'X'+str(int(v_['4C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(I)+','+str(int(v_['4C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['I+3'] = 3+I
            ename = 'EGA'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            ename = 'EGA'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['4C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGA'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['4C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            ename = 'EGB'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['4C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['4C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['I+6'] = 6+I
            v_['I+8'] = 8+I
            ename = 'EGA'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            ename = 'EGA'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['8C']))+','+str(int(v_['I+8']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGA'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+8']))+','+str(int(v_['8C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            ename = 'EGB'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['8C']))+','+str(int(v_['I+8']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+6']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+8']))+','+str(int(v_['8C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['I+9'] = 9+I
            ename = 'EGA'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            ename = 'EGA'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+8']))+','+str(int(v_['8C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGA'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['8C']))+','+str(int(v_['I+8']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            ename = 'EGB'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+8']))+','+str(int(v_['8C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I+9']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['8C']))+','+str(int(v_['I+8']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['6C']),int(v_['7C'])+1):
            v_['I2'] = 2*I
            v_['I2+1'] = 1+v_['I2']
            ename = 'EGA'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            ename = 'EGA'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['5C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGA'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['5C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            ename = 'EGB'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['5C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I2+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['5C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['I2+2'] = 2+v_['I2']
            ename = 'EGA'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMA1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA1"])
            ename = 'EGA'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['5C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGA'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['5C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCOMB1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB1"])
            ename = 'EGB'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(I)+','+str(int(v_['5C']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EGB'+str(int(v_['I2+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['5C']))+','+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGA17'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA2"])
        vname = 'X5,4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGB17'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMB2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB2"])
        vname = 'X5,4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGA18'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA2"])
        vname = 'X4,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5,4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGB18'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMB2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB2"])
        vname = 'X4,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5,4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGA19'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA2"])
        vname = 'X5,8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGB19'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMB2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB2"])
        vname = 'X5,8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGA20'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMA2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMA2"])
        vname = 'X8,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5,8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EGB20'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCOMB2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCOMB2"])
        vname = 'X8,5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5,8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['F'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EB'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['1']),int(v_['NLINK'])+1):
            ig = ig_['GB'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EGB'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['GA'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EGA'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COLR2-MN-31-31"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE ELEMENTS *
#  ROUTINE             *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(self):

        import jax.numpy as jnp
        self.efpar = jnp.array([])
        self.efpar = jtu.arrset( self.efpar,0,80.0)
        self.efpar = jtu.arrset( self.efpar,1,20.0)
        return pbm

    @staticmethod
    def eBETA2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CB = 20.0
        f_   = EV_[0]/(CB-EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, CB/((CB-EV_[0])**2))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2*CB/((CB-EV_[0])**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eBETA1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CB = 100.0
        f_   = EV_[0]/(CB-EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, CB/((CB-EV_[0])**2))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2*CB/((CB-EV_[0])**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCOMA1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CIJ = 1000.0
        f_   = EV_[0]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, (                   (CIJ-self.efpar[1]*EV_[1])/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**2))
            g_ = jtu.np_like_set(g_, 1, (                   EV_[0]*self.efpar[1]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**2))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (                       2*self.efpar[0]*(CIJ-self.efpar[1]*EV_[1])/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), (self.efpar[1]*(CIJ+self.efpar[0]*EV_[0]-self.efpar[1]*EV_[1])/                      (CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (                       2*self.efpar[1]*self.efpar[1]*EV_[0]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCOMB1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CIJ = 1000.0
        f_   = EV_[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, (                   (CIJ-self.efpar[0]*EV_[1])/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**2))
            g_ = jtu.np_like_set(g_, 1, (                   EV_[0]*self.efpar[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**2))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (                       2*self.efpar[1]*(CIJ-self.efpar[0]*EV_[1])/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), (self.efpar[0]*(CIJ+self.efpar[1]*EV_[0]-self.efpar[0]*EV_[1])/                      (CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (                       2*self.efpar[0]*self.efpar[0]*EV_[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCOMA2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CIJ = 10000.0
        f_   = EV_[0]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, (                   (CIJ-self.efpar[1]*EV_[1])/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**2))
            g_ = jtu.np_like_set(g_, 1, (                   EV_[0]*self.efpar[1]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**2))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (                       2*self.efpar[0]*(CIJ-self.efpar[1]*EV_[1])/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), (self.efpar[1]*(CIJ+self.efpar[0]*EV_[0]-self.efpar[1]*EV_[1])/                      (CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (                       2*self.efpar[1]*self.efpar[1]*EV_[0]/(CIJ-(self.efpar[0]*EV_[0]+self.efpar[1]*EV_[1]))**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCOMB2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CIJ = 10000.0
        f_   = EV_[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, (                   (CIJ-self.efpar[0]*EV_[1])/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**2))
            g_ = jtu.np_like_set(g_, 1, (                   EV_[0]*self.efpar[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**2))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (                       2*self.efpar[1]*(CIJ-self.efpar[0]*EV_[1])/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), (self.efpar[0]*(CIJ+self.efpar[1]*EV_[0]-self.efpar[0]*EV_[1])/                      (CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (                       2*self.efpar[0]*self.efpar[0]*EV_[0]/(CIJ-(self.efpar[1]*EV_[0]+self.efpar[0]*EV_[1]))**3))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

