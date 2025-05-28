import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LINVERSE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
#    Problem : LINVERSE
#    *********
# 
#    The problem is to jtu.find the positive definite lower bidiagonal
#    matrix L such that the matrix L(inv)L(inv-transp) best approximates,
#    in the Frobenius norm, a given symmetric target matrix T.
#    More precisely, one is  interested in the positive definite lower
#    bidiagonal L such that
# 
#         || L T L(transp) - I ||     is minimum.
#                                F
# 
#    The positive definite character of L is imposed by requiring
#    that all its diagonal entries to be at least equal to EPSILON,
#    a strictly positive real number.
# 
#    Many variants of the problem can be obtained by varying the target
#    matrix T and the scalar EPSILON.  In the present problem,
#    a) T is chosen to be pentadiagonal with T(i,j) = sin(i)cos(j) (j .leq. i)
#    b) EPSILON = 1.D-8
# 
#    Source:
#    Ph. Toint, private communication, 1991.
# 
#    SIF input: Ph. Toint, March 1991.
# 
#    classification = "C-CSBR2-AN-V-0"
# 
#    Dimension of the matrix
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER  n = 19    original value
# IE N                   100            $-PARAMETER  n = 199
# IE N                   500            $-PARAMETER  n = 999
# IE N                   1000           $-PARAMETER  n = 1999
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LINVERSE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        v_['EPSILON'] = 1.0e-8
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['N-1'] = -1+v_['N']
        v_['N-2'] = -2+v_['N']
        for J in range(int(v_['1']),int(v_['N-2'])+1):
            v_['J+2'] = 2+J
            v_['RJ'] = float(J)
            v_['COSJ'] = jnp.cos(v_['RJ'])
            for I in range(int(J),int(v_['J+2'])+1):
                v_['RI'] = float(I)
                v_['SINI'] = jnp.sin(v_['RI'])
                v_['T'+str(I)+','+str(J)] = v_['SINI']*v_['COSJ']
        v_['RN-1'] = float(v_['N-1'])
        v_['SINI'] = jnp.sin(v_['RN-1'])
        v_['COSJ'] = jnp.cos(v_['RN-1'])
        v_['T'+str(int(v_['N-1']))+','+str(int(v_['N-1']))] = v_['SINI']*v_['COSJ']
        v_['RN'] = float(v_['N'])
        v_['SINI'] = jnp.sin(v_['RN'])
        v_['T'+str(int(v_['N']))+','+str(int(v_['N-1']))] = v_['SINI']*v_['COSJ']
        v_['COSJ'] = jnp.cos(v_['RN'])
        v_['T'+str(int(v_['N']))+','+str(int(v_['N']))] = v_['SINI']*v_['COSJ']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('A'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'A'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('B'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'B'+str(I))
        [iv,ix_,_] = jtu.s2mpj_ii('A'+str(int(v_['N'])),ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A'+str(int(v_['N'])))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for J in range(int(v_['1']),int(v_['N-2'])+1):
            v_['J+1'] = 1+J
            v_['J+2'] = 2+J
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(J)+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(int(v_['J+1']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(0.5))
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(int(v_['J+2']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(0.5))
        [ig,ig_,_] = jtu.s2mpj_ii('O'+str(int(v_['N-1']))+','+str(int(v_['N-1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('O'+str(int(v_['N']))+','+str(int(v_['N-1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(0.5))
        [ig,ig_,_] = jtu.s2mpj_ii('O'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['O'+str(I)+','+str(I)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['A'+str(I)], v_['EPSILON'])
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(-1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'S'+str(int(v_['1']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['1']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['2']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['2']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['2']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['2']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['1']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['2']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['3']))+','+str(int(v_['2']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'S'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'U'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'A'+str(int(v_['3']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'V'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'W'+str(int(v_['3']))+','+str(int(v_['3']))
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'B'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['4']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            v_['I-2'] = -2+I
            ename = 'S'+str(I)+','+str(int(v_['I-2']))
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
            vname = 'A'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'S'+str(I)+','+str(int(v_['I-2']))
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
            vname = 'A'+str(int(v_['I-2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'V'+str(I)+','+str(int(v_['I-2']))
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
            vname = 'B'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'V'+str(I)+','+str(int(v_['I-2']))
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
            vname = 'A'+str(int(v_['I-2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            for J in range(int(v_['I-1']),int(I)+1):
                v_['J-1'] = -1+J
                ename = 'S'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
                vname = 'A'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'A'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'U'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
                vname = 'A'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'B'+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'V'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
                vname = 'B'+str(int(v_['I-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'A'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'W'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
                vname = 'B'+str(int(v_['I-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'B'+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(-1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for ig in range(0,ngrp):
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        ig = ig_['O'+str(int(v_['1']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['1']))+','+str(int(v_['1']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['1']))+','+str(int(v_['1']))])))
        ig = ig_['O'+str(int(v_['2']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['2']))+','+str(int(v_['1']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['1']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['V'+str(int(v_['2']))+','+str(int(v_['1']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['1']))+','+str(int(v_['1']))])))
        ig = ig_['O'+str(int(v_['3']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['3']))+','+str(int(v_['1']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['1']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['V'+str(int(v_['3']))+','+str(int(v_['1']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['1']))])))
        ig = ig_['O'+str(int(v_['2']))+','+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['2']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['2']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['U'+str(int(v_['2']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['1']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['V'+str(int(v_['2']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['1']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['W'+str(int(v_['2']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['1']))+','+str(int(v_['1']))])))
        ig = ig_['O'+str(int(v_['3']))+','+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['3']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['2']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['U'+str(int(v_['3']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['1']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['V'+str(int(v_['3']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['2']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['W'+str(int(v_['3']))+','+str(int(v_['2']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['1']))])))
        ig = ig_['O'+str(int(v_['3']))+','+str(int(v_['3']))]
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['3']))+','+str(int(v_['3']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['3']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['U'+str(int(v_['3']))+','+str(int(v_['3']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['2']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['V'+str(int(v_['3']))+','+str(int(v_['3']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['3']))+','+str(int(v_['2']))])))
        posel = len(self.grelt[ig])
        self.grelt  = (               jtu.loaset(self.grelt,ig,posel,ie_['W'+str(int(v_['3']))+','+str(int(v_['3']))]))
        self.grelw  = (               jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['2']))+','+str(int(v_['2']))])))
        for I in range(int(v_['4']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            v_['I-2'] = -2+I
            ig = ig_['O'+str(I)+','+str(int(v_['I-2']))]
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(int(v_['I-2']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(int(v_['I-2']))])))
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['V'+str(I)+','+str(int(v_['I-2']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['I-1']))+','+str(int(v_['I-2']))])))
            ig = ig_['O'+str(I)+','+str(int(v_['I-1']))]
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(int(v_['I-1']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(int(v_['I-1']))])))
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['U'+str(I)+','+str(int(v_['I-1']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(int(v_['I-2']))])))
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['V'+str(I)+','+str(int(v_['I-1']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['I-1']))+','+str(int(v_['I-1']))])))
            posel = len(self.grelt[ig])
            self.grelt  = (                   jtu.loaset(self.grelt,ig,posel,ie_['W'+str(I)+','+str(int(v_['I-1']))]))
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['I-1']))+','+str(int(v_['I-2']))])))
            ig = ig_['O'+str(I)+','+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(I)]))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U'+str(I)+','+str(I)])
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(int(v_['I-1']))])))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['V'+str(I)+','+str(I)])
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(I)+','+str(int(v_['I-1']))])))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['W'+str(I)+','+str(I)])
            self.grelw  = (                   jtu.loaset(self.grelw,ig,posel,float(v_['T'+str(int(v_['I-1']))+','+str(int(v_['I-1']))])))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN(10)           6.00000000
# LO SOLTN(100)          68.0000000
# LO SOLTN(500)          340.000000
# LO SOLTN(1000)         ???
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSBR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_*GVAR_
        if nargout>1:
            g_ = GVAR_+GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

