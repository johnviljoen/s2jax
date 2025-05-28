import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class KISSING:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem: KISSING NUMBER PROBLEM
#                                                                    
#    Source: This problem is associated to the family of Hard-Spheres 
#    problem. It belongs to the family of sphere packing problems, a 
#    class of challenging problems dating from the beginning of the 
#    17th century which is related to practical problems in Chemistry, 
#    Biology and Physics. It consists on maximizing the minimum pairwise 
#    distance between NP points on a sphere in \R^{MDIM}. 
#    This problem may be reduced to a nonconvex nonlinear optimization 
#    problem with a potentially large number of (nonoptimal) points 
#    satisfying optimality conditions. We have, thus, a class of problems 
#    indexed by the parameters MDIM and NP, that provides a suitable 
#    set of test problems for evaluating nonlinear programming codes.
#    After some algebric manipulations, we can formulate this problem as
#                             Minimize z
#                             subject to
#        
#       z \geq <x_i, x_j> for all different pair of indices i, j
#       
#                             ||x_i||^2 = 1    for all i = 1,...,NP
#      The goal is to jtu.find an objective value less than 0.5 (This means
#      that the NP points stored belong to the sphere and every distance
#      between two of them is greater than 1.0).
#      Obs: the starting point is aleatorally chosen although each 
#      variable belongs to [-1.,1.].
#      References:
#      [1] "Validation of an Augmented Lagrangian algorithm with a 
#           Gauss-Newton Hessian approximation using a set of 
#           Hard-Spheres problems", N. Krejic, J. M. Martinez, M. Mello 
#           and E. A. Pilotta, Tech. Report RP 29/98, IMECC-UNICAMP, 
#           Campinas, 1998.
#      [2] "Inexact-Restoration Algorithm for Constrained Optimization",
#           J. M. Martinez and E. A. Pilotta, Tech. Report, IMECC-UNICAMP, 
#           Campinas, 1998.
#      [3]  "Sphere Packings, Lattices and Groups", J. H. Conway and 
#            N. J. C. Sloane, Springer-Verlag, NY, 1988.
#      SIF input: September 29, 1998
# 		 Jose Mario Martinez
#                 Elvio Angel Pilotta
# 
#    classification = "C-CLQR2-RN-V-V"
# 
# **********************************************************************
# 
#    Number of points: NP >= 12
# 
#           Alternative values for the SIF file parameters:
# IE NP                   12            $-PARAMETER
# IE NP                   13            $-PARAMETER
# IE NP                   14            $-PARAMETER
# IE NP                   15            $-PARAMETER
# IE NP                   22            $-PARAMETER
# IE NP                   23            $-PARAMETER
# IE NP                   24            $-PARAMETER
# IE NP                   25            $-PARAMETER
# IE NP                   26            $-PARAMETER
# IE NP                   27            $-PARAMETER
# IE NP	                 37            $-PARAMETER
# IE NP                   38            $-PARAMETER
# IE NP                   39            $-PARAMETER
# IE NP                   40            $-PARAMETER
# IE NP                   41            $-PARAMETER
# IE NP                   42            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'KISSING'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['NP'] = int(25);  #  SIF file default value
        else:
            v_['NP'] = int(args[0])
# IE MDIM                 3             $-PARAMETER
        if nargin<2:
            v_['MDIM'] = int(3);  #  SIF file default value
        else:
            v_['MDIM'] = int(args[1])
# IE MDIM                 4             $-PARAMETER
# IE MDIM                 5             $-PARAMETER
        v_['N-'] = -1+v_['NP']
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['NP'])+1):
            for J in range(int(v_['1']),int(v_['MDIM'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J))
        [iv,ix_,_] = jtu.s2mpj_ii('Z',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Z')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['1']),int(v_['N-'])+1):
            v_['I+'] = 1+I
            for J in range(int(v_['I+']),int(v_['NP'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('IC'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'IC'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Z']])
                valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['NP'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('EC'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'EC'+str(I))
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
        for I in range(int(v_['1']),int(v_['NP'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['EC'+str(I)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for I in range(int(v_['1']),int(v_['NP'])+1):
            for J in range(int(v_['1']),int(v_['MDIM'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(J)]]), -float('Inf'))
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(J)]]), +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['Z'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['Z'], +float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X1,1']]), float(-0.10890604))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X1,2']]), float(0.85395078))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X1,3']]), float(-0.45461680))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X2,1']]), float(0.49883922))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X2,2']]), float(-0.18439316))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X2,3']]), float(-0.04798594))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X3,1']]), float(0.28262888))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X3,2']]), float(-0.48054070))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X3,3']]), float(0.46715332))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,1']]), float(-0.00580106))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,2']]), float(-0.49987584))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X4,3']]), float(-0.44130302))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,1']]), float(0.81712540))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,2']]), float(-0.36874258))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X5,3']]), float(-0.68321896))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X6,1']]), float(0.29642426))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X6,2']]), float(0.82315508))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X6,3']]), float(0.35938150))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X7,1']]), float(0.09215152))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X7,2']]), float(-0.53564686))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X7,3']]), float(0.00191436))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,1']]), float(0.11700318))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,2']]), float(0.96722760))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X8,3']]), float(-0.14916438))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X9,1']]), float(0.01791524))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X9,2']]), float(0.17759446))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X9,3']]), float(-0.61875872))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X10,1']]), float(-0.63833630))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X10,2']]), float(0.80830972))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X10,3']]), float(0.45846734))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X11,1']]), float(0.28446456))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X11,2']]), float(0.45686938))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X11,3']]), float(0.16368980))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X12,1']]), float(0.76557382))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X12,2']]), float(0.16700944))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X12,3']]), float(-0.31647534))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        [it,iet_,_] = jtu.s2mpj_ii( 'eQUA', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N-'])+1):
            v_['I+'] = 1+I
            for J in range(int(v_['I+']),int(v_['NP'])+1):
                for K in range(int(v_['1']),int(v_['MDIM'])+1):
                    ename = 'A'+str(I)+','+str(J)+','+str(K)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
                    ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
                    vname = 'X'+str(I)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NP'])+1):
            for K in range(int(v_['1']),int(v_['MDIM'])+1):
                ename = 'B'+str(I)+','+str(K)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eQUA')
                ielftype = jtu.arrset(ielftype,ie,iet_["eQUA"])
                vname = 'X'+str(I)+','+str(K)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N-'])+1):
            v_['I+'] = 1+I
            for J in range(int(v_['I+']),int(v_['NP'])+1):
                for K in range(int(v_['1']),int(v_['MDIM'])+1):
                    ig = ig_['IC'+str(I)+','+str(J)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (                           jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(J)+','+str(K)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['1']),int(v_['NP'])+1):
            for K in range(int(v_['1']),int(v_['MDIM'])+1):
                ig = ig_['EC'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(K)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# XL SOLUTION             4.47214D-01
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
        self.pbclass   = "C-CLQR2-RN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD(self, nargout,*args):

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

    @staticmethod
    def eQUA(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[0]+EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

