from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class KISSING2:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem: A second formulation of the KISSING NUMBER PROBLEM
#                                                                    
#    Source: This problem is associated to the family of Hard-Spheres 
#    problem. It belongs to the family of sphere packing problems, a 
#    class of challenging problems dating from the beginning of the 
#    17th century which is related to practical problems in Chemistry, 
#    Biology and Physics. Given a fixed unit sphere at the origin in R^n, 
#    the problem consists of arranging a further m unit spheres so that 
#    sum of the distances to these spheres is as small as possible.
#    This problem may be reduced to a nonconvex nonlinear optimization 
#    problem with a potentially large number of (nonoptimal) points 
#    satisfying optimality conditions. We have, thus, a class of problems 
#    indexed by the parameters m and n, that provides a suitable 
#    set of test problems for evaluating nonlinear programming codes.
#    After some algebric manipulations, we can formulate this problem as
#               m
#     Minimize sum <p_i,p_i> - m n
#              i=1
#     subject to
#        
#      <p_i - p_j, p_i - p_j> >= 4 for all different pair of indices i, j
#      and  
#      <p_i, p_i> >= 4 for all indices i
#   
#      as well as n(n-1)/2 normalisation constraints fixing components.
#      The goal is to jtu.find an objective value equal to 0.
#      [1]  "Sphere Packings, Lattices and Groups", J. H. Conway and 
#            N. J. C. Sloane, Springer-Verlag, NY, 1988.
#    SIF input: Nick Gould, September 2000
# 
#    classification = "C-CQQR2-RN-V-V"
# 
# **********************************************************************
# 
#    Number of points: m
# 
#           Alternative values for the SIF file parameters:
# IE m                   24             $-PARAMETER  number of points
# IE m                   25             $-PARAMETER  number of points
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'KISSING2'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['m'] = int(25);  #  SIF file default value
        else:
            v_['m'] = int(args[0])
# IE m                   100            $-PARAMETER  number of points
# IE n                    4             $-PARAMETER  dimension of sphere
        if nargin<2:
            v_['n'] = int(4);  #  SIF file default value
        else:
            v_['n'] = int(args[1])
# IE n                    8             $-PARAMETER  dimension of sphere
        v_['1'] = 1
        v_['2'] = 2
        v_['n-1'] = v_['n']-v_['1']
        v_['rm'] = float(v_['m'])
        v_['rn'] = float(v_['n'])
        v_['RM+N'] = v_['rm']+v_['rn']
        v_['mn'] = v_['rm']*v_['rn']
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['PI/m'] = v_['PI']/v_['rm']
        v_['2PI/m'] = 2.0*v_['PI/m']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['m'])+1):
            for J in range(int(v_['1']),int(v_['n'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('P'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'P'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['m'])+1):
            for J in range(int(v_['1']),int(v_['m'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(I)+','+str(J))
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
        self.gconst = jtu.arrset(self.gconst,ig_['OBJ'],float(v_['mn']))
        for I in range(int(v_['1']),int(v_['m'])+1):
            for J in range(int(v_['1']),int(v_['m'])+1):
                self.gconst = jtu.arrset(self.gconst,ig_['C'+str(I)+','+str(J)],float(4.0))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for I in range(int(v_['1']),int(v_['m'])+1):
            for J in range(int(v_['1']),int(v_['n'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['P'+str(I)+','+str(J)]]), -float('Inf'))
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['P'+str(I)+','+str(J)]]), +float('Inf'))
        for I in range(int(v_['2']),int(v_['n'])+1):
            for J in range(int(I),int(v_['n'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['P'+str(I)+','+str(J)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['P'+str(I)+','+str(J)]]), 0.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['m'])+1):
            v_['RI'] = float(I)
            v_['2PIi/m'] = v_['2PI/m']*v_['RI']
            v_['cos'] = jnp.cos(v_['2PIi/m'])
            v_['sin'] = jnp.sin(v_['2PIi/m'])
            v_['cos'] = v_['cos']+v_['cos']
            v_['sin'] = v_['sin']+v_['sin']
            if('P'+str(I)+','+str(int(v_['1'])) in ix_):
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['P'+str(I)+','+str(int(v_['1']))]]), float(v_['cos']))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(I)+','+str(int(v_['1']))]),float(v_['cos'])))
            for J in range(int(v_['2']),int(v_['n-1'])+1):
                if('P'+str(I)+','+str(J) in ix_):
                    self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['P'+str(I)+','+str(J)]]), float(v_['sin']))
                else:
                    self.y0  = (                           jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(I)+','+str(J)]),float(v_['sin'])))
        if('P'+str(int(v_['m']))+','+str(int(v_['n'])) in ix_):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['P'+str(int(v_['m']))+','+str(int(v_['n']))]]), float(v_['cos']))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(int(v_['m']))+','+str(int(v_['n']))]),float(v_['cos'])))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD1', iet_)
        elftv = jtu.loaset(elftv,it,0,'P')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD2', iet_)
        elftv = jtu.loaset(elftv,it,0,'Q')
        elftv = jtu.loaset(elftv,it,1,'R')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['m'])+1):
            v_['I-'] = -1+I
            v_['I+'] = 1+I
            for J in range(int(v_['1']),int(v_['I-'])+1):
                for K in range(int(v_['1']),int(v_['n'])+1):
                    ename = 'E'+str(I)+','+str(J)+','+str(K)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'ePROD2')
                    ielftype = jtu.arrset(ielftype,ie,iet_["ePROD2"])
                    vname = 'P'+str(I)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='Q')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'P'+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='R')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            for K in range(int(v_['1']),int(v_['n'])+1):
                ename = 'E'+str(I)+','+str(I)+','+str(K)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'ePROD1')
                ielftype = jtu.arrset(ielftype,ie,iet_["ePROD1"])
                vname = 'P'+str(I)+','+str(K)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='P')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            for J in range(int(v_['I+']),int(v_['m'])+1):
                for K in range(int(v_['1']),int(v_['n'])+1):
                    ename = 'E'+str(I)+','+str(J)+','+str(K)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'ePROD2')
                    ielftype = jtu.arrset(ielftype,ie,iet_["ePROD2"])
                    vname = 'P'+str(I)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='Q')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'P'+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='R')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['m'])+1):
            for K in range(int(v_['1']),int(v_['n'])+1):
                ig = ig_['OBJ']
                posel = len(self.grelt[ig])
                self.grelt  = (                       jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(I)+','+str(K)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            for J in range(int(v_['1']),int(v_['m'])+1):
                for K in range(int(v_['1']),int(v_['n'])+1):
                    ig = ig_['C'+str(I)+','+str(J)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (                           jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)+','+str(K)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# XL SOLUTION             0.00000D+00   $ n=4, m = 24
# XL SOLUTION             6.48030D+00   $ n=4, m = 25 one of many local solutions
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CQQR2-RN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD1(self, nargout,*args):

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

    @staticmethod
    def ePROD2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((1,2))
        IV_ = jnp.zeros(1)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        f_   = IV_[0]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[0]+IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

