from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LUKSAN17LS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKSAN17LS
#    *********
# 
#    Problem 17 (sparse trigonometric) in the paper
# 
#      L. Luksan
#      Hybrid methods in large sparse nonlinear least squares
#      J. Optimization Theory & Applications 89(3) 575-595 (1996)
# 
#    SIF input: Nick Gould, June 2017.
# 
#    least-squares version
# 
#    classification = "C-CSUR2-AN-V-0"
# 
#   seed for dimensions
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKSAN17LS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['S'] = 49
        v_['N'] = 2*v_['S']
        v_['N'] = 2+v_['N']
        v_['M'] = 4*v_['S']
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['ONE'] = 1.0
        v_['Y1'] = 30.6
        v_['Y2'] = 72.2
        v_['Y3'] = 124.4
        v_['Y4'] = 187.4
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
        for I in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        v_['K'] = 1
        for J in range(int(v_['1']),int(v_['S'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K']))],float(v_['Y1']))
            v_['K'] = 1+v_['K']
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K']))],float(v_['Y2']))
            v_['K'] = 1+v_['K']
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K']))],float(v_['Y3']))
            v_['K'] = 1+v_['K']
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K']))],float(v_['Y4']))
            v_['K'] = 1+v_['K']
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(-0.8))
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(1.2))
        for I in range(int(v_['3']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(-1.2))
        for I in range(int(v_['4']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(0.8))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eACOSX', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'A')
        [it,iet_,_] = jtu.s2mpj_ii( 'eASINX', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = jtu.loaset(elftp,it,0,'A')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for Q in range(int(v_['1']),int(v_['4'])+1):
            v_['RQ'] = float(Q)
            v_['RQ2'] = v_['RQ']*v_['RQ']
            v_['K'] = 1
            v_['I'] = 0
            for J in range(int(v_['1']),int(v_['S'])+1):
                v_['I+Q'] = v_['I']+Q
                for L in range(int(v_['1']),int(v_['4'])+1):
                    v_['RL'] = float(L)
                    v_['RL2'] = v_['RL']*v_['RL']
                    v_['A'] = v_['RL']*v_['RQ2']
                    v_['A'] = -1.0*v_['A']
                    ename = 'S'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eASINX')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eASINX"])
                    ename = 'S'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    vname = 'X'+str(int(v_['I+Q']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    ename = 'S'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
                    self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['A']))
                    v_['A'] = v_['RL2']*v_['RQ']
                    ename = 'C'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eACOSX')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eACOSX"])
                    ename = 'C'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    vname = 'X'+str(int(v_['I+Q']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    ename = 'C'+str(int(v_['K']))+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
                    self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['A']))
                    v_['K'] = 1+v_['K']
                v_['I'] = 2+v_['I']
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
        for K in range(int(v_['1']),int(v_['M'])+1):
            for Q in range(int(v_['1']),int(v_['4'])+1):
                ig = ig_['E'+str(K)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(K)+','+str(Q)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
                posel = posel+1
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C'+str(K)+','+str(Q)])
                self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eASINX(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ASINX = self.elpar[iel_][0]*jnp.sin(EV_[0])
        f_   = ASINX
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*jnp.cos(EV_[0]))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -ASINX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eACOSX(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ACOSX = self.elpar[iel_][0]*jnp.cos(EV_[0])
        f_   = ACOSX
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -self.elpar[iel_][0]*jnp.sin(EV_[0]))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -ACOSX)
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

