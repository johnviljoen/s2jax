from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LUKVLI3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKVLI3
#    *********
# 
#    Source: Problem 5.3, the chained Powell singular function with 
#    simplified trigonometric exponential constraints, 
#    due to L. Luksan and J. Vlcek,
#    "Sparse and partially separable test problems for 
#    unconstrained and equality constrained optimization",
#    Technical Report 767, Inst. Computer Science, Academy of Sciences
#    of the Czech Republic, 182 07 Prague, Czech Republic, 1999
# 
#    Equality constraints changed to inequalities
# 
#    SIF input: Nick Gould, April 2001
# 
#    classification = "C-COOR2-AY-V-V"
# 
#    some useful parameters, including N, the number of variables.
# 
#           Alternative values for the SIF file parameters:
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER
# IE N                   10000          $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKVLI3'

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
# IE N                   100000         $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['6'] = 6
        v_['N/2'] = int(jnp.fix(v_['N']/v_['2']))
        v_['N/2-1'] = -1+v_['N/2']
        v_['N-1'] = -1+v_['N']
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
        for I in range(int(v_['1']),int(v_['N/2-1'])+1):
            v_['2I'] = 2*I
            v_['2I-1'] = -1+v_['2I']
            v_['2I+1'] = 1+v_['2I']
            v_['2I+2'] = 2+v_['2I']
            [ig,ig_,_] = jtu.s2mpj_ii('OA'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I-1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I']))]])
            valA = jtu.append(valA,float(10.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OB'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I+1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I+2']))]])
            valA = jtu.append(valA,float(-1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OC'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I+1']))]])
            valA = jtu.append(valA,float(-2.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OD'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I-1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['2I+2']))]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'C2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(4.0))
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
        self.gconst = jtu.arrset(self.gconst,ig_['C1'],float(5.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C2'],float(3.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(3.0))
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(-1.0))
        for I in range(int(v_['3']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(0.0))
        for I in range(int(v_['4']),int(v_['N'])+1,int(v_['4'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eCUBE', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eXEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        [it,iet_,_] = jtu.s2mpj_ii( 'eSINASINB', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'C11'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eCUBE')
        ielftype = jtu.arrset(ielftype,ie,iet_["eCUBE"])
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'C12'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSINASINB')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSINASINB"])
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X'+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'C21'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXEXP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXEXP"])
        vname = 'X'+str(int(v_['N-1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X'+str(int(v_['N']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2P',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gL2P',igt_)
        grftp = []
        grftp = jtu.loaset(grftp,it,0,'P')
        [it,igt_,_] = jtu.s2mpj_ii('gL4P',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gL4P',igt_)
        grftp = jtu.loaset(grftp,it,0,'P')
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        self.grpar   = []
        for I in range(int(v_['1']),int(v_['N/2-1'])+1):
            ig = ig_['OA'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2P')
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='P')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(1.0))
            ig = ig_['OB'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2P')
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='P')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(5.0))
            ig = ig_['OC'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL4P')
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='P')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(1.0))
            ig = ig_['OD'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL4P')
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='P')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(10.0))
        ig = ig_['C1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C11'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(3.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C12'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        ig = ig_['C2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C21'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               1.15775E+01
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-AY-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eXEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EXPX = jnp.exp(EV_[0]-EV_[1])
        XEXPX = EV_[0]*EXPX
        f_   = XEXPX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EXPX+XEXPX)
            g_ = jtu.np_like_set(g_, 1, -XEXPX)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0*EXPX+XEXPX)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -EXPX-XEXPX)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), XEXPX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eSINASINB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        SINA = jnp.sin(EV_[0]-EV_[1])
        SINB = jnp.sin(EV_[0]+EV_[1])
        COSA = jnp.cos(EV_[0]-EV_[1])
        COSB = jnp.cos(EV_[0]+EV_[1])
        f_   = SINA*SINB
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, SINA*COSB+COSA*SINB)
            g_ = jtu.np_like_set(g_, 1, SINA*COSB-COSA*SINB)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0*(COSA*COSB-SINA*SINB))
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -2.0*(COSA*COSB+SINA*SINB))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCUBE(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]**3
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 3.0*EV_[0]**2)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 6.0*EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2P(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= self.grpar[igr_][0]*GVAR_*GVAR_
        if nargout>1:
            g_ = 2.0*self.grpar[igr_][0]*GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 2.0*self.grpar[igr_][0]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def gL4P(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= self.grpar[igr_][0]*GVAR_**4
        if nargout>1:
            g_ = 4.0*self.grpar[igr_][0]*GVAR_**3
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 12.0*self.grpar[igr_][0]*GVAR_**2
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

