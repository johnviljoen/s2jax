import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS68:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS68
#    *********
# 
#    This is a cost optimal inspection plan.
# 
#    Source: problem 68 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: Nick Gould, August 1991.
# 
#    classification = "C-COOR2-MN-4-2"
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS68'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 4
        v_['A'] = 0.0001
        v_['B'] = 1.0
        v_['D'] = 1.0
        v_['NN'] = 24.0
        v_['1'] = 1
        v_['AN'] = v_['A']*v_['NN']
        v_['ROOTN'] = jnp.sqrt(v_['NN'])
        v_['DROOTN'] = v_['D']*v_['ROOTN']
        v_['-DROOTN'] = -v_['DROOTN']
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
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'C1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(1.0e+0))
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'C2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(1.0e+0))
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
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['X1'], 0.0001)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X1'], 100.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X2'], 100.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X3'], 2.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X4'], 2.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        self.y0 = jnp.full((self.m,1),float(1.0))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eRECIP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        [it,iet_,_] = jtu.s2mpj_ii( 'eNASTYEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X3')
        elftv = jtu.loaset(elftv,it,2,'X4')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'B')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePHI', iet_)
        elftv = jtu.loaset(elftv,it,0,'X2')
        elftp = jtu.loaset(elftp,it,0,'P')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'OE1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eRECIP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eRECIP"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'OE2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eNASTYEXP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eNASTYEXP"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='B')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['B']))
        ename = 'C1E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePHI')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePHI"])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.0e+0))
        ename = 'C2E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePHI')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePHI"])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['DROOTN']))
        ename = 'C2E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePHI')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePHI"])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['-DROOTN']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['AN']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0e+0))
        ig = ig_['C1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C1E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-2.0e+0))
        ig = ig_['C2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C2E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0e+0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C2E2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0e+0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# XL SOLUTION            -9.20389D-01
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
        self.pbclass   = "C-COOR2-MN-4-2"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(self):

        import jax.numpy as jnp
        self.efpar = jnp.array([])
        self.efpar = jtu.arrset( self.efpar,0,3.9894228040143270e-01)
        return pbm

    @staticmethod
    def eRECIP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = 1.0e+0/EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -1.0e+0/EV_[0]**2)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e+0/EV_[0]**3)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eNASTYEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        E = jnp.exp(EV_[0])
        R = self.elpar[iel_][0]*(E-1.0e+0)-EV_[1]
        S = E-1.0e+0+EV_[2]
        F1 = -(-(R*EV_[2]/S))/(EV_[0]*EV_[0])
        F2 = -R/(S*EV_[0])
        F3 = -EV_[2]/(S*EV_[0])
        F4 = (EV_[2]*R)/(EV_[0]*S*S)
        F11 = 2.0e+0*(-(R*EV_[2]/S))/(EV_[0]*EV_[0]*EV_[0])
        F12 = E*EV_[2]*((R/S)-self.elpar[iel_][0])/(EV_[0]*S)
        F11 = F11+F12
        F12 = R/(S*EV_[0]*EV_[0])
        F13 = EV_[2]/(S*EV_[0]*EV_[0])
        F14 = R/(S*S*EV_[0])
        F15 = -1.0e+0/(S*EV_[0])
        F16 = -EV_[2]*R/(S*S*EV_[0]*EV_[0])
        F17 = EV_[2]/(S*S*EV_[0])
        F18 = -2.0e+0*EV_[2]*R/(EV_[0]*S*S*S)
        f_   = EV_[2]*R/(S*EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -F1-(self.elpar[iel_][0]*E*F3)-(E*F4))
            g_ = jtu.np_like_set(g_, 1, F3)
            g_ = jtu.np_like_set(g_, 2, -F2-F4)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (-F11-(2.0e+0*self.elpar[iel_][0]*E*F13)-(2.0e+0*E*F16)-                      (2.0e+0*self.elpar[iel_][0]*E*E*F17)-(E*E*F18)))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), F13+(E*F17))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), (-(E*F14)-(self.elpar[iel_][0]*E*F15)-F16-(self.elpar[iel_][0]*E*F17)-                      (E*F18)-F12))
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), F15+F17)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -(2.0e+0*F14)-F18)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePHI(self, nargout,*args):

        import jax.numpy as jnp
        from scipy import special
        x2    = args[0]
        iel_  = args[1]
        mx2pp = -x2+self.elpar[iel_][0]
        E     = jnp.exp(-5.0e-1*mx2pp**2)
        arg   =  7.071067811865475e-1 * jnp.abs( mx2pp )
        if mx2pp >= 0 :
            f_ =  0.5 + 0.5 * special.erf( arg )
        else:
            f_ =  0.5 - 0.5 * special.erf( arg )
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            dim  = len(x2)
            g_    = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -self.efpar[0]*E)
            if nargout>2:
                H_      = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -self.efpar[0]*E*mx2pp)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
