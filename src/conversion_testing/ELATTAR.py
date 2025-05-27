from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class ELATTAR:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : ELATTAR
#    *********
# 
#    A nonlinear minmax problem in six variables.
# 
#    The problem is nonconvex and has several local minima.
# 
#    Source: 
#    R.A. El-Attar, M. Vidyasagar and S.R.K. Dutta,
#    "An algorithm for l_1-approximation",
#    SINUM 16, pp.70-86, 1979.
# 
#    SIF input: Ph. Toint, Nov 1993.
# 
#    classification = "C-CLOR2-AN-7-102"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'ELATTAR'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['6'] = 6
        v_['51'] = 51
        v_['T'] = 0.0
        for I in range(int(v_['1']),int(v_['51'])+1):
            v_['T'+str(I)] = v_['T']
            v_['T'] = 0.1+v_['T']
            v_['ETI'] = jnp.exp(v_['T'+str(I)])
            v_['Y'+str(I)] = 0.5*v_['ETI']
            v_['-2TI'] = -2.0*v_['T'+str(I)]
            v_['E-2TI'] = jnp.exp(v_['-2TI'])
            v_['Y'+str(I)] = v_['Y'+str(I)]-v_['E-2TI']
            v_['-3TI'] = -3.0*v_['T'+str(I)]
            v_['E-3TI'] = jnp.exp(v_['-3TI'])
            v_['E-3TI/2'] = 0.5*v_['E-3TI']
            v_['Y'+str(I)] = v_['Y'+str(I)]+v_['E-3TI/2']
            v_['-3TI/2'] = 0.5*v_['-3TI']
            v_['E-3TI/2'] = jnp.exp(v_['-3TI/2'])
            v_['7TI'] = 7.0*v_['T'+str(I)]
            v_['S7TI'] = jnp.sin(v_['7TI'])
            v_['TT'] = v_['E-3TI/2']*v_['S7TI']
            v_['TT'] = 1.5*v_['TT']
            v_['Y'+str(I)] = v_['Y'+str(I)]+v_['TT']
            v_['5TI'] = 5.0*v_['T'+str(I)]
            v_['-5TI/2'] = -0.5*v_['5TI']
            v_['E-5TI/2'] = jnp.exp(v_['-5TI/2'])
            v_['S5TI'] = jnp.sin(v_['5TI'])
            v_['TT'] = v_['E-5TI/2']*v_['S5TI']
            v_['Y'+str(I)] = v_['Y'+str(I)]+v_['TT']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['6'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        [iv,ix_,_] = jtu.s2mpj_ii('U',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'U')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U']])
        valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['1']),int(v_['51'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'F'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U']])
            valA = jtu.append(valA,float(-1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('MF'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'MF'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U']])
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
        for I in range(int(v_['1']),int(v_['51'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['F'+str(I)],float(v_['Y'+str(I)]))
            v_['-Y'+str(I)] = -1.0*v_['Y'+str(I)]
            self.gconst = jtu.arrset(self.gconst,ig_['MF'+str(I)],float(v_['-Y'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(-2.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(-2.0)))
        if('X2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(-2.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X2']),float(-2.0)))
        if('X3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(7.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3']),float(7.0)))
        if('X5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X5'], float(-2.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5']),float(-2.0)))
        if('X6' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X6'], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X6']),float(1.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eET1', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftv = jtu.loaset(elftv,it,2,'V3')
        elftv = jtu.loaset(elftv,it,3,'V4')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        [it,iet_,_] = jtu.s2mpj_ii( 'eET2', iet_)
        elftv = jtu.loaset(elftv,it,0,'V5')
        elftv = jtu.loaset(elftv,it,1,'V6')
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['51'])+1):
            ename = 'EL1'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eET1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eET1"])
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
            ename = 'EL2'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eET2')
            ielftype = jtu.arrset(ielftype,ie,iet_["eET2"])
            vname = 'X5'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V5')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X6'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V6')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['51'])+1):
            ig = ig_['F'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EL1'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EL2'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            ig = ig_['MF'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EL1'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EL2'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution       
# LO SOLTN               0.1427066255
# LO SOLTN               74.206179244
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLOR2-AN-7-102"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eET1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A = -EV_[1]*self.elpar[iel_][0]
        B = EV_[2]*self.elpar[iel_][0]+EV_[3]
        EA = jnp.exp(A)
        CB = jnp.cos(B)
        SB = jnp.sin(B)
        EACB = EA*CB
        EASB = EA*SB
        V1EACB = EV_[0]*EACB
        V1EASB = EV_[0]*EASB
        f_   = V1EACB
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EACB)
            g_ = jtu.np_like_set(g_, 1, -self.elpar[iel_][0]*V1EACB)
            g_ = jtu.np_like_set(g_, 2, -self.elpar[iel_][0]*V1EASB)
            g_ = jtu.np_like_set(g_, 3, -V1EASB)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -self.elpar[iel_][0]*EACB)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), -self.elpar[iel_][0]*EASB)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), -EASB)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][0]*self.elpar[iel_][0]*V1EACB)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), self.elpar[iel_][0]*self.elpar[iel_][0]*V1EASB)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), self.elpar[iel_][0]*V1EASB)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -self.elpar[iel_][0]*self.elpar[iel_][0]*V1EACB)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), -self.elpar[iel_][0]*V1EACB)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -V1EACB)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eET2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A = -EV_[1]*self.elpar[iel_][0]
        EA = jnp.exp(A)
        B = EV_[0]*EA
        f_   = B
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EA)
            g_ = jtu.np_like_set(g_, 1, -self.elpar[iel_][0]*B)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -self.elpar[iel_][0]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][0]*self.elpar[iel_][0]*B)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

