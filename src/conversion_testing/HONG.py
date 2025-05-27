from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HONG:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    Source: Se June Hong/Chid Apte
# 
#    SIF input: A.R.Conn, Jan 1991.
# 
#    classification = "C-COLR2-AN-4-1"
# 
#   Problem parameters
# 
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HONG'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('T1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T1')
        [iv,ix_,_] = jtu.s2mpj_ii('T2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T2')
        [iv,ix_,_] = jtu.s2mpj_ii('T3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T3')
        [iv,ix_,_] = jtu.s2mpj_ii('T4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T4')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('SUM1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SUM1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T4']])
        valA = jtu.append(valA,float(1.0))
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
        self.gconst = jtu.arrset(self.gconst,ig_['SUM1'],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),0.0)
        self.xupper = jnp.full((self.n,1),1.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.5))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'P1')
        elftp = jtu.loaset(elftp,it,1,'P2')
        elftp = jtu.loaset(elftp,it,2,'P3')
        elftp = jtu.loaset(elftp,it,3,'P4')
        elftp = jtu.loaset(elftp,it,4,'P5')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'E1'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_['eEXP'])
        vname = 'T1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),float(0.5))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(25.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.92))
        posep = jnp.where(elftp[ielftype[ie]]=='P4')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.08))
        posep = jnp.where(elftp[ielftype[ie]]=='P5')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.38))
        ename = 'E2'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_['eEXP'])
        vname = 'T2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),float(0.5))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(50.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-2.95))
        posep = jnp.where(elftp[ielftype[ie]]=='P4')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(3.95))
        posep = jnp.where(elftp[ielftype[ie]]=='P5')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.11))
        ename = 'E3'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_['eEXP'])
        vname = 'T3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),float(0.5))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(-4.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.66))
        posep = jnp.where(elftp[ielftype[ie]]=='P4')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(1657834.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P5')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.48))
        ename = 'E4'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_['eEXP'])
        vname = 'T4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),float(0.5))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(20000.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P3')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.11))
        posep = jnp.where(elftp[ielftype[ie]]=='P4')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.89))
        posep = jnp.where(elftp[ielftype[ie]]=='P5')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00035))
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
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution unknown
        self.objlower = -4.0
        self.objupper = 300.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COLR2-AN-4-1"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        XTOT = self.elpar[iel_][0]+self.elpar[iel_][1]*EV_[0]
        EP5 = jnp.exp(self.elpar[iel_][4]*XTOT)
        f_   = self.elpar[iel_][2]+self.elpar[iel_][3]*EP5
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][1]*self.elpar[iel_][3]*self.elpar[iel_][4]*EP5)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), ()
                      self.elpar[iel_][1]*self.elpar[iel_][1]*self.elpar[iel_][3]*self.elpar[iel_][4]*self.elpar[iel_][4]*EP5)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

