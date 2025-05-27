from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class POLAK2:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : POLAK2
#    *********
# 
#    A nonlinear minmax problem in ten variables.
# 
#    Source: 
#    E. Polak, D.H. Mayne and J.E. Higgins,
#    "Superlinearly convergent algorithm for min-max problems"
#    JOTA 69, pp. 407-439, 1991.
# 
#    SIF input: Ph. Toint, Nov 1993.
# 
#    classification = "C-CLOR2-AN-11-2"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'POLAK2'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['10'] = 10
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['10'])+1):
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
        [ig,ig_,_] = jtu.s2mpj_ii('F1',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'F1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F2',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'F2')
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
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.1))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(100.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(100.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eEL', iet_)
        elftv = jtu.loaset(elftv,it,0,'XX1')
        elftv = jtu.loaset(elftv,it,1,'XX2')
        elftv = jtu.loaset(elftv,it,2,'XX3')
        elftv = jtu.loaset(elftv,it,3,'XX4')
        elftv = jtu.loaset(elftv,it,4,'XX5')
        elftv = jtu.loaset(elftv,it,5,'XX6')
        elftv = jtu.loaset(elftv,it,6,'XX7')
        elftv = jtu.loaset(elftv,it,7,'XX8')
        elftv = jtu.loaset(elftv,it,8,'XX9')
        elftv = jtu.loaset(elftv,it,9,'XX10')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'P')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eEL')
        ielftype = jtu.arrset(ielftype,ie,iet_["eEL"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX5')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX6')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX7')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX8')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX9')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX10')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.0))
        ename = 'E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eEL')
        ielftype = jtu.arrset(ielftype,ie,iet_["eEL"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX5')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX6')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX7')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX8')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX9')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1))
        posev = jnp.where(elftv[ielftype[ie]]=='XX10')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-2.0))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['F1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['F2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               54.598146
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
        self.pbclass   = "C-CLOR2-AN-11-2"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eEL(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A = 1.0e-8*EV_[0]*EV_[0]+(EV_[1]+self.elpar[iel_][0])**2
        A = A+EV_[2]*EV_[2]+4.0*EV_[3]*EV_[3]
        A = A+EV_[4]*EV_[4]+EV_[5]*EV_[5]+EV_[6]*EV_[6]
        A = A+EV_[7]*EV_[7]+EV_[8]*EV_[8]+EV_[9]*EV_[9]
        EA = jnp.exp(A)
        f_   = EA
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 2.0e-8*EV_[0]*EA)
            g_ = jtu.np_like_set(g_, 1, 2.0*(EV_[1]+self.elpar[iel_][0])*EA)
            g_ = jtu.np_like_set(g_, 2, 2.0*EV_[2]*EA)
            g_ = jtu.np_like_set(g_, 3, 8.0*EV_[3]*EA)
            g_ = jtu.np_like_set(g_, 4, 2.0*EV_[4]*EA)
            g_ = jtu.np_like_set(g_, 5, 2.0*EV_[5]*EA)
            g_ = jtu.np_like_set(g_, 6, 2.0*EV_[6]*EA)
            g_ = jtu.np_like_set(g_, 7, 2.0*EV_[7]*EA)
            g_ = jtu.np_like_set(g_, 8, 2.0*EV_[8]*EA)
            g_ = jtu.np_like_set(g_, 9, 2.0*EV_[9]*EA)
            if nargout>2:
                H_ = jnp.zeros((10,10))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e-8*EA*(1.0+2.0e-8*EV_[0]**2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 4.0e-8*EV_[0]*(EV_[1]+self.elpar[iel_][0])*EA)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 4.0e-8*EV_[0]*EV_[2]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), 1.6e-7*EV_[0]*EV_[3]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), 4.0e-8*EV_[0]*EV_[4]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([0,5]), 4.0e-8*EV_[0]*EV_[5]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([5,0]), H_[0,5])
                H_ = jtu.np_like_set(H_, jnp.array([0,6]), 4.0e-8*EV_[0]*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,0]), H_[0,6])
                H_ = jtu.np_like_set(H_, jnp.array([0,7]), 4.0e-8*EV_[0]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,0]), H_[0,7])
                H_ = jtu.np_like_set(H_, jnp.array([0,8]), 4.0e-8*EV_[0]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,0]), H_[0,8])
                H_ = jtu.np_like_set(H_, jnp.array([0,9]), 4.0e-8*EV_[0]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,0]), H_[0,9])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 2.0*EA*(1.0+2.0*(EV_[1]+self.elpar[iel_][0])**2))
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[2]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), 16.0*(EV_[1]+self.elpar[iel_][0])*EV_[3]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[4]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,5]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[5]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([5,1]), H_[1,5])
                H_ = jtu.np_like_set(H_, jnp.array([1,6]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,1]), H_[1,6])
                H_ = jtu.np_like_set(H_, jnp.array([1,7]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,1]), H_[1,7])
                H_ = jtu.np_like_set(H_, jnp.array([1,8]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,1]), H_[1,8])
                H_ = jtu.np_like_set(H_, jnp.array([1,9]), 4.0*(EV_[1]+self.elpar[iel_][0])*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,1]), H_[1,9])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 2.0*EA*(1.0+2.0*EV_[2]*EV_[2]))
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), 16.0*EV_[2]*EV_[3]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), 4.0*EV_[2]*EV_[4]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,5]), 4.0*EV_[2]*EV_[5]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([5,2]), H_[2,5])
                H_ = jtu.np_like_set(H_, jnp.array([2,6]), 4.0*EV_[2]*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,2]), H_[2,6])
                H_ = jtu.np_like_set(H_, jnp.array([2,7]), 4.0*EV_[2]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,2]), H_[2,7])
                H_ = jtu.np_like_set(H_, jnp.array([2,8]), 4.0*EV_[2]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,2]), H_[2,8])
                H_ = jtu.np_like_set(H_, jnp.array([2,9]), 4.0*EV_[2]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,2]), H_[2,9])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), 8.0*EA*(1.0+8.0*EV_[3]*EV_[3]))
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), 16.0*EV_[3]*EV_[4]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,5]), 16.0*EV_[3]*EV_[5]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([5,3]), H_[3,5])
                H_ = jtu.np_like_set(H_, jnp.array([3,6]), 16.0*EV_[3]*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,3]), H_[3,6])
                H_ = jtu.np_like_set(H_, jnp.array([3,7]), 16.0*EV_[3]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,3]), H_[3,7])
                H_ = jtu.np_like_set(H_, jnp.array([3,8]), 16.0*EV_[3]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,3]), H_[3,8])
                H_ = jtu.np_like_set(H_, jnp.array([3,9]), 16.0*EV_[3]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,3]), H_[3,9])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), 2.0*EA*(1.0+2.0*EV_[4]*EV_[4]))
                H_ = jtu.np_like_set(H_, jnp.array([4,5]), 4.0*EV_[4]*EV_[5]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([5,4]), H_[4,5])
                H_ = jtu.np_like_set(H_, jnp.array([4,6]), 4.0*EV_[4]*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,4]), H_[4,6])
                H_ = jtu.np_like_set(H_, jnp.array([4,7]), 4.0*EV_[4]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,4]), H_[4,7])
                H_ = jtu.np_like_set(H_, jnp.array([4,8]), 4.0*EV_[4]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,4]), H_[4,8])
                H_ = jtu.np_like_set(H_, jnp.array([4,9]), 4.0*EV_[4]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,4]), H_[4,9])
                H_ = jtu.np_like_set(H_, jnp.array([5,5]), 2.0*EA*(1.0+2.0*EV_[5]*EV_[5]))
                H_ = jtu.np_like_set(H_, jnp.array([5,6]), 4.0*EV_[5]*EV_[6]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([6,5]), H_[5,6])
                H_ = jtu.np_like_set(H_, jnp.array([5,7]), 4.0*EV_[5]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,5]), H_[5,7])
                H_ = jtu.np_like_set(H_, jnp.array([5,8]), 4.0*EV_[5]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,5]), H_[5,8])
                H_ = jtu.np_like_set(H_, jnp.array([5,9]), 4.0*EV_[5]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,5]), H_[5,9])
                H_ = jtu.np_like_set(H_, jnp.array([6,6]), 2.0*EA*(1.0+2.0*EV_[6]*EV_[6]))
                H_ = jtu.np_like_set(H_, jnp.array([6,7]), 4.0*EV_[6]*EV_[7]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([7,6]), H_[6,7])
                H_ = jtu.np_like_set(H_, jnp.array([6,8]), 4.0*EV_[6]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,6]), H_[6,8])
                H_ = jtu.np_like_set(H_, jnp.array([6,9]), 4.0*EV_[6]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,6]), H_[6,9])
                H_ = jtu.np_like_set(H_, jnp.array([7,7]), 2.0*EA*(1.0+2.0*EV_[7]*EV_[7]))
                H_ = jtu.np_like_set(H_, jnp.array([7,8]), 4.0*EV_[7]*EV_[8]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([8,7]), H_[7,8])
                H_ = jtu.np_like_set(H_, jnp.array([7,9]), 4.0*EV_[7]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,7]), H_[7,9])
                H_ = jtu.np_like_set(H_, jnp.array([8,8]), 2.0*EA*(1.0+2.0*EV_[8]*EV_[8]))
                H_ = jtu.np_like_set(H_, jnp.array([8,9]), 4.0*EV_[8]*EV_[9]*EA)
                H_ = jtu.np_like_set(H_, jnp.array([9,8]), H_[8,9])
                H_ = jtu.np_like_set(H_, jnp.array([9,9]), 2.0*EA*(1.0+2.0*EV_[9]*EV_[9]))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

