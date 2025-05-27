from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS112:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS112
#    *********
# 
#    This problem is a chemical equilibrium problem involving 3 linear
#    equality constraints.
# 
#    Source: problem 80 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: A.R. Conn, Mar 1990.
# 
#    classification = "C-COLR2-MY-10-3"
# 
#    N is the number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS112'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 10
        v_['1'] = 1
        v_['C1'] = -6.089
        v_['C2'] = -17.164
        v_['C3'] = -34.054
        v_['C4'] = -5.914
        v_['C5'] = -24.721
        v_['C6'] = -14.986
        v_['C7'] = -24.100
        v_['C8'] = -10.708
        v_['C9'] = -26.662
        v_['C10'] = -22.179
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['C'+str(I)]))
        [ig,ig_,_] = jtu.s2mpj_ii('CON1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X6']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X10']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CON2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X6']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X7']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CON3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X7']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X8']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X9']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X10']])
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
        self.gconst = jtu.arrset(self.gconst,ig_['CON1'],float(2.0))
        self.gconst = jtu.arrset(self.gconst,ig_['CON2'],float(1.0))
        self.gconst = jtu.arrset(self.gconst,ig_['CON3'],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),1.0e-6)
        self.xupper = jnp.full((self.n,1),+float('inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.1))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eLOG', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eLOGSUM', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'X1')
        elftv = jtu.loaset(elftv,it,2,'X2')
        elftv = jtu.loaset(elftv,it,3,'X3')
        elftv = jtu.loaset(elftv,it,4,'X4')
        elftv = jtu.loaset(elftv,it,5,'X5')
        elftv = jtu.loaset(elftv,it,6,'X6')
        elftv = jtu.loaset(elftv,it,7,'X7')
        elftv = jtu.loaset(elftv,it,8,'X8')
        elftv = jtu.loaset(elftv,it,9,'X9')
        elftv = jtu.loaset(elftv,it,10,'X10')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'XLOGX'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eLOG')
            ielftype = jtu.arrset(ielftype,ie,iet_["eLOG"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'XLOGS'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eLOGSUM')
            ielftype = jtu.arrset(ielftype,ie,iet_["eLOGSUM"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X5'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X5')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X6'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X6')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X7'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X7')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X8'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X8')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X9'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X9')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X10'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-6),None,float(0.1))
            posev = jnp.where(elftv[ielftype[ie]]=='X10')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XLOGX'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XLOGS'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               -47.707579
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
        self.pbclass   = "C-COLR2-MY-10-3"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eLOG(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        LOGX = jnp.log(EV_[0])
        f_   = EV_[0]*LOGX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, LOGX+1.0)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 1.0/EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eLOGSUM(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,11))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,4]), U_[1,4]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,5]), U_[1,5]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,6]), U_[1,6]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,7]), U_[1,7]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,8]), U_[1,8]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,9]), U_[1,9]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,10]), U_[1,10]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        LOGSUM = jnp.log(IV_[1])
        f_   = -IV_[0]*LOGSUM
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -LOGSUM)
            g_ = jtu.np_like_set(g_, 1, -IV_[0]/IV_[1])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -1.0/IV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), IV_[0]/IV_[1]**2)
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

