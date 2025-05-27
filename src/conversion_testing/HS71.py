from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS71:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS71
#    *********
# 
#    Source: problem 71 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: B Baudson, Apr 1989.
# 
#    classification = "C-COOR2-AY-4-2"
# 
#    Number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS71'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 4
        v_['1'] = 1
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
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C1')
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'C2')
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
        self.gconst = jtu.arrset(self.gconst,ig_['C1'],float(25.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C2'],float(40.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),1.0)
        self.xupper = jnp.full((self.n,1),5.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(1.0)))
        if('X2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(5.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X2']),float(5.0)))
        if('X3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(5.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3']),float(5.0)))
        if('X4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X4']),float(1.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eLP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftv = jtu.loaset(elftv,it,2,'V1')
        elftv = jtu.loaset(elftv,it,3,'V2')
        elftv = jtu.loaset(elftv,it,4,'V3')
        [it,iet_,_] = jtu.s2mpj_ii( 'en4PRD', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftv = jtu.loaset(elftv,it,2,'V3')
        elftv = jtu.loaset(elftv,it,3,'V4')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eLP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eLP"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'en4PRD')
        ielftype = jtu.arrset(ielftype,ie,iet_["en4PRD"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E4'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E5'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E6'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(5.0),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
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
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['C1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['C2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E5'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SOLTN               17.0140173
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-AY-4-2"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eLP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((3,5))
        IV_ = jnp.zeros(3)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([2,2]), U_[2,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([2,3]), U_[2,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([2,4]), U_[2,4]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 2, U_[2:3,:].dot(EV_))
        f_   = IV_[0]*IV_[1]*IV_[2]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[1]*IV_[2])
            g_ = jtu.np_like_set(g_, 1, IV_[0]*IV_[2])
            g_ = jtu.np_like_set(g_, 2, IV_[0]*IV_[1])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), IV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), IV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), IV_[0])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
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
    def en4PRD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]*EV_[2]*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1]*EV_[2]*EV_[3])
            g_ = jtu.np_like_set(g_, 1, EV_[0]*EV_[2]*EV_[3])
            g_ = jtu.np_like_set(g_, 2, EV_[0]*EV_[1]*EV_[3])
            g_ = jtu.np_like_set(g_, 3, EV_[0]*EV_[1]*EV_[2])
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), EV_[2]*EV_[3])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), EV_[1]*EV_[3])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), EV_[1]*EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), EV_[0]*EV_[3])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), EV_[0]*EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), EV_[0]*EV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

