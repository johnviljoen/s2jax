import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class CBRATU3D:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : CBRATU3D
#    *********
# 
#    The complex 3D Bratu problem on the unit cube, using finite
#    differences.
# 
#    Source: Problem 3 in
#    J.J. More',
#    "A collection of nonlinear model problems"
#    Proceedings of the AMS-SIAM Summer seminar on the Computational
#    Solution of Nonlinear Systems of Equations, Colorado, 1988.
#    Argonne National Laboratory MCS-P60-0289, 1989.
# 
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CNOR2-MN-V-V"
# 
#    P is the number of points in one side of the unit cube
#    There are 2*P**3 variables
# 
#           Alternative values for the SIF file parameters:
# IE P                   3              $-PARAMETER n = 54   original value
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CBRATU3D'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['P'] = int(3);  #  SIF file default value
        else:
            v_['P'] = int(args[0])
# IE P                   4              $-PARAMETER n = 128
# IE P                   7              $-PARAMETER n = 686
# IE P                   10             $-PARAMETER n = 2000
# IE P                   12             $-PARAMETER n = 3456
        if nargin<2:
            v_['LAMBDA'] = float(6.80812);  #  SIF file default value
        else:
            v_['LAMBDA'] = float(args[1])
        v_['1'] = 1
        v_['2'] = 2
        v_['1.0'] = 1.0
        v_['P-1'] = -1+v_['P']
        v_['RP-1'] = float(v_['P-1'])
        v_['H'] = v_['1.0']/v_['RP-1']
        v_['H2'] = v_['H']*v_['H']
        v_['C'] = v_['H2']*v_['LAMBDA']
        v_['-C'] = -1.0*v_['C']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for J in range(int(v_['1']),int(v_['P'])+1):
            for I in range(int(v_['1']),int(v_['P'])+1):
                for K in range(int(v_['1']),int(v_['P'])+1):
                    [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I)+','+str(J)+','+str(K),ix_)
                    self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I)+','+str(J)+','+str(K))
                    [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J)+','+str(K),ix_)
                    self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J)+','+str(K))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['2']),int(v_['P-1'])+1):
            v_['R'] = 1+I
            v_['S'] = -1+I
            for J in range(int(v_['2']),int(v_['P-1'])+1):
                v_['V'] = 1+J
                v_['W'] = -1+J
                for K in range(int(v_['2']),int(v_['P-1'])+1):
                    v_['Y'] = 1+K
                    v_['Z'] = -1+K
                    [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I)+','+str(J)+','+str(K),ig_)
                    gtype = jtu.arrset(gtype,ig,'==')
                    cnames = jtu.arrset(cnames,ig,'G'+str(I)+','+str(J)+','+str(K))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(I)+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(6.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(int(v_['R']))+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(int(v_['S']))+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(I)+','+str(int(v_['V']))+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(I)+','+str(int(v_['W']))+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(I)+','+str(J)+','+str(int(v_['Y']))]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['U'+str(I)+','+str(J)+','+str(int(v_['Z']))]])
                    valA = jtu.append(valA,float(-1.0))
                    [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I)+','+str(J)+','+str(K),ig_)
                    gtype = jtu.arrset(gtype,ig,'==')
                    cnames = jtu.arrset(cnames,ig,'F'+str(I)+','+str(J)+','+str(K))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(6.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(int(v_['R']))+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(int(v_['S']))+','+str(J)+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(int(v_['V']))+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(int(v_['W']))+','+str(K)]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)+','+str(int(v_['Y']))]])
                    valA = jtu.append(valA,float(-1.0))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)+','+str(int(v_['Z']))]])
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
        for J in range(int(v_['1']),int(v_['P'])+1):
            for K in range(int(v_['1']),int(v_['P'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['1']))+','+str(J)+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['1']))+','+str(J)+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['P']))+','+str(J)+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['P']))+','+str(J)+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['P']))+','+str(J)+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['P']))+','+str(J)+','+str(K)]]), 0.0)
        for I in range(int(v_['2']),int(v_['P-1'])+1):
            for K in range(int(v_['1']),int(v_['P'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(int(v_['P']))+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(int(v_['P']))+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(int(v_['1']))+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(int(v_['1']))+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['P']))+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['P']))+','+str(K)]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))+','+str(K)]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))+','+str(K)]]), 0.0)
        for I in range(int(v_['2']),int(v_['P-1'])+1):
            for J in range(int(v_['2']),int(v_['P-1'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(J)+','+str(int(v_['1']))]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(J)+','+str(int(v_['1']))]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(J)+','+str(int(v_['P']))]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(J)+','+str(int(v_['P']))]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(J)+','+str(int(v_['1']))]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(J)+','+str(int(v_['1']))]]), 0.0)
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(J)+','+str(int(v_['P']))]]), 0.0)
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(J)+','+str(int(v_['P']))]]), 0.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eRPART', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCPART', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['2']),int(v_['P-1'])+1):
            for J in range(int(v_['2']),int(v_['P-1'])+1):
                for K in range(int(v_['2']),int(v_['P-1'])+1):
                    ename = 'A'+str(I)+','+str(J)+','+str(K)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eRPART')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eRPART"])
                    vname = 'U'+str(I)+','+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
                    posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(I)+','+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
                    posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    ename = 'B'+str(I)+','+str(J)+','+str(K)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eCPART')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eCPART"])
                    vname = 'U'+str(I)+','+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
                    posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(I)+','+str(J)+','+str(K)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
                    posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['P-1'])+1):
            for J in range(int(v_['2']),int(v_['P-1'])+1):
                for K in range(int(v_['2']),int(v_['P-1'])+1):
                    ig = ig_['G'+str(I)+','+str(J)+','+str(K)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (                           jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(J)+','+str(K)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-C']))
                    ig = ig_['F'+str(I)+','+str(J)+','+str(K)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (                           jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(J)+','+str(K)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-C']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
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
        self.pbclass   = "C-CNOR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eRPART(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EXPU = jnp.exp(EV_[0])
        EXPUC = EXPU*jnp.cos(EV_[1])
        EXPUS = EXPU*jnp.sin(EV_[1])
        f_   = EXPUC
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EXPUC)
            g_ = jtu.np_like_set(g_, 1, -EXPUS)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), EXPUC)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -EXPUS)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -EXPUC)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCPART(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EXPU = jnp.exp(EV_[0])
        EXPUC = EXPU*jnp.cos(EV_[1])
        EXPUS = EXPU*jnp.sin(EV_[1])
        f_   = EXPUS
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EXPUS)
            g_ = jtu.np_like_set(g_, 1, EXPUC)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), EXPUS)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), EXPUC)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -EXPUS)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

