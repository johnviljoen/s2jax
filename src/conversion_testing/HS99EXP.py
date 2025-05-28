import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS99EXP:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS99EXP
#    *********
# 
#    Source: an expanded form of problem 99 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: Ph. Toint, April 1991.
# 
#    classification = "C-COOR2-AN-31-21"
# 
#    Problem data
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS99EXP'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['T1'] = 0.0
        v_['T2'] = 25.0
        v_['T3'] = 50.0
        v_['T4'] = 100.0
        v_['T5'] = 150.0
        v_['T6'] = 200.0
        v_['T7'] = 290.0
        v_['T8'] = 380.0
        v_['A1'] = 0.0
        v_['A2'] = 50.0
        v_['A3'] = 50.0
        v_['A4'] = 75.0
        v_['A5'] = 75.0
        v_['A6'] = 75.0
        v_['A7'] = 100.0
        v_['A8'] = 100.0
        v_['B'] = 32.0
        v_['1'] = 1
        v_['2'] = 2
        v_['7'] = 7
        v_['8'] = 8
        for I in range(int(v_['2']),int(v_['8'])+1):
            v_['I-1'] = -1+I
            v_['DT'+str(I)] = v_['T'+str(I)]-v_['T'+str(int(v_['I-1']))]
            v_['DTISQ'] = v_['DT'+str(I)]*v_['DT'+str(I)]
            v_['DT'+str(I)] = 0.5*v_['DTISQ']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['7'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('R'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'R'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('Q'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Q'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('S'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'S'+str(I))
        [iv,ix_,_] = jtu.s2mpj_ii('R'+str(int(v_['8'])),ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'R'+str(int(v_['8'])))
        [iv,ix_,_] = jtu.s2mpj_ii('Q'+str(int(v_['8'])),ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q'+str(int(v_['8'])))
        [iv,ix_,_] = jtu.s2mpj_ii('S'+str(int(v_['8'])),ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'S'+str(int(v_['8'])))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['2']),int(v_['8'])+1):
            v_['I-1'] = -1+I
            [ig,ig_,_] = jtu.s2mpj_ii('R'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'R'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('Q'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'Q'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['S'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['DT'+str(I)]))
            [ig,ig_,_] = jtu.s2mpj_ii('S'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'S'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['S'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['S'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['8']))]])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(-1.0))
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
        for I in range(int(v_['2']),int(v_['7'])+1):
            v_['RHS'] = v_['DT'+str(I)]*v_['B']
            self.gconst = jtu.arrset(self.gconst,ig_['Q'+str(I)],float(v_['RHS']))
            v_['RHS'] = v_['DT'+str(I)]*v_['B']
            self.gconst = jtu.arrset(self.gconst,ig_['S'+str(I)],float(v_['RHS']))
        self.gconst = jtu.arrset(self.gconst,ig_['Q'+str(int(v_['8']))],float(100000.0))
        self.gconst = jtu.arrset(self.gconst,ig_['S'+str(int(v_['8']))],float(1000.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['R'+str(int(v_['1']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['R'+str(int(v_['1']))], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['Q'+str(int(v_['1']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Q'+str(int(v_['1']))], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['S'+str(int(v_['1']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['S'+str(int(v_['1']))], 0.0)
        for I in range(int(v_['1']),int(v_['7'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(I)], 0.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(I)], 1.58)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['7'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(0.5))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSN', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCS', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['7'])+1):
            ename = 'SNX'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSN')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSN"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'CSX'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCS')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCS"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
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
        ig = ig_['OBJ']
        self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        for I in range(int(v_['2']),int(v_['8'])+1):
            v_['I-1'] = -1+I
            v_['W'] = v_['A'+str(I)]*v_['DT'+str(I)]
            ig = ig_['R'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['CSX'+str(int(v_['I-1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['W']))
            ig = ig_['S'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SNX'+str(int(v_['I-1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['W']))
            v_['W'] = v_['A'+str(I)]*v_['DT'+str(I)]
            ig = ig_['Q'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SNX'+str(int(v_['I-1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['W']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               -831079892.0
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
        self.pbclass   = "C-COOR2-AN-31-21"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSN(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        SNX = jnp.sin(EV_[0])
        f_   = SNX
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, jnp.cos(EV_[0]))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -SNX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCS(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        CSX = jnp.cos(EV_[0])
        f_   = CSX
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -jnp.sin(EV_[0]))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -CSX)
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

