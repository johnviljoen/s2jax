from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LUKVLI17:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKVLI17
#    *********
# 
#    Source: Problem 5.17, the chained modified HS52 problem, 
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
#    classification = "C-COQR2-AY-V-V"
# 
#    some useful parameters, including N, the number of variables.
# 
#           Alternative values for the SIF file parameters:
# IE N                   97             $-PARAMETER
# IE N                   997            $-PARAMETER
# IE N                   9997           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKVLI17'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(17);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   99997          $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['N-1'] = -1+v_['N']
        v_['(N-1)/4'] = int(jnp.fix(v_['N-1']/v_['4']))
        v_['NC'] = v_['3']*v_['(N-1)/4']
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
        for I in range(int(v_['1']),int(v_['(N-1)/4'])+1):
            v_['I-1'] = -1+I
            v_['J'] = v_['4']*v_['I-1']
            v_['J+1'] = 1+v_['J']
            v_['J+2'] = 2+v_['J']
            v_['J+3'] = 3+v_['J']
            v_['J+4'] = 4+v_['J']
            v_['J+5'] = 5+v_['J']
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ1'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+1']))]])
            valA = jtu.append(valA,float(4.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+2']))]])
            valA = jtu.append(valA,float(-1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ2'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+2']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+3']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ3'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+4']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ4'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+5']))]])
            valA = jtu.append(valA,float(1.0))
        for K in range(int(v_['1']),int(v_['NC'])+1,int(v_['3'])):
            v_['K+1'] = 1+K
            v_['K+2'] = 2+K
            v_['K+3'] = 3+K
            v_['K+4'] = 4+K
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['K+1']))]])
            valA = jtu.append(valA,float(3.0))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['K+1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['K+1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['K+3']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['K+4']))]])
            valA = jtu.append(valA,float(-2.0))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['K+2'])),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['K+2'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['K+4']))]])
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
        for I in range(int(v_['1']),int(v_['(N-1)/4'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['OBJ2'+str(I)],float(2.0))
            self.gconst = jtu.arrset(self.gconst,ig_['OBJ3'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['OBJ4'+str(I)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(2.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQR', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for K in range(int(v_['1']),int(v_['NC'])+1,int(v_['3'])):
            v_['K+1'] = 1+K
            v_['K+2'] = 2+K
            ename = 'E'+str(K)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
            vname = 'X'+str(K)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(2.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
            ename = 'E'+str(int(v_['K+1']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['K+2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(2.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
            ename = 'E'+str(int(v_['K+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['K+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(2.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gL4',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['(N-1)/4'])+1):
            ig = ig_['OBJ1'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['OBJ2'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL4')
            ig = ig_['OBJ3'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['OBJ4'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        for K in range(int(v_['1']),int(v_['NC'])+1,int(v_['3'])):
            v_['K+1'] = 1+K
            v_['K+2'] = 2+K
            ig = ig_['C'+str(K)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(K)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['C'+str(int(v_['K+1']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['C'+str(int(v_['K+2']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+2']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               7.74126E+01
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COQR2-AY-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQR(self, nargout,*args):

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
            g_ = jtu.np_like_set(g_, 0, 2.0*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
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

    @staticmethod
    def gL4(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_**4
        if nargout>1:
            g_ = 4.0*GVAR_**3
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 12.0*GVAR_**2
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

