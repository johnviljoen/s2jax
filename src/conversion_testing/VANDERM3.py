from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class VANDERM3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A nonlinear equation problem, subject to monotonicity constraints.
#    The Jacobian is a dense Vandermonde matrix.
# 
#    Problems VANDERM1, VANDERM2, VANDERM3 and VANDERM4 differ by the rhs
#    of the equation.  They are increasingly degenerate.
# 
#    The problem is non-convex.
# 
#    Source:
#    A. Neumaier, private communication, 1991.
# 
#    SIF input: Ph. L. Toint, May 1993.
#               minor correction by Ph. Shott, Jan 1995.
# 
#    classification = "C-CNOR2-AN-V-V"
# 
#    Size of the system (N must be even)
# 
#           Alternative values for the SIF file parameters:
# IE N                   2              $-PARAMETER
# IE N                   3              $-PARAMETER
# IE N                   4              $-PARAMETER
# IE N                   5              $-PARAMETER
# IE N                   10             $-PARAMETER     original value
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'VANDERM3'

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
        v_['1'] = 1
        v_['2'] = 2
        v_['RN'] = float(v_['N'])
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['2'])):
            v_['I-1'] = -1+I
            v_['RI'] = float(I)
            v_['TMP'] = v_['RI']/v_['RN']
            v_['AL'+str(int(v_['I-1']))] = v_['TMP']
            v_['AL'+str(I)] = v_['TMP']
        v_['A'+str(int(v_['N']))] = 1.0
        v_['A'+str(int(v_['1']))] = 0.0
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['A'+str(int(v_['1']))] = v_['AL'+str(I)]+v_['A'+str(int(v_['1']))]
        for K in range(int(v_['2']),int(v_['N'])+1):
            v_['RK'] = float(K)
            v_['A'+str(K)] = 0.0
            for I in range(int(v_['1']),int(v_['N'])+1):
                v_['LOGAL'] = jnp.log(v_['AL'+str(I)])
                v_['KLOGAL'] = v_['LOGAL']*v_['RK']
                v_['ALK'] = jnp.exp(v_['KLOGAL'])
                v_['A'+str(K)] = v_['A'+str(K)]+v_['ALK']
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
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'E'+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(1.0))
        for K in range(int(v_['2']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'E'+str(K))
        for I in range(int(v_['2']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            [ig,ig_,_] = jtu.s2mpj_ii('M'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'M'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I-1']))]])
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
        for K in range(int(v_['1']),int(v_['N'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(K)],float(v_['A'+str(K)]))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['TMP'] = v_['RI-1']/v_['RN']
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['TMP']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePOWER', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'PWR')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for K in range(int(v_['2']),int(v_['N'])+1):
            v_['RK'] = float(K)
            for I in range(int(v_['1']),int(v_['N'])+1):
                ename = 'E'+str(I)+','+str(K)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
                ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
                vname = 'X'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RK']))
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
        ig = ig_['E'+str(int(v_['1']))]
        self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        for K in range(int(v_['2']),int(v_['N'])+1):
            ig = ig_['E'+str(K)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            for I in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['E'+str(K)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(K)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SOLUTION            0.0
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
        self.pbclass   = "C-CNOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePOWER(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]**self.elpar[iel_][0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*EV_[0]**(self.elpar[iel_][0]-1.0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), ()
                      self.elpar[iel_][0]*(self.elpar[iel_][0]-1.0)*EV_[0]**(self.elpar[iel_][0]-2.0))
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

