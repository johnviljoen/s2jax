from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class JUNKTURN:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : JUNKTURN
#    *********
# 
#    The spacecraft orientation problem by Junkins and Turner. This is a
#    nonlinear optimal control problem.
# 
#    The problem is not convex.
# 
#    Source:
#    A.I Tyatushkin, A.I. Zholudev and N. M. Erinchek,
#    "The gradient method for solving optimal control problems with phase
#    constraints", 
#    in "System Modelling and Optimization", P. Kall, ed., pp. 456--464,
#    Springer Verlag, Lecture Notes in Control and Information Sciences 180, 1992.
#    This reference itself refers to:
#    I.L. Junkins and I.D. Turner,
#    "Optimal continuous torque attitude maneuvers",
#    AIAA/AAS Astrodynamics Conference, Palo Alto, 1978.
# 
#    SIF input: Ph. Toint, February 1994.
# 
#    classification = "C-CQQR2-MN-V-V"
# 
#    Number of discretized points in [0,100] - 1.
#    The number of variables is    10 * ( N + 1 )
#    The number of constraints is  7 * N
#    N should be large enough to ensure feasibility.
# 
#           Alternative values for the SIF file parameters:
# IE N                   50             $-PARAMETER n =     510, m =    350
# IE N                   100            $-PARAMETER n =    1010, m =    700
# IE N                   500            $-PARAMETER n =    5010, m =   3500
# IE N                   1000           $-PARAMETER n =   10010, m =   7000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'JUNKTURN'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(5);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   10000          $-PARAMETER n =  100010, m =  70000
# IE N                   20000          $-PARAMETER n =  200010, m = 140000
# IE N                   100000         $-PARAMETER n = 1000010, m = 700000
        v_['N-1'] = -1+v_['N']
        v_['RN'] = float(v_['N'])
        v_['H'] = 100.0/v_['RN']
        v_['H/2'] = 0.5*v_['H']
        v_['6H/5'] = 1.2*v_['H']
        v_['SH'] = 1.0909*v_['H']
        v_['S1H'] = -0.08333*v_['H']
        v_['S2H'] = 0.18182*v_['H']
        v_['-H/2'] = -0.5*v_['H']
        v_['-H'] = -1.0*v_['H']
        v_['-H/10'] = -0.1*v_['H']
        v_['H/4'] = 0.25*v_['H']
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['6'] = 6
        v_['7'] = 7
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['7'])+1):
            for T in range(int(v_['0']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(T),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(T))
        for I in range(int(v_['1']),int(v_['3'])+1):
            for T in range(int(v_['0']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I)+','+str(T),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I)+','+str(T))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for T in range(int(v_['1']),int(v_['N'])+1):
            v_['T-1'] = -1+T
            for I in range(int(v_['1']),int(v_['7'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I)+','+str(T),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'C'+str(I)+','+str(T))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(T)]])
                valA = jtu.append(valA,float(-1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(int(v_['T-1']))]])
                valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['5']))+','+str(T),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['5']))+','+str(T))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['1']))+','+str(T)]])
            valA = jtu.append(valA,float(v_['H']))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['6']))+','+str(T),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['6']))+','+str(T))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['2']))+','+str(T)]])
            valA = jtu.append(valA,float(v_['6H/5']))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['7']))+','+str(T),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['7']))+','+str(T))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['3']))+','+str(T)]])
            valA = jtu.append(valA,float(v_['SH']))
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
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['0']))]]), 1.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['0']))]]), 1.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['0']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['3']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['3']))+','+str(int(v_['0']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['4']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['4']))+','+str(int(v_['0']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['5']))+','+str(int(v_['0']))]]), 0.01)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['5']))+','+str(int(v_['0']))]]), 0.01)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['6']))+','+str(int(v_['0']))]]), 0.005)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['6']))+','+str(int(v_['0']))]]), 0.005)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['7']))+','+str(int(v_['0']))]]), 0.001)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['7']))+','+str(int(v_['0']))]]), 0.001)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['N']))]]), 0.43047)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['N']))]]), 0.43047)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['N']))]]), 0.70106)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['N']))]]), 0.70106)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['3']))+','+str(int(v_['N']))]]), 0.0923)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['3']))+','+str(int(v_['N']))]]), 0.0923)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['4']))+','+str(int(v_['N']))]]), 0.56098)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['4']))+','+str(int(v_['N']))]]), 0.56098)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['5']))+','+str(int(v_['N']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['5']))+','+str(int(v_['N']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['6']))+','+str(int(v_['N']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['6']))+','+str(int(v_['N']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['7']))+','+str(int(v_['N']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['7']))+','+str(int(v_['N']))]]), 0.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['0']))]]), float(1.0))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['5']))+','+str(int(v_['0']))]]), float(0.01))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['6']))+','+str(int(v_['0']))]]), float(0.005))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['7']))+','+str(int(v_['0']))]]), float(0.001))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['N']))]]), float(0.43047))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['N']))]]), float(0.70106))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['3']))+','+str(int(v_['N']))]]), float(0.0923))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['4']))+','+str(int(v_['N']))]]), float(0.56098))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for T in range(int(v_['0']),int(v_['N'])+1):
            ename = 'U1S'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'U'+str(int(v_['1']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'U2S'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'U'+str(int(v_['2']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'U3S'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'U'+str(int(v_['3']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for T in range(int(v_['1']),int(v_['N'])+1):
            ename = 'P15'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['1']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P16'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['1']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P17'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['1']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P25'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['2']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P26'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['2']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P27'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['2']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P35'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['3']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P36'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['3']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P37'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['3']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P45'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['4']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P46'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['4']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P47'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['4']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P56'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P57'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['5']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'P67'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'X'+str(int(v_['6']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['7']))+','+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
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
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U1S'+str(int(v_['0']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/4']))
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U1S'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U2S'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U3S'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['U1S'+str(int(v_['N']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/4']))
        for T in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['C'+str(int(v_['1']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P25'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P36'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P47'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            ig = ig_['C'+str(int(v_['2']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P15'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P37'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P46'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            ig = ig_['C'+str(int(v_['3']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P16'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P27'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P45'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            ig = ig_['C'+str(int(v_['4']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P17'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P26'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H/2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P35'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/2']))
            ig = ig_['C'+str(int(v_['5']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P67'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S1H']))
            ig = ig_['C'+str(int(v_['6']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P57'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H/10']))
            ig = ig_['C'+str(int(v_['7']))+','+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P56'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S2H']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN(500)          7.417771100D-5
# LO SOLTN(1000)         1.224842784D-5
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
        self.pbclass   = "C-CQQR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
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
            f_   = f_.item()
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

