from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class RDW2D51F:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : RDW2D51F
#    *********
# 
#    A finite-element approximation to the distributed optimal control problem
# 
#       min 1/2||u-v||_L2^2 + beta ||f||_L2^2
# 
#    subject to - nabla^2 u = f
# 
#    where v is given on and within the boundary of a unit [0,1] box in 
#    2 dimensions, and u = v on its boundary. The discretization uses 
#    quadrilateral elememts. There are simple bounds on the controls f
# 
#    The problem is stated as a quadratic program
# 
#    Source:  example 5.1 in 
#     T. Rees, H. S. Dollar and A. J. Wathen
#     "Optimal solvers for PDE-constrained optimization"
#     SIAM J. Sci. Comp. (to appear) 2009
# 
#    with the control bounds as specified in 
# 
#     M. Stoll and A. J. Wathen
#     "Preconditioning for PDE constrained optimization with 
#      control constraints"
#     OUCL Technical Report 2009
# 
#    SIF input: Nick Gould, May 2009
#               correction by S. Gratton & Ph. Toint, May 2024
# 
#    classification = "C-CQLR2-AN-V-V"
# 
#    Number of nodes in each direction (a power of 2)
# 
#           Alternative values for the SIF file parameters:
# IE N                   2             $-PARAMETER
# IE N                   4             $-PARAMETER
# IE N                   8             $-PARAMETER
# IE N                   16            $-PARAMETER
# IE N                   32            $-PARAMETER
# IE N                   64            $-PARAMETER
# IE N                   128           $-PARAMETER
# IE N                   256           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'RDW2D51F'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(4);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   512           $-PARAMETER
# IE N                   1024          $-PARAMETER
# IE N                   2048          $-PARAMETER
# IE N                   4096          $-PARAMETER
# IE N                   8192          $-PARAMETER
# IE N                   16384         $-PARAMETER
        if nargin<2:
            v_['BETA'] = float(0.01);  #  SIF file default value
        else:
            v_['BETA'] = float(args[1])
        v_['ZERO'] = 0.0
        v_['ONE'] = 1.0
        v_['TWO'] = 2.0
        v_['SIX'] = 6.0
        v_['THIRTYSIX'] = 36.0
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['N-'] = -1+v_['N']
        v_['N-1'] = -1+v_['N']
        v_['N-2'] = -2+v_['N']
        v_['N/2'] = int(jnp.fix(v_['N']/v_['2']))
        v_['N/2+1'] = v_['N/2']+v_['1']
        v_['RN'] = float(v_['N'])
        v_['H'] = v_['ONE']/v_['RN']
        v_['H**2'] = v_['H']*v_['H']
        v_['H**2/36'] = v_['H**2']/v_['THIRTYSIX']
        v_['-H**2/36'] = -1.0*v_['H**2/36']
        v_['2BETA'] = 2.0*v_['BETA']
        v_['2BH**2/36'] = v_['2BETA']*v_['H**2/36']
        v_['1/6'] = v_['ONE']/v_['SIX']
        for I in range(int(v_['0']),int(v_['N/2'])+1):
            v_['RI'] = float(I)
            v_['2RI'] = 2.0*v_['RI']
            v_['2RIH'] = v_['2RI']*v_['H']
            v_['2RIH-1'] = v_['2RIH']-v_['ONE']
            v_['2RIH-1S'] = v_['2RIH-1']*v_['2RIH-1']
            for J in range(int(v_['0']),int(v_['N/2'])+1):
                v_['RJ'] = float(J)
                v_['2RJ'] = 2.0*v_['RJ']
                v_['2RJH'] = v_['2RJ']*v_['H']
                v_['2RJH-1'] = v_['2RJH']-v_['ONE']
                v_['2RJH-1S'] = v_['2RJH-1']*v_['2RJH-1']
                v_['V'] = v_['2RIH-1S']*v_['2RJH-1S']
                v_['V'+str(I)+','+str(J)] = v_['V']
        for I in range(int(v_['N/2+1']),int(v_['N'])+1):
            for J in range(int(v_['N/2+1']),int(v_['N'])+1):
                v_['V'+str(I)+','+str(J)] = v_['ZERO']
        for I in range(int(v_['0']),int(v_['N/2'])+1):
            for J in range(int(v_['N/2+1']),int(v_['N'])+1):
                v_['V'+str(I)+','+str(J)] = v_['ZERO']
        for I in range(int(v_['N/2+1']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(v_['N/2'])+1):
                v_['V'+str(I)+','+str(J)] = v_['ZERO']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('F'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'F'+str(I)+','+str(J))
        for I in range(int(v_['0']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            for J in range(int(v_['1']),int(v_['N-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('L'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'L'+str(I)+','+str(J))
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
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['0']))+','+str(int(v_['0']))]]), (v_['V'+)
             str(int(v_['0']))+','+str(int(v_['0']))])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['0']))+','+str(int(v_['0']))]]), (v_['V'+)
             str(int(v_['0']))+','+str(int(v_['0']))])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['N']))+','+str(int(v_['0']))]]), (v_['V'+)
             str(int(v_['N']))+','+str(int(v_['0']))])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['N']))+','+str(int(v_['0']))]]), (v_['V'+)
             str(int(v_['N']))+','+str(int(v_['0']))])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['0']))+','+str(int(v_['N']))]]), (v_['V'+)
             str(int(v_['0']))+','+str(int(v_['N']))])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['0']))+','+str(int(v_['N']))]]), (v_['V'+)
             str(int(v_['0']))+','+str(int(v_['N']))])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['N']))+','+str(int(v_['N']))]]), (v_['V'+)
             str(int(v_['N']))+','+str(int(v_['N']))])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['N']))+','+str(int(v_['N']))]]), (v_['V'+)
             str(int(v_['N']))+','+str(int(v_['N']))])
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['0']))+','+str(I)]]), (v_['V'+str(int(v_['0']))+)
                 ','+str(I)])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['0']))+','+str(I)]]), (v_['V'+str(int(v_['0']))+)
                 ','+str(I)])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(int(v_['N']))+','+str(I)]]), (v_['V'+str(int(v_['N']))+)
                 ','+str(I)])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(int(v_['N']))+','+str(I)]]), (v_['V'+str(int(v_['N']))+)
                 ','+str(I)])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(int(v_['0']))]]), (v_['V'+str(I)+)
                 ','+str(int(v_['0']))])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(int(v_['0']))]]), (v_['V'+str(I)+)
                 ','+str(int(v_['0']))])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['U'+str(I)+','+str(int(v_['N']))]]), (v_['V'+str(I)+)
                 ','+str(int(v_['N']))])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['U'+str(I)+','+str(int(v_['N']))]]), (v_['V'+str(I)+)
                 ','+str(int(v_['N']))])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['0']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['0']))+','+str(int(v_['0']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['N']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['N']))+','+str(int(v_['0']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['0']))+','+str(int(v_['N']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['0']))+','+str(int(v_['N']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['N']))+','+str(int(v_['N']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['N']))+','+str(int(v_['N']))]]), 0.0)
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['0']))+','+str(I)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['0']))+','+str(I)]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(int(v_['N']))+','+str(I)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(int(v_['N']))+','+str(I)]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(I)+','+str(int(v_['0']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(I)+','+str(int(v_['0']))]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(I)+','+str(int(v_['N']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(I)+','+str(int(v_['N']))]]), 0.0)
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['RI'] = float(I)
            v_['X1'] = v_['RI']*v_['H']
            v_['X1**2'] = v_['X1']*v_['X1']
            v_['-X1**2'] = -1.0*v_['X1**2']
            v_['2-X1'] = v_['TWO']-v_['X1']
            v_['0.1(2-X1)'] = 0.1*v_['2-X1']
            for J in range(int(v_['1']),int(v_['N-1'])+1):
                v_['RJ'] = float(J)
                v_['X2'] = v_['RJ']*v_['H']
                v_['X2**2'] = v_['X2']*v_['X2']
                v_['ARG'] = v_['-X1**2']-v_['X2**2']
                v_['EARG'] = jnp.exp(v_['ARG'])
                v_['UA'] = v_['0.1(2-X1)']*v_['EARG']
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['F'+str(I)+','+str(J)]]), v_['UA'])
            for J in range(int(v_['1']),int(v_['N/2'])+1):
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(I)+','+str(J)]]), 0.6)
            for J in range(int(v_['N/2+1']),int(v_['N-1'])+1):
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['F'+str(I)+','+str(J)]]), 0.9)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['0']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(v_['N'])+1):
                v_['I+J'] = I+J
                v_['RI+J'] = float(v_['I+J'])
                v_['RI+J/N'] = v_['RI+J']/v_['RN']
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eM', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftv = jtu.loaset(elftv,it,3,'U4')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'V1')
        elftp = jtu.loaset(elftp,it,1,'V2')
        elftp = jtu.loaset(elftp,it,2,'V3')
        elftp = jtu.loaset(elftp,it,3,'V4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eM0', iet_)
        elftv = jtu.loaset(elftv,it,0,'F1')
        elftv = jtu.loaset(elftv,it,1,'F2')
        elftv = jtu.loaset(elftv,it,2,'F3')
        elftv = jtu.loaset(elftv,it,3,'F4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftv = jtu.loaset(elftv,it,3,'U4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eB', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftv = jtu.loaset(elftv,it,3,'U4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eC', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftv = jtu.loaset(elftv,it,3,'U4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eD', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftv = jtu.loaset(elftv,it,3,'U4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eP', iet_)
        elftv = jtu.loaset(elftv,it,0,'F1')
        elftv = jtu.loaset(elftv,it,1,'F2')
        elftv = jtu.loaset(elftv,it,2,'F3')
        elftv = jtu.loaset(elftv,it,3,'F4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'F1')
        elftv = jtu.loaset(elftv,it,1,'F2')
        elftv = jtu.loaset(elftv,it,2,'F3')
        elftv = jtu.loaset(elftv,it,3,'F4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eR', iet_)
        elftv = jtu.loaset(elftv,it,0,'F1')
        elftv = jtu.loaset(elftv,it,1,'F2')
        elftv = jtu.loaset(elftv,it,2,'F3')
        elftv = jtu.loaset(elftv,it,3,'F4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eS', iet_)
        elftv = jtu.loaset(elftv,it,0,'F1')
        elftv = jtu.loaset(elftv,it,1,'F2')
        elftv = jtu.loaset(elftv,it,2,'F3')
        elftv = jtu.loaset(elftv,it,3,'F4')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            v_['I+'] = I+v_['1']
            for J in range(int(v_['0']),int(v_['N-1'])+1):
                v_['J+'] = J+v_['1']
                ename = 'E'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eM')
                ielftype = jtu.arrset(ielftype,ie,iet_["eM"])
                vname = 'U'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                posep = jnp.where(elftp[ielftype[ie]]=='V1')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['V'+str(I)+','+str(J)]))
                posep = jnp.where(elftp[ielftype[ie]]=='V2')[0]
                self.elpar  = (
                      jtu.loaset(self.elpar,ie,posep[0],float(v_['V'+str(I)+','+str(int(v_['J+']))])))
                posep = jnp.where(elftp[ielftype[ie]]=='V3')[0]
                self.elpar  = (
                      jtu.loaset(self.elpar,ie,posep[0],float(v_['V'+str(int(v_['I+']))+','+str(J)])))
                posep = jnp.where(elftp[ielftype[ie]]=='V4')[0]
                self.elpar  = (
                      jtu.loaset(self.elpar,ie,posep[0],float(v_['V'+str(int(v_['I+']))+','+str(int(v_['J+']))])))
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            v_['I+'] = I+v_['1']
            for J in range(int(v_['0']),int(v_['N-1'])+1):
                v_['J+'] = J+v_['1']
                ename = 'F'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eM0')
                ielftype = jtu.arrset(ielftype,ie,iet_["eM0"])
                vname = 'F'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            v_['I+'] = I+v_['1']
            for J in range(int(v_['0']),int(v_['N-1'])+1):
                v_['J+'] = J+v_['1']
                ename = 'A'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eA')
                ielftype = jtu.arrset(ielftype,ie,iet_["eA"])
                vname = 'U'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'B'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eB')
                ielftype = jtu.arrset(ielftype,ie,iet_["eB"])
                vname = 'U'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'C'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eC')
                ielftype = jtu.arrset(ielftype,ie,iet_["eC"])
                vname = 'U'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'D'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eD')
                ielftype = jtu.arrset(ielftype,ie,iet_["eD"])
                vname = 'U'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'U'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            v_['I+'] = I+v_['1']
            for J in range(int(v_['0']),int(v_['N-1'])+1):
                v_['J+'] = J+v_['1']
                ename = 'P'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eP')
                ielftype = jtu.arrset(ielftype,ie,iet_["eP"])
                vname = 'F'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'Q'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eQ"])
                vname = 'F'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'R'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eR')
                ielftype = jtu.arrset(ielftype,ie,iet_["eR"])
                vname = 'F'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'S'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eS')
                ielftype = jtu.arrset(ielftype,ie,iet_["eS"])
                vname = 'F'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'F'+str(int(v_['I+']))+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='F4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            for J in range(int(v_['0']),int(v_['N-1'])+1):
                ig = ig_['OBJ']
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['H**2/36']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['F'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['2BH**2/36']))
        for I in range(int(v_['1']),int(v_['N-2'])+1):
            v_['I+'] = I+v_['1']
            for J in range(int(v_['1']),int(v_['N-2'])+1):
                v_['J+'] = J+v_['1']
                ig = ig_['L'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
                ig = ig_['L'+str(I)+','+str(int(v_['J+']))]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
                ig = ig_['L'+str(int(v_['I+']))+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
                ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['J+']))]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['D'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
                ig = ig_['L'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
                ig = ig_['L'+str(I)+','+str(int(v_['J+']))]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Q'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
                ig = ig_['L'+str(int(v_['I+']))+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['R'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
                ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['J+']))]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        for I in range(int(v_['1']),int(v_['N-2'])+1):
            v_['I+'] = I+v_['1']
            ig = ig_['L'+str(I)+','+str(int(v_['N-']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(int(v_['N-']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(I)+','+str(int(v_['1']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(int(v_['0']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['N-']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['C'+str(I)+','+str(int(v_['N-']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['1']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['D'+str(I)+','+str(int(v_['0']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(I)+','+str(int(v_['N-']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['P'+str(I)+','+str(int(v_['N-']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(I)+','+str(int(v_['1']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['Q'+str(I)+','+str(int(v_['0']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['N-']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['R'+str(I)+','+str(int(v_['N-']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(int(v_['I+']))+','+str(int(v_['1']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(int(v_['0']))]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        for J in range(int(v_['1']),int(v_['N-2'])+1):
            v_['J+'] = J+v_['1']
            ig = ig_['L'+str(int(v_['N-']))+','+str(J)]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['A'+str(int(v_['N-']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['J+']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['B'+str(int(v_['N-']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['1']))+','+str(J)]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['C'+str(int(v_['0']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['J+']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['D'+str(int(v_['0']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
            ig = ig_['L'+str(int(v_['N-']))+','+str(J)]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['P'+str(int(v_['N-']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['J+']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['Q'+str(int(v_['N-']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(int(v_['1']))+','+str(J)]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['R'+str(int(v_['0']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
            ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['J+']))]
            posel = len(self.grelt[ig])
            self.grelt  = (
                  jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['0']))+','+str(J)]))
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['N-']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['A'+str(int(v_['N-']))+','+str(int(v_['N-']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
        ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['B'+str(int(v_['N-']))+','+str(int(v_['0']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
        ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['N-']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['C'+str(int(v_['0']))+','+str(int(v_['N-']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
        ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['D'+str(int(v_['0']))+','+str(int(v_['0']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/6']))
        ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['N-']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['P'+str(int(v_['N-']))+','+str(int(v_['N-']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        ig = ig_['L'+str(int(v_['N-']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['Q'+str(int(v_['N-']))+','+str(int(v_['0']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['N-']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['R'+str(int(v_['0']))+','+str(int(v_['N-']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        ig = ig_['L'+str(int(v_['1']))+','+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt  = (
              jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['0']))+','+str(int(v_['0']))]))
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-H**2/36']))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CQLR2-AN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eM(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        UV1 = EV_[0]-self.elpar[iel_][0]
        UV2 = EV_[1]-self.elpar[iel_][1]
        UV3 = EV_[2]-self.elpar[iel_][2]
        UV4 = EV_[3]-self.elpar[iel_][3]
        f_   = (2.0*UV1**2+2.0*UV2**2+2.0*UV3**2+2.0*UV4**2+2.0*UV1*UV2+2.0*UV1*UV3+
             UV1*UV4+UV2*UV3+2.0*UV2*UV4+2.0*UV3*UV4)
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 4.0*UV1+2.0*UV2+2.0*UV3+UV4)
            g_ = jtu.np_like_set(g_, 1, 2.0*UV1+4.0*UV2+UV3+2.0*UV4)
            g_ = jtu.np_like_set(g_, 2, 2.0*UV1+UV2+4.0*UV3+2.0*UV4)
            g_ = jtu.np_like_set(g_, 3, UV1+2.0*UV2+2.0*UV3+4.0*UV4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), 4.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eM0(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_    = (
              2.0*EV_[0]**2+2.0*EV_[1]**2+2.0*EV_[2]**2+2.0*EV_[3]**2+2.0*EV_[0]*EV_[1]+2.0*EV_[0]*EV_[2]+EV_[0]*EV_[3]+EV_[1]*EV_[2]+2.0*EV_[1]*EV_[3]+2.0*EV_[2]*EV_[3])
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 4.0*EV_[0]+2.0*EV_[1]+2.0*EV_[2]+EV_[3])
            g_ = jtu.np_like_set(g_, 1, 2.0*EV_[0]+4.0*EV_[1]+EV_[2]+2.0*EV_[3])
            g_ = jtu.np_like_set(g_, 2, 2.0*EV_[0]+EV_[1]+4.0*EV_[2]+2.0*EV_[3])
            g_ = jtu.np_like_set(g_, 3, EV_[0]+2.0*EV_[1]+2.0*EV_[2]+4.0*EV_[3])
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 4.0)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), 2.0)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), 4.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = 4.0
        C2 = -1.0
        C3 = -1.0
        C4 = -2.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = -1.0
        C2 = 4.0
        C3 = -2.0
        C4 = -1.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eC(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = -1.0
        C2 = -2.0
        C3 = 4.0
        C4 = -1.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = -2.0
        C2 = -1.0
        C3 = -1.0
        C4 = 4.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = 4.0
        C2 = 2.0
        C3 = 2.0
        C4 = 1.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = 2.0
        C2 = 4.0
        C3 = 1.0
        C4 = 2.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = 2.0
        C2 = 1.0
        C3 = 4.0
        C4 = 2.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eS(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C1 = 1.0
        C2 = 2.0
        C3 = 2.0
        C4 = 4.0
        f_   = C1*EV_[0]+C2*EV_[1]+C3*EV_[2]+C4*EV_[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C1)
            g_ = jtu.np_like_set(g_, 1, C2)
            g_ = jtu.np_like_set(g_, 2, C3)
            g_ = jtu.np_like_set(g_, 3, C4)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

