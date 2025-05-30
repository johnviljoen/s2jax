import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class JNLBRNGB:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : JNLBRNGB
#    *********
# 
#    The quadratic journal bearing problem (with excentricity = 0.5)
# 
#    Source:
#    J. More' and G. Toraldo,
#    "On the Solution of Large Quadratic-Programming Problems with Bound
#    Constraints", 
#    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.
# 
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CQBR2-AY-V-0"
# 
#    The rectangle is discretized into (pt-1)(py-1) little rectangles. The
#    heights of the considered surface above the corners of these little
#    rectangles are the problem variables,  There are px*py of them.
# 
#    PT is the number of points along the T (\theta) side of the rectangle
#    PY is the number of points along the Y side of the rectangle
# 
#           Alternative values for the SIF file parameters:
# IE PT                  4              $-PARAMETER  n=16
# IE PY                  4              $-PARAMETER
# 
# IE PT                  10             $-PARAMETER  n=100
# IE PY                  10             $-PARAMETER
# 
# IE PT                  23             $-PARAMETER  n=529
# IE PY                  23             $-PARAMETER
# 
# IE PT                  32             $-PARAMETER  n=1024
# IE PY                  32             $-PARAMETER
# 
# IE PT                  34             $-PARAMETER  n=1156
# IE PY                  34             $-PARAMETER
# 
# IE PT                  75             $-PARAMETER  n=5625   original value
# IE PY                  75             $-PARAMETER           original value
# 
# IE PT                  100            $-PARAMETER  n=10000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'JNLBRNGB'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['PT'] = int(5);  #  SIF file default value
        else:
            v_['PT'] = int(args[0])
# IE PY                  100            $-PARAMETER
        if nargin<2:
            v_['PY'] = int(5);  #  SIF file default value
        else:
            v_['PY'] = int(args[1])
# IE PT                  125            $-PARAMETER  n=15625
# IE PY                  125            $-PARAMETER
        if nargin<3:
            v_['EX'] = float(0.5);  #  SIF file default value
        else:
            v_['EX'] = float(args[2])
        v_['LT'] = 6.2831853
        v_['LY'] = 20.0
        v_['PT-1'] = -1+v_['PT']
        v_['RPT-1'] = float(v_['PT-1'])
        v_['HT1'] = 1.0/v_['RPT-1']
        v_['HT'] = v_['HT1']*v_['LT']
        v_['1/HT'] = 1.0/v_['HT']
        v_['PY-1'] = -1+v_['PY']
        v_['RPY-1'] = float(v_['PY-1'])
        v_['HY1'] = 1.0/v_['RPY-1']
        v_['HY'] = v_['HY1']*v_['LY']
        v_['1/HY'] = 1.0/v_['HY']
        v_['HTHY'] = v_['HT']*v_['HY']
        v_['HT/HY'] = v_['HT']*v_['1/HY']
        v_['HY/HT'] = v_['HY']*v_['1/HT']
        v_['EXHTHY'] = v_['HTHY']*v_['EX']
        v_['CLINC'] = -1.0*v_['EXHTHY']
        v_['1'] = 1
        v_['2'] = 2
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['PT'])+1):
            for J in range(int(v_['1']),int(v_['PY'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['2']),int(v_['PT-1'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['XI1'] = v_['RI-1']*v_['HT']
            v_['SXI1'] = jnp.sin(v_['XI1'])
            v_['COEFF'] = v_['SXI1']*v_['CLINC']
            for J in range(int(v_['2']),int(v_['PY-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['COEFF']))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for J in range(int(v_['1']),int(v_['PY'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['PT']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['PT']))+','+str(J)]]), 0.0)
        for I in range(int(v_['2']),int(v_['PT-1'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['PY']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['PY']))]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))]]), 0.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eISQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['2']),int(v_['PT-1'])+1):
            v_['I+1'] = 1+I
            v_['I-1'] = -1+I
            for J in range(int(v_['2']),int(v_['PY-1'])+1):
                v_['J-1'] = -1+J
                v_['J+1'] = 1+J
                ename = 'A'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                self.x0 = jnp.zeros((self.n,1))
                vname = 'X'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'B'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'C'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(int(v_['I-1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'D'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(I)+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['PT-1'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['XI1'] = v_['RI-1']*v_['HT']
            v_['CXI1'] = jnp.cos(v_['XI1'])
            v_['ECX'] = v_['CXI1']*v_['EX']
            v_['ECX1'] = 1.0+v_['ECX']
            v_['E12'] = v_['ECX1']*v_['ECX1']
            v_['WI'] = v_['ECX1']*v_['E12']
            v_['2WI'] = v_['WI']+v_['WI']
            v_['RI'] = float(I)
            v_['XI+1'] = v_['RI']*v_['HT']
            v_['CXI+1'] = jnp.cos(v_['XI+1'])
            v_['E+CX0'] = v_['CXI+1']*v_['EX']
            v_['E+CX1'] = 1.0+v_['E+CX0']
            v_['E22'] = v_['E+CX1']*v_['E+CX1']
            v_['WI+1'] = v_['E+CX1']*v_['E22']
            v_['I-2'] = -2+I
            v_['RI-2'] = float(v_['I-2'])
            v_['XI-1'] = v_['RI-2']*v_['HT']
            v_['CXI-1'] = jnp.cos(v_['XI-1'])
            v_['E-CX0'] = v_['CXI-1']*v_['EX']
            v_['E-CX1'] = 1.0+v_['E-CX0']
            v_['E32'] = v_['E-CX1']*v_['E-CX1']
            v_['WI-1'] = v_['E-CX1']*v_['E32']
            v_['PM0'] = v_['2WI']*v_['WI+1']
            v_['PM1'] = 0.0833333333*v_['PM0']
            v_['MU/HY2'] = v_['PM1']*v_['HT/HY']
            v_['MU/HT2'] = v_['PM1']*v_['HY/HT']
            v_['PL0'] = v_['2WI']*v_['WI-1']
            v_['PL1'] = 0.0833333333*v_['PL0']
            v_['LA/HY2'] = v_['PL1']*v_['HT/HY']
            v_['LA/HT2'] = v_['PL1']*v_['HY/HT']
            for J in range(int(v_['2']),int(v_['PY-1'])+1):
                ig = ig_['G'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['MU/HT2']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['MU/HY2']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['LA/HT2']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['D'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['LA/HY2']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN(4)            -1.8551D+01
# LO SOLTN(10)           -7.2552D+00
# LO SOLTN(23)           -6.5068D+00
# LO SOLTN(32)           -6.4401D+00
# LO SOLTN(75)           -6.3297D+00
# LO SOLTN(100)          -6.3007D+00
# LO SOLTN(125)          -6.2807D+00
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CQBR2-AY-V-0"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eISQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((1,2))
        IV_ = jnp.zeros(1)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        f_   = IV_[0]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[0]+IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
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

