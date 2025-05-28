import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HANGING:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HANGING
#    *********
# 
#    A catenary problem in 3 dimensions.  A rectangular grid is hung from its
#    4 corners under gravity.  The problem is to determine the resulting shape.
# 
#    Source:  
#    an example in a talk by Nesterova and Vial, LLN, 1994.
# 
#    SIF input: Ph. Toint, November 1994.
# 
#    classification = "C-CLQR2-AY-V-V"
# 
#    dimension of the grid
# 
#           Alternative values for the SIF file parameters:
# IE NX                  3              $-PARAMETER n = 27
# IE NY                  3              $-PARAMETER
# 
# IE NX                  5              $-PARAMETER n = 90
# IE NY                  6              $-PARAMETER
# 
# IE NX                  10             $-PARAMETER n = 300  original value
# IE NY                  10             $-PARAMETER
# 
# IE NX                  20             $-PARAMETER n = 1800
# IE NY                  30             $-PARAMETER
# 
# IE NX                  40             $-PARAMETER n = 3600
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HANGING'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['NX'] = int(3);  #  SIF file default value
        else:
            v_['NX'] = int(args[0])
# IE NY                  30             $-PARAMETER
        if nargin<2:
            v_['NY'] = int(3);  #  SIF file default value
        else:
            v_['NY'] = int(args[1])
        v_['LX'] = 1.8
        v_['LY'] = 1.8
        v_['1'] = 1
        v_['NX-1'] = -1+v_['NX']
        v_['NY-1'] = -1+v_['NY']
        v_['LX2'] = v_['LX']*v_['LX']
        v_['LY2'] = v_['LY']*v_['LY']
        v_['RNX'] = float(v_['NX'])
        v_['RNY'] = float(v_['NY'])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['NX'])+1):
            for J in range(int(v_['1']),int(v_['NY'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J))
                [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(I)+','+str(J))
                [iv,ix_,_] = jtu.s2mpj_ii('Z'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'Z'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['NX'])+1):
            for J in range(int(v_['1']),int(v_['NY'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Z'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['1']),int(v_['NX'])+1):
            for J in range(int(v_['1']),int(v_['NY-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('RC'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'RC'+str(I)+','+str(J))
        for I in range(int(v_['1']),int(v_['NX-1'])+1):
            for J in range(int(v_['1']),int(v_['NY'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('DC'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'DC'+str(I)+','+str(J))
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
        for I in range(int(v_['1']),int(v_['NX'])+1):
            for J in range(int(v_['1']),int(v_['NY-1'])+1):
                self.gconst  = (                       jtu.arrset(self.gconst,ig_['RC'+str(I)+','+str(J)],float(v_['LX2'])))
        for I in range(int(v_['1']),int(v_['NX-1'])+1):
            for J in range(int(v_['1']),int(v_['NY'])+1):
                self.gconst  = (                       jtu.arrset(self.gconst,ig_['DC'+str(I)+','+str(J)],float(v_['LY2'])))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Z'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Z'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['NX']))+','+str(int(v_['1']))]]), v_['RNX'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['NX']))+','+str(int(v_['1']))]]), v_['RNX'])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['NX']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['NX']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Z'+str(int(v_['NX']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Z'+str(int(v_['NX']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['NY']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['NY']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['NY']))]]), v_['RNY'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['NY']))]]), v_['RNY'])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Z'+str(int(v_['1']))+','+str(int(v_['NY']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Z'+str(int(v_['1']))+','+str(int(v_['NY']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), v_['RNX'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), v_['RNX'])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), v_['RNY'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), v_['RNY'])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Z'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Z'+str(int(v_['NX']))+','+str(int(v_['NY']))]]), 0.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['NX'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            for J in range(int(v_['1']),int(v_['NY'])+1):
                v_['J-1'] = -1+J
                v_['RJ-1'] = float(v_['J-1'])
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(J)]]), float(v_['RI-1']))
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['Y'+str(I)+','+str(J)]]), float(v_['RJ-1']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eISQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'XX')
        elftv = jtu.loaset(elftv,it,1,'YY')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for J in range(int(v_['1']),int(v_['NY-1'])+1):
            v_['J+1'] = 1+J
            for I in range(int(v_['1']),int(v_['NX'])+1):
                ename = 'RX'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'RY'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'Y'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'RZ'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'Z'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Z'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NX-1'])+1):
            v_['I+1'] = 1+I
            for J in range(int(v_['1']),int(v_['NY'])+1):
                ename = 'DX'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'DY'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'Y'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'DZ'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eISQ'])
                vname = 'Z'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Z'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['NX'])+1):
            for J in range(int(v_['1']),int(v_['NY-1'])+1):
                ig = ig_['RC'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RX'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
                posel = posel+1
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RY'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RZ'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['1']),int(v_['NX-1'])+1):
            for J in range(int(v_['1']),int(v_['NY'])+1):
                ig = ig_['DC'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DX'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
                posel = posel+1
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DY'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DZ'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN(3,3)          -6.1184107487
# LO SOLTN(5,6)          -77.260229515
# LO SOLTN(10,10)        -620.17603242
# LO SOLTN(20,30)        -1025.4292887
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLQR2-AY-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

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

