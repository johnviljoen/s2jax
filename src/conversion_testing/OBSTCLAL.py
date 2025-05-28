import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class OBSTCLAL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : OBSTCLAL
#    *********
# 
#    A quadratic obstacle problem by Dembo and Tulowitzki
# 
#    The problem comes from the obstacle problem on a rectangle.
#    The rectangle is discretized into (px-1)(py-1) little rectangles. The
#    heights of the considered surface above the corners of these little
#    rectangles are the problem variables,  There are px*py of them.
# 
#    Source:
#    R. Dembo and U. Tulowitzki,
#    "On the minimization of quadratic functions subject to box
#    constraints",
#    WP 71, Yale University (new Haven, USA), 1983.
# 
#    See also More 1989 (Problem A, Starting point L
# 
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CQBR2-AY-V-0"
# 
#    PX is the number of points along the X side of the rectangle
#    PY is the number of points along the Y side of the rectangle
# 
#           Alternative values for the SIF file parameters:
# IE PX                  4              $-PARAMETER n = 16
# IE PY                  4              $-PARAMETER
# 
# IE PX                  10             $-PARAMETER n = 100     original value
# IE PY                  10             $-PARAMETER             original value
# 
# IE PX                  23             $-PARAMETER n = 529
# IE PY                  23             $-PARAMETER
# 
# IE PX                  32             $-PARAMETER n = 1024
# IE PY                  32             $-PARAMETER
# 
# IE PX                  75             $-PARAMETER n = 5625
# IE PY                  75             $-PARAMETER
# 
# IE PX                  100            $-PARAMETER n = 10000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'OBSTCLAL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['PX'] = int(5);  #  SIF file default value
        else:
            v_['PX'] = int(args[0])
# IE PY                  100            $-PARAMETER
        if nargin<2:
            v_['PY'] = int(20);  #  SIF file default value
        else:
            v_['PY'] = int(args[1])
# IE PX                  125            $-PARAMETER n = 15625
# IE PY                  125            $-PARAMETER
        if nargin<3:
            v_['C'] = float(1.0);  #  SIF file default value
        else:
            v_['C'] = float(args[2])
        v_['PX-1'] = -1+v_['PX']
        v_['RPX-1'] = float(v_['PX-1'])
        v_['HX'] = 1.0/v_['RPX-1']
        v_['PY-1'] = -1+v_['PY']
        v_['RPY-1'] = float(v_['PY-1'])
        v_['HY'] = 1.0/v_['RPY-1']
        v_['HXHY'] = v_['HX']*v_['HY']
        v_['1/HX'] = 1.0/v_['HX']
        v_['1/HY'] = 1.0/v_['HY']
        v_['HX/HY'] = v_['HX']*v_['1/HY']
        v_['HY/HX'] = v_['HY']*v_['1/HX']
        v_['HY/4HX'] = 0.25*v_['HY/HX']
        v_['HX/4HY'] = 0.25*v_['HX/HY']
        v_['C0'] = v_['HXHY']*v_['C']
        v_['LC'] = -1.0*v_['C0']
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
        for J in range(int(v_['1']),int(v_['PX'])+1):
            for I in range(int(v_['1']),int(v_['PY'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            for J in range(int(v_['2']),int(v_['PX-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['LC']))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xupper = jnp.full((self.n,1),2000.0)
        self.xlower = jnp.zeros((self.n,1))
        for J in range(int(v_['1']),int(v_['PX'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['PY']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['PY']))+','+str(J)]]), 0.0)
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['PX']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['PX']))]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))]]), 0.0)
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['XSI1'] = v_['RI-1']*v_['HY']
            v_['3XSI1'] = 3.2*v_['XSI1']
            v_['SXSI1'] = jnp.sin(v_['3XSI1'])
            for J in range(int(v_['2']),int(v_['PX-1'])+1):
                v_['J-1'] = -1+J
                v_['RJ-1'] = float(v_['J-1'])
                v_['XSI2'] = v_['RJ-1']*v_['HX']
                v_['3XSI2'] = 3.3*v_['XSI2']
                v_['SXSI2'] = jnp.sin(v_['3XSI2'])
                v_['LOW'] = v_['SXSI1']*v_['SXSI2']
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(I)+','+str(J)]]), v_['LOW'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for J in range(int(v_['1']),int(v_['PX'])+1):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['1']))+','+str(J)]]), float(0.0))
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['PY']))+','+str(J)]]), float(0.0))
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(int(v_['PX']))]]), float(0.0))
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(int(v_['1']))]]), float(0.0))
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['XSI1'] = v_['RI-1']*v_['HY']
            v_['3XSI1'] = 3.2*v_['XSI1']
            v_['SXSI1'] = jnp.sin(v_['3XSI1'])
            for J in range(int(v_['2']),int(v_['PX-1'])+1):
                v_['J-1'] = -1+J
                v_['RJ-1'] = float(v_['J-1'])
                v_['XSI2'] = v_['RJ-1']*v_['HX']
                v_['3XSI2'] = 3.3*v_['XSI2']
                v_['SXSI2'] = jnp.sin(v_['3XSI2'])
                v_['LOW'] = v_['SXSI1']*v_['SXSI2']
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(J)]]), float(v_['LOW']))
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
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            for J in range(int(v_['2']),int(v_['PX-1'])+1):
                v_['J-1'] = -1+J
                v_['J+1'] = 1+J
                ename = 'A'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'B'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'C'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(int(v_['I-1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'D'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
                vname = 'X'+str(I)+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,float(2000.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['PY-1'])+1):
            for J in range(int(v_['2']),int(v_['PX-1'])+1):
                ig = ig_['G'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['HY/4HX']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['HX/4HY']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['HY/4HX']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['D'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['HX/4HY']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN(4)            0.753659754
# LO SOLTN(10)           1.397897560
# LO SOLTN(23)           1.678027027
# LO SOLTN(32)           1.748270031
# LO SOLTN(75)           ???
# LO SOLTN(100)          ???
# LO SOLTN(125)          ???
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CQBR2-AY-V-0"
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

