import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class VAREIGVL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : VAREIGVL
#    *********
# 
#    The variational eigenvalue by Auchmuty.
#    This problems features a banded matrix of bandwidth 2M+1 = 9.
# 
#    This problem has N least-squares groups, each having a linear part
#    only and N nonlinear elements,
#    plus a least q-th power group having N nonlinear elements.
# 
#    Source: problem 1 in
#    J.J. More',
#    "A collection of nonlinear model problems"
#    Proceedings of the AMS-SIAM Summer seminar on the Computational
#    Solution of Nonlinear Systems of Equations, Colorado, 1988.
#    Argonne National Laboratory MCS-P60-0289, 1989.
# 
#    SIF input: Ph. Toint, Dec 1989.
#               correction by Ph. Shott, January, 1995
#               and Nick Gould, December, 2019, May 2024
# 
#    classification = "C-COUR2-AN-V-0"
# 
#    Number of variables -1 (variable)
# 
#           Alternative values for the SIF file parameters:
# IE N                   19             $-PARAMETER
# IE N                   49             $-PARAMETER     original value
# IE N                   99             $-PARAMETER
# IE N                   499            $-PARAMETER
# IE N                   999            $-PARAMETER
# IE N                   4999           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'VAREIGVL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(19);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE M                   4              $-PARAMETER  .le. N
# IE M                   5              $-PARAMETER  .le. N
        if nargin<2:
            v_['M'] = int(6);  #  SIF file default value
        else:
            v_['M'] = int(args[1])
        if nargin<3:
            v_['Q'] = float(1.5);  #  SIF file default value
        else:
            v_['Q'] = float(args[2])
        v_['1'] = 1
        v_['-1.0'] = -1.0
        v_['N+1'] = 1+v_['N']
        v_['-M'] = -1*v_['M']
        v_['M+1'] = 1+v_['M']
        v_['N-M'] = v_['N']+v_['-M']
        v_['N-M+1'] = 1+v_['N-M']
        v_['N2'] = v_['N']*v_['N']
        v_['RN2'] = float(v_['N2'])
        v_['-1/N2'] = v_['-1.0']/v_['RN2']
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
        [iv,ix_,_] = jtu.s2mpj_ii('MU',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MU')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['RI'] = float(I)
            v_['-I'] = -1.0*v_['RI']
            v_['I+M'] = I+v_['M']
            for J in range(int(v_['1']),int(v_['I+M'])+1):
                v_['RJ'] = float(J)
                v_['IJ'] = v_['RI']*v_['RJ']
                v_['SIJ'] = jnp.sin(v_['IJ'])
                v_['J-I'] = v_['RJ']+v_['-I']
                v_['J-ISQ'] = v_['J-I']*v_['J-I']
                v_['ARG'] = v_['J-ISQ']*v_['-1/N2']
                v_['EX'] = jnp.exp(v_['ARG'])
                v_['AIJ'] = v_['SIJ']*v_['EX']
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['AIJ']))
        for I in range(int(v_['M+1']),int(v_['N-M'])+1):
            v_['RI'] = float(I)
            v_['-I'] = -1.0*v_['RI']
            v_['I-M'] = I+v_['-M']
            v_['I+M'] = I+v_['M']
            for J in range(int(v_['I-M']),int(v_['I+M'])+1):
                v_['RJ'] = float(J)
                v_['IJ'] = v_['RI']*v_['RJ']
                v_['SIJ'] = jnp.sin(v_['IJ'])
                v_['J-I'] = v_['RJ']+v_['-I']
                v_['J-ISQ'] = v_['J-I']*v_['J-I']
                v_['ARG'] = v_['J-ISQ']*v_['-1/N2']
                v_['EX'] = jnp.exp(v_['ARG'])
                v_['AIJ'] = v_['SIJ']*v_['EX']
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['AIJ']))
        for I in range(int(v_['N-M+1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['-I'] = -1.0*v_['RI']
            v_['I-M'] = I+v_['-M']
            for J in range(int(v_['I-M']),int(v_['N'])+1):
                v_['RJ'] = float(J)
                v_['IJ'] = v_['RI']*v_['RJ']
                v_['SIJ'] = jnp.sin(v_['IJ'])
                v_['J-I'] = v_['RJ']+v_['-I']
                v_['J-ISQ'] = v_['J-I']*v_['J-I']
                v_['ARG'] = v_['J-ISQ']*v_['-1/N2']
                v_['EX'] = jnp.exp(v_['ARG'])
                v_['AIJ'] = v_['SIJ']*v_['EX']
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['AIJ']))
        [ig,ig_,_] = jtu.s2mpj_ii('G'+str(int(v_['N+1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MU'], float(0.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'M')
        elftv = jtu.loaset(elftv,it,1,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'P'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'MU'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='M')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'S'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gLQ',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gLQ',igt_)
        grftp = []
        grftp = jtu.loaset(grftp,it,0,'POWER')
        [it,igt_,_] = jtu.s2mpj_ii('gLQ2',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gLQ2',igt_)
        grftp = jtu.loaset(grftp,it,0,'POWER')
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        self.grpar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['G'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gLQ')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['P'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='POWER')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(2.0))
        ig = ig_['G'+str(int(v_['N+1']))]
        self.grftype = jtu.arrset(self.grftype,ig,'gLQ2')
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['G'+str(int(v_['N+1']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G'+str(int(v_['N+1']))]
        posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='POWER')[0]
        self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(v_['Q']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COUR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

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

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gLQ(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        IPOWER = self.grpar[igr_][0]
        PM1 = IPOWER-1
        f_= GVAR_**IPOWER/self.grpar[igr_][0]
        if nargout>1:
            g_ = GVAR_**PM1
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = PM1*GVAR_**(IPOWER-2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def gLQ2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_**self.grpar[igr_][0]/self.grpar[igr_][0]
        if nargout>1:
            g_ = GVAR_**(self.grpar[igr_][0]-1.0e0)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = (self.grpar[igr_][0]-1.0e0)*GVAR_**(self.grpar[igr_][0]-2.0e0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

