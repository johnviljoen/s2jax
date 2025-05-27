from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class WATSON:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : WATSON
#    *********
# 
#    Watson problem in 12 variables.
# 
#    This function  is a nonlinear least squares with 31 groups.  Each
#    group has 1 nonlinear and 1 linear elements.
# 
#    Source:  problem 20 in
#    J.J. More', B.S. Garbow and K.E. Hillstrom,
#    "Testing Unconstrained Optimization Software",
#    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.
# 
#    See also Buckley#128 (p. 100).
# 
#    SIF input: Ph. Toint, Dec 1989.
#    (bug fix July 2007)
# 
#    classification = "C-CSUR2-AN-V-0"
# 
#    The number of variables can be varied, but should be smaller than
#    31 and larger than 12.
# 
#    Number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   12             $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'WATSON'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(12);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   31             $-PARAMETER
        v_['M'] = 31
        v_['1'] = 1
        v_['2'] = 2
        v_['29'] = 29
        v_['30'] = 30
        v_['29'] = 29.0
        v_['1/29'] = 1.0/v_['29']
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
        for I in range(int(v_['1']),int(v_['29'])+1):
            v_['RI'] = float(I)
            v_['TI'] = v_['RI']*v_['1/29']
            v_['LNTI'] = jnp.log(v_['TI'])
            for J in range(int(v_['2']),int(v_['N'])+1):
                v_['RJ'] = float(J)
                v_['RJ-1'] = -1.0+v_['RJ']
                v_['RJ-2'] = -2.0+v_['RJ']
                v_['AE'] = v_['RJ-2']*v_['LNTI']
                v_['C0'] = jnp.exp(v_['AE'])
                v_['C'] = v_['C0']*v_['RJ-1']
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['C']))
        [ig,ig_,_] = jtu.s2mpj_ii('G'+str(int(v_['30'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('G'+str(int(v_['M'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(1.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%  CONSTANTS %%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.full((ngrp,1),1.0)
        self.gconst = jtu.arrset(self.gconst,ig_['G'+str(int(v_['30']))],float(0.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eMWSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftv = jtu.loaset(elftv,it,2,'V3')
        elftv = jtu.loaset(elftv,it,3,'V4')
        elftv = jtu.loaset(elftv,it,4,'V5')
        elftv = jtu.loaset(elftv,it,5,'V6')
        elftv = jtu.loaset(elftv,it,6,'V7')
        elftv = jtu.loaset(elftv,it,7,'V8')
        elftv = jtu.loaset(elftv,it,8,'V9')
        elftv = jtu.loaset(elftv,it,9,'V10')
        elftv = jtu.loaset(elftv,it,10,'V11')
        elftv = jtu.loaset(elftv,it,11,'V12')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T1')
        elftp = jtu.loaset(elftp,it,1,'T2')
        elftp = jtu.loaset(elftp,it,2,'T3')
        elftp = jtu.loaset(elftp,it,3,'T4')
        elftp = jtu.loaset(elftp,it,4,'T5')
        elftp = jtu.loaset(elftp,it,5,'T6')
        elftp = jtu.loaset(elftp,it,6,'T7')
        elftp = jtu.loaset(elftp,it,7,'T8')
        elftp = jtu.loaset(elftp,it,8,'T9')
        elftp = jtu.loaset(elftp,it,9,'T10')
        elftp = jtu.loaset(elftp,it,10,'T11')
        elftp = jtu.loaset(elftp,it,11,'T12')
        [it,iet_,_] = jtu.s2mpj_ii( 'eMSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['29'])+1):
            v_['RI'] = float(I)
            v_['TI'] = v_['RI']*v_['1/29']
            v_['LNTI'] = jnp.log(v_['TI'])
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eMWSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eMWSQ"])
            self.x0 = jnp.zeros((self.n,1))
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X5'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V5')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X6'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V6')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X7'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V7')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X8'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V8')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X9'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V9')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X10'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V10')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X11'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V11')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X12'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V12')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            for J in range(int(v_['1']),int(v_['N'])+1):
                v_['J-1'] = -1+J
                v_['RJ-1'] = float(v_['J-1'])
                v_['CE0'] = v_['RJ-1']*v_['LNTI']
                v_['CE'+str(J)] = jnp.exp(v_['CE0'])
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            posep = jnp.where(elftp[ielftype[ie]]=='T1')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE1']))
            posep = jnp.where(elftp[ielftype[ie]]=='T2')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE2']))
            posep = jnp.where(elftp[ielftype[ie]]=='T3')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE3']))
            posep = jnp.where(elftp[ielftype[ie]]=='T4')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE4']))
            posep = jnp.where(elftp[ielftype[ie]]=='T5')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE5']))
            posep = jnp.where(elftp[ielftype[ie]]=='T6')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE6']))
            posep = jnp.where(elftp[ielftype[ie]]=='T7')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE7']))
            posep = jnp.where(elftp[ielftype[ie]]=='T8')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE8']))
            posep = jnp.where(elftp[ielftype[ie]]=='T9')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE9']))
            posep = jnp.where(elftp[ielftype[ie]]=='T10')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE10']))
            posep = jnp.where(elftp[ielftype[ie]]=='T11')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE11']))
            posep = jnp.where(elftp[ielftype[ie]]=='T12')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CE12']))
        ename = 'E'+str(int(v_['M']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eMSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eMSQ"])
        ename = 'E'+str(int(v_['M']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
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
        for ig in range(0,ngrp):
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        for I in range(int(v_['1']),int(v_['29'])+1):
            ig = ig_['G'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G'+str(int(v_['M']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['M']))])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLTN(12)           2.27559922D-9
# LO SOLTN(31)           1.53795068D-9
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-AN-V-0"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eMSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = -EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -EV_[0]-EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -2.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eMWSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U  = (
              self.elpar[iel_][0]*EV_[0]+self.elpar[iel_][1]*EV_[1]+self.elpar[iel_][2]*EV_[2]+self.elpar[iel_][3]*EV_[3]+self.elpar[iel_][4]*EV_[4]+self.elpar[iel_][5]*EV_[5]+self.elpar[iel_][6]*EV_[6]+self.elpar[iel_][7]*EV_[7]+self.elpar[iel_][8]*EV_[8]+self.elpar[iel_][9]*EV_[9]+self.elpar[iel_][10]*EV_[10]+self.elpar[iel_][11]*EV_[11])
        TWOT1 = self.elpar[iel_][0]+self.elpar[iel_][0]
        TWOT2 = self.elpar[iel_][1]+self.elpar[iel_][1]
        TWOT3 = self.elpar[iel_][2]+self.elpar[iel_][2]
        TWOT4 = self.elpar[iel_][3]+self.elpar[iel_][3]
        TWOT5 = self.elpar[iel_][4]+self.elpar[iel_][4]
        TWOT6 = self.elpar[iel_][5]+self.elpar[iel_][5]
        TWOT7 = self.elpar[iel_][6]+self.elpar[iel_][6]
        TWOT8 = self.elpar[iel_][7]+self.elpar[iel_][7]
        TWOT9 = self.elpar[iel_][8]+self.elpar[iel_][8]
        TWOT10 = self.elpar[iel_][9]+self.elpar[iel_][9]
        TWOT11 = self.elpar[iel_][10]+self.elpar[iel_][10]
        TWOT12 = self.elpar[iel_][11]+self.elpar[iel_][11]
        f_   = -U*U
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -TWOT1*U)
            g_ = jtu.np_like_set(g_, 1, -TWOT2*U)
            g_ = jtu.np_like_set(g_, 2, -TWOT3*U)
            g_ = jtu.np_like_set(g_, 3, -TWOT4*U)
            g_ = jtu.np_like_set(g_, 4, -TWOT5*U)
            g_ = jtu.np_like_set(g_, 5, -TWOT6*U)
            g_ = jtu.np_like_set(g_, 6, -TWOT7*U)
            g_ = jtu.np_like_set(g_, 7, -TWOT8*U)
            g_ = jtu.np_like_set(g_, 8, -TWOT9*U)
            g_ = jtu.np_like_set(g_, 9, -TWOT10*U)
            g_ = jtu.np_like_set(g_, 10, -TWOT11*U)
            g_ = jtu.np_like_set(g_, 11, -TWOT12*U)
            if nargout>2:
                H_ = jnp.zeros((12,12))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -TWOT1*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -TWOT1*self.elpar[iel_][1])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), -TWOT1*self.elpar[iel_][2])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), -TWOT1*self.elpar[iel_][3])
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), -TWOT1*self.elpar[iel_][4])
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([0,5]), -TWOT1*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,0]), H_[0,5])
                H_ = jtu.np_like_set(H_, jnp.array([0,6]), -TWOT1*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,0]), H_[0,6])
                H_ = jtu.np_like_set(H_, jnp.array([0,7]), -TWOT1*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,0]), H_[0,7])
                H_ = jtu.np_like_set(H_, jnp.array([0,8]), -TWOT1*self.elpar[iel_][8])
                H_ = jtu.np_like_set(H_, jnp.array([8,0]), H_[0,8])
                H_ = jtu.np_like_set(H_, jnp.array([0,9]), -TWOT1*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,0]), H_[0,9])
                H_ = jtu.np_like_set(H_, jnp.array([0,10]), -TWOT1*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,0]), H_[0,10])
                H_ = jtu.np_like_set(H_, jnp.array([0,11]), -TWOT1*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,0]), H_[0,11])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -TWOT2*self.elpar[iel_][1])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), -TWOT2*self.elpar[iel_][2])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), -TWOT2*self.elpar[iel_][3])
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), -TWOT2*self.elpar[iel_][4])
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,5]), -TWOT2*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,1]), H_[1,5])
                H_ = jtu.np_like_set(H_, jnp.array([1,6]), -TWOT2*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,1]), H_[1,6])
                H_ = jtu.np_like_set(H_, jnp.array([1,7]), -TWOT2*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,1]), H_[1,7])
                H_ = jtu.np_like_set(H_, jnp.array([1,8]), -TWOT2*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,1]), H_[1,8])
                H_ = jtu.np_like_set(H_, jnp.array([1,9]), -TWOT2*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,1]), H_[1,9])
                H_ = jtu.np_like_set(H_, jnp.array([1,10]), -TWOT2*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,1]), H_[1,10])
                H_ = jtu.np_like_set(H_, jnp.array([1,11]), -TWOT2*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,1]), H_[1,11])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -TWOT3*self.elpar[iel_][2])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), -TWOT3*self.elpar[iel_][3])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), -TWOT3*self.elpar[iel_][4])
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,5]), -TWOT3*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,2]), H_[2,5])
                H_ = jtu.np_like_set(H_, jnp.array([2,6]), -TWOT3*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,2]), H_[2,6])
                H_ = jtu.np_like_set(H_, jnp.array([2,7]), -TWOT3*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,2]), H_[2,7])
                H_ = jtu.np_like_set(H_, jnp.array([2,8]), -TWOT3*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,2]), H_[2,8])
                H_ = jtu.np_like_set(H_, jnp.array([2,9]), -TWOT3*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,2]), H_[2,9])
                H_ = jtu.np_like_set(H_, jnp.array([2,10]), -TWOT3*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,2]), H_[2,10])
                H_ = jtu.np_like_set(H_, jnp.array([2,11]), -TWOT3*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,2]), H_[2,11])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -TWOT4*self.elpar[iel_][3])
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), -TWOT4*self.elpar[iel_][4])
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,5]), -TWOT4*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,3]), H_[3,5])
                H_ = jtu.np_like_set(H_, jnp.array([3,6]), -TWOT4*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,3]), H_[3,6])
                H_ = jtu.np_like_set(H_, jnp.array([3,7]), -TWOT4*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,3]), H_[3,7])
                H_ = jtu.np_like_set(H_, jnp.array([3,8]), -TWOT4*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,3]), H_[3,8])
                H_ = jtu.np_like_set(H_, jnp.array([3,9]), -TWOT4*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,3]), H_[3,9])
                H_ = jtu.np_like_set(H_, jnp.array([3,10]), -TWOT4*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,3]), H_[3,10])
                H_ = jtu.np_like_set(H_, jnp.array([3,11]), -TWOT4*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,3]), H_[3,11])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), -TWOT5*self.elpar[iel_][4])
                H_ = jtu.np_like_set(H_, jnp.array([4,5]), -TWOT5*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,4]), H_[4,5])
                H_ = jtu.np_like_set(H_, jnp.array([4,6]), -TWOT5*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,4]), H_[4,6])
                H_ = jtu.np_like_set(H_, jnp.array([4,7]), -TWOT5*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,4]), H_[4,7])
                H_ = jtu.np_like_set(H_, jnp.array([4,8]), -TWOT5*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,4]), H_[4,8])
                H_ = jtu.np_like_set(H_, jnp.array([4,9]), -TWOT5*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,4]), H_[4,9])
                H_ = jtu.np_like_set(H_, jnp.array([4,10]), -TWOT5*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,4]), H_[4,10])
                H_ = jtu.np_like_set(H_, jnp.array([4,11]), -TWOT5*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,4]), H_[4,11])
                H_ = jtu.np_like_set(H_, jnp.array([5,5]), -TWOT6*self.elpar[iel_][5])
                H_ = jtu.np_like_set(H_, jnp.array([5,6]), -TWOT6*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,5]), H_[5,6])
                H_ = jtu.np_like_set(H_, jnp.array([5,7]), -TWOT6*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,5]), H_[5,7])
                H_ = jtu.np_like_set(H_, jnp.array([5,8]), -TWOT6*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,5]), H_[5,8])
                H_ = jtu.np_like_set(H_, jnp.array([5,9]), -TWOT6*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,5]), H_[5,9])
                H_ = jtu.np_like_set(H_, jnp.array([5,10]), -TWOT6*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,5]), H_[5,10])
                H_ = jtu.np_like_set(H_, jnp.array([5,11]), -TWOT6*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,5]), H_[5,11])
                H_ = jtu.np_like_set(H_, jnp.array([6,6]), -TWOT7*self.elpar[iel_][6])
                H_ = jtu.np_like_set(H_, jnp.array([6,7]), -TWOT7*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,6]), H_[6,7])
                H_ = jtu.np_like_set(H_, jnp.array([6,8]), -TWOT7*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,6]), H_[6,8])
                H_ = jtu.np_like_set(H_, jnp.array([6,9]), -TWOT7*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,6]), H_[6,9])
                H_ = jtu.np_like_set(H_, jnp.array([6,10]), -TWOT7*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,6]), H_[6,10])
                H_ = jtu.np_like_set(H_, jnp.array([6,11]), -TWOT7*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,6]), H_[6,11])
                H_ = jtu.np_like_set(H_, jnp.array([7,7]), -TWOT8*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([7,8]), -TWOT8*self.elpar[iel_][7])
                H_ = jtu.np_like_set(H_, jnp.array([8,7]), H_[7,8])
                H_ = jtu.np_like_set(H_, jnp.array([7,9]), -TWOT8*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,7]), H_[7,9])
                H_ = jtu.np_like_set(H_, jnp.array([7,10]), -TWOT8*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,7]), H_[7,10])
                H_ = jtu.np_like_set(H_, jnp.array([7,11]), -TWOT8*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,7]), H_[7,11])
                H_ = jtu.np_like_set(H_, jnp.array([8,8]), -TWOT9*self.elpar[iel_][8])
                H_ = jtu.np_like_set(H_, jnp.array([8,9]), -TWOT9*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,8]), H_[8,9])
                H_ = jtu.np_like_set(H_, jnp.array([8,10]), -TWOT9*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,8]), H_[8,10])
                H_ = jtu.np_like_set(H_, jnp.array([8,11]), -TWOT9*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,8]), H_[8,11])
                H_ = jtu.np_like_set(H_, jnp.array([9,9]), -TWOT10*self.elpar[iel_][9])
                H_ = jtu.np_like_set(H_, jnp.array([9,10]), -TWOT10*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,9]), H_[9,10])
                H_ = jtu.np_like_set(H_, jnp.array([9,11]), -TWOT10*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,9]), H_[9,11])
                H_ = jtu.np_like_set(H_, jnp.array([10,10]), -TWOT11*self.elpar[iel_][10])
                H_ = jtu.np_like_set(H_, jnp.array([10,11]), -TWOT11*self.elpar[iel_][11])
                H_ = jtu.np_like_set(H_, jnp.array([11,10]), H_[10,11])
                H_ = jtu.np_like_set(H_, jnp.array([11,11]), -TWOT12*self.elpar[iel_][11])
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

