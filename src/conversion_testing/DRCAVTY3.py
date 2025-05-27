from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DRCAVTY3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DRCAVTY3
#    *********
# 
#    This system of nonlinear equations models the stream function
#    corresponding to an incompressible fluid flow in a driven cavity 
#    (after elimination of the vorticity). 
# 
#    The problem is nonconvex.
#    It differs from the problems DRCAVTY1 and DRCAVTY2 by the value 
#    chosen for the Reynolds number.
# 
#    Source:  
#    P.N. Brown and Y. Saad, 
#    "Hybrid Krylov Methods for Nonlinear Systems of Equations",
#    SIAM J. Sci. Stat. Comput. 11, pp. 450-481, 1990.
#    The boundary conditions have been set according to
#    I.E. Kaporin and O. Axelsson,
#    "On a class of nonlinear equation solvers based on the residual norm
#    reduction over a sequence of affine subspaces",
#    SIAM J, Sci. Comput. 16(1), 1995.
# 
#    SIF input: Ph. Toint, Jan 1995.
# 
#    classification = "C-CNQR2-MY-V-V"
# 
#    Discretization mesh: n = (M+3)**2 - fixed variables
# 
#           Alternative values for the SIF file parameters:
# IE M                   10             $-PARAMETER  n =   100   original value
# IE M                   31             $-PARAMETER  n =   961
# IE M                   63             $-PARAMETER  n =  3969
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DRCAVTY3'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['M'] = int(10);  #  SIF file default value
        else:
            v_['M'] = int(args[0])
# IE M                   100            $-PARAMETER  n = 10000
        if nargin<2:
            v_['RE'] = float(4500.0);  #  SIF file default value
        else:
            v_['RE'] = float(args[1])
        v_['M+2'] = 2+v_['M']
        v_['RM+2'] = float(v_['M+2'])
        v_['H'] = 1.0/v_['RM+2']
        v_['-1'] = -1
        v_['0'] = 0
        v_['1'] = 1
        v_['M+1'] = 1+v_['M']
        v_['H/2'] = 0.5*v_['H']
        v_['-H/2'] = -1.0*v_['H/2']
        v_['RE/4'] = 0.25*v_['RE']
        v_['-RE/4'] = -1.0*v_['RE/4']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['-1']),int(v_['M+2'])+1):
            for J in range(int(v_['-1']),int(v_['M+2'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['I-2'] = -2+I
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            v_['I+2'] = 2+I
            for J in range(int(v_['1']),int(v_['M'])+1):
                v_['J-2'] = -2+J
                v_['J-1'] = -1+J
                v_['J+1'] = 1+J
                v_['J+2'] = 2+J
                [ig,ig_,_] = jtu.s2mpj_ii('E'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'E'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(20.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I-1']))+','+str(J)]])
                valA = jtu.append(valA,float(-8.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I+1']))+','+str(J)]])
                valA = jtu.append(valA,float(-8.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(int(v_['J-1']))]])
                valA = jtu.append(valA,float(-8.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(-8.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I-1']))+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(2.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I+1']))+','+str(int(v_['J-1']))]])
                valA = jtu.append(valA,float(2.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I-1']))+','+str(int(v_['J-1']))]])
                valA = jtu.append(valA,float(2.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I+1']))+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(2.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I-2']))+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(int(v_['I+2']))+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(int(v_['J-2']))]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(int(v_['J+2']))]])
                valA = jtu.append(valA,float(1.0))
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
        for J in range(int(v_['-1']),int(v_['M+2'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['-1']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['-1']))+','+str(J)]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['0']))+','+str(J)]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['0']))+','+str(J)]]), 0.0)
        for I in range(int(v_['1']),int(v_['M'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(I)+','+str(int(v_['-1']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(I)+','+str(int(v_['-1']))]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(I)+','+str(int(v_['0']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(I)+','+str(int(v_['0']))]]), 0.0)
        for I in range(int(v_['1']),int(v_['M'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(I)+','+str(int(v_['M+1']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(I)+','+str(int(v_['M+1']))]]), 0.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(I)+','+str(int(v_['M+2']))]]), 0.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(I)+','+str(int(v_['M+2']))]]), 0.0)
        for J in range(int(v_['-1']),int(v_['M+2'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['M+1']))+','+str(J)]]), v_['-H/2'])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['M+1']))+','+str(J)]]), v_['-H/2'])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['M+2']))+','+str(J)]]), v_['H/2'])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['M+2']))+','+str(J)]]), v_['H/2'])
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eIPR', iet_)
        elftv = jtu.loaset(elftv,it,0,'A1')
        elftv = jtu.loaset(elftv,it,1,'A2')
        elftv = jtu.loaset(elftv,it,2,'B1')
        elftv = jtu.loaset(elftv,it,3,'B2')
        elftv = jtu.loaset(elftv,it,4,'B3')
        elftv = jtu.loaset(elftv,it,5,'B4')
        elftv = jtu.loaset(elftv,it,6,'B5')
        elftv = jtu.loaset(elftv,it,7,'B6')
        elftv = jtu.loaset(elftv,it,8,'B7')
        elftv = jtu.loaset(elftv,it,9,'B8')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['I-2'] = -2+I
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            v_['I+2'] = 2+I
            for J in range(int(v_['1']),int(v_['M'])+1):
                v_['J-2'] = -2+J
                v_['J-1'] = -1+J
                v_['J+1'] = 1+J
                v_['J+2'] = 2+J
                ename = 'X'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eIPR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eIPR'])
                self.x0 = jnp.zeros((self.n,1))
                vname = 'Y'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='A1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='A2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-2']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B5')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B6')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B7')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+2']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B8')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'Z'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eIPR')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eIPR'])
                vname = 'Y'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='A1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='A2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J-2']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B2')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B3')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B4')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B5')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I-1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B6')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(int(v_['I+1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B7')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)+','+str(int(v_['J+2']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='B8')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            for J in range(int(v_['1']),int(v_['M'])+1):
                ig = ig_['E'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['X'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RE/4']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Z'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RE/4']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CNQR2-MY-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eIPR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,10))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,4]), U_[1,4]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,5]), U_[1,5]-4)
        U_ = jtu.np_like_set(U_, jnp.array([1,6]), U_[1,6]+4)
        U_ = jtu.np_like_set(U_, jnp.array([1,7]), U_[1,7]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,8]), U_[1,8]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,9]), U_[1,9]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[0]*IV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
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

