import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class READING3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : READING3
#    *********
# 
#    A nonlinear optimal control problem from Nancy Nichols
#    with a periodic boundary condition.
#    This problem arises in tide modelling.
# 
#    Source:
#    S. Lyle and N.K. Nichols,
#    "Numerical Methods for Optimal Control Problems with State Constraints",
#    Numerical Analysis Report 8/91, Dept of Mathematics, 
#    University of Reading, UK.
# 
#    SIF input: Nick Gould, July 1991.
# 
#    classification = "C-COOR2-MN-V-V"
# 
#    Number of discretized points in [0,1]
# 
#           Alternative values for the SIF file parameters:
# IE N                   2              $-PARAMETER n=6, m=3
# IE N                   5              $-PARAMETER n=12, m=6
# IE N                   50             $-PARAMETER n=102, m=51
# IE N                   100            $-PARAMETER n=202, m=101   original value
# IE N                   500            $-PARAMETER n=1002, m=501
# IE N                   1000           $-PARAMETER n=2002, m=1001
# IE N                   2000           $-PARAMETER n=4002, m=2001
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'READING3'

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
# IE N                   5000           $-PARAMETER n=10002, m=5001
        v_['PI'] = 3.1415926535
        v_['2PI'] = 2.0*v_['PI']
        v_['A'] = 0.07716
        v_['1/A'] = 1.0/v_['A']
        v_['1/2A'] = 0.5*v_['1/A']
        v_['2A'] = 2.0*v_['A']
        v_['-2A'] = -1.0*v_['2A']
        v_['-1/2A'] = 1.0/v_['-2A']
        v_['N-1'] = -1+v_['N']
        v_['RN'] = float(v_['N'])
        v_['H'] = 1.0/v_['RN']
        v_['2/H'] = 2.0*v_['RN']
        v_['H/2'] = 0.5*v_['H']
        v_['1/H'] = 1.0*v_['RN']
        v_['-1/H'] = -1.0*v_['RN']
        v_['0'] = 0
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('I'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['TI'] = v_['RI']*v_['H']
            v_['2PITI'] = v_['2PI']*v_['TI']
            v_['CTI'] = jnp.cos(v_['2PITI'])
            v_['CCTI'] = v_['CTI']*v_['-1/2A']
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['TI-1'] = v_['RI-1']*v_['H']
            v_['2PITI-1'] = v_['2PI']*v_['TI-1']
            v_['CTI-1'] = jnp.cos(v_['2PITI-1'])
            v_['CCTI-1'] = v_['CTI-1']*v_['-1/2A']
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['-1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(v_['CCTI']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['CCTI-1']))
        [ig,ig_,_] = jtu.s2mpj_ii('PERIOD',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'PERIOD')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['0']))]])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['N']))]])
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
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for I in range(int(v_['0']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(I)], -0.5)
            self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(I)], 0.5)
        for I in range(int(v_['0']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['U'+str(I)], 0.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['U'+str(I)], 1.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'P')
        [it,iet_,_] = jtu.s2mpj_ii( 'eENERGY', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'X')
        elftp = jtu.loaset(elftp,it,0,'T')
        elftp = jtu.loaset(elftp,it,1,'HOVER2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['0']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['TI'] = v_['RI']*v_['H']
            ename = 'I'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eENERGY')
            ielftype = jtu.arrset(ielftype,ie,iet_["eENERGY"])
            self.x0 = jnp.zeros((self.n,1))
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['TI']))
            posep = jnp.where(elftp[ielftype[ie]]=='HOVER2')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['H/2']))
        for I in range(int(v_['0']),int(v_['N'])+1):
            ename = 'NC'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='P')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['1/2A']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            ig = ig_['I'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['I'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['I'+str(int(v_['I-1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            ig = ig_['C'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['NC'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['NC'+str(int(v_['I-1']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
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
        self.pbclass   = "C-COOR2-MN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = self.elpar[iel_][0]*EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*EV_[1])
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][0]*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eENERGY(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C2PIT = jnp.cos(2.0*3.141592653589*self.elpar[iel_][0])
        f_   = self.elpar[iel_][1]*EV_[0]*(EV_[1]-C2PIT)**2
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][1]*(EV_[1]-C2PIT)**2)
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][1]*2.0*EV_[0]*(EV_[1]-C2PIT))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), self.elpar[iel_][1]*2.0*(EV_[1]-C2PIT))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][1]*2.0*EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

