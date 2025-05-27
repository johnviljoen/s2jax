from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LOTSCHD:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A simple quadratic program inspired by the economic lot scheduling
#    problem.
# 
#    Source:
#    an exercize for L. Watson course on LANCELOT in the Spring 1993.
# 
#    SIF input: T. Kuan, Virginia Tech., Spring 1993.
# 
#    classification = "C-CQLR2-AN-12-7"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LOTSCHD'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['X1'] = 1.502
        v_['X2'] = 1.126
        v_['X3'] = 0.815
        v_['X4'] = 1.268
        v_['X5'] = 1.502
        v_['X6'] = 0.740
        v_['A1'] = 1.8
        v_['A2'] = 3.2
        v_['A3'] = 6.1
        v_['A4'] = 3.2
        v_['A5'] = 1.8
        v_['A6'] = 7.4
        v_['C1'] = 11.0
        v_['C2'] = 3.0
        v_['C3'] = 20.0
        v_['C4'] = 17.0
        v_['C5'] = 9.0
        v_['C6'] = 20.0
        v_['C7'] = 126.1
        v_['1'] = 1
        v_['2'] = 2
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
        for I in range(int(v_['1']),int(v_['6'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('T'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'T'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['6'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(v_['X'+str(I)]))
            [ig,ig_,_] = jtu.s2mpj_ii('CONS7',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS7')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('CONS'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(v_['A'+str(I)]))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['2']),int(v_['4'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CONS'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CONS2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CONS2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U3']])
        valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['2'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CONS3',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS3')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['4']),int(v_['6'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CONS3',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS3')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CONS4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CONS4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T1']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U1']])
        valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['5']),int(v_['6'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CONS4',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS4')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CONS5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CONS5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T1']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U1']])
        valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['5'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CONS6',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'CONS6')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
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
        for I in range(int(v_['1']),int(v_['7'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['CONS'+str(I)],float(v_['C'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gSQUARE',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['6'])+1):
            ig = ig_['OBJ'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CQLR2-AN-12-7"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSQUARE(self,nargout,*args):

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

