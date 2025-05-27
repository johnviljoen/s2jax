from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class NUFFIELD:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : NUFFIELD
#    *********
# 
#    A problem from economics.
#    Maximize a 2-D integral representing consumer surplus subject to 
#    linear and quadratic constraints representing incentive compatibility
# 
#    Let v( . , . ) : R^2 -> R, Omega = [a,a+1] x [a,a+1], and
#    the corners A, B, C, D be as follows:
# 
#            (a+1,a+1)
#        A *-----* B
#          |     |
#          |     |
#        D *-----* C
#        (a,a)  
# 
#    The problem is to maximize
# 
#       (a+1) line integral_{AB U BC} v(w)dw 
#        - a line integral_{CD U DA} v(w)dw
#        - 3 volume integral_{Omega} v(w)dw
# 
#    subject to v being symmetric (i.e., v(x,y) = v(y,x))
#               v(a,a) = 0
#               nabla_w v(w) >= 0
#               < e, nabla_w v(w) > <= 1
#         and   nabla_ww v(w) positive definite
# 
#    this last constraint is guaranteed by ensuring that
# 
#               d^2 v/dx^2 >= 0
#               d^2 v/dy^2 >= 0
#               ( d^2 v/dx^2 )( d^2 v/dy^2 ) >= ( d^2 v/dxdy )^2
# 
#    Symmetry is ensured by only considering v(x,y) for x <= y
# 
#    Here v(x,y) is the consumer surplus. that is if the consumer values good 
#    1 at x pounds and good 2 at y pounds then they will have a utility 
#    equivalent to v(x,y) pounds after being faced with the optimal monopoly 
#    pricing strategy. (Apparently, from this we can infer what the optimal 
#    pricing strategy was... ).
# 
#    More background is available from
# 
#    "Optimal Selling Strategies: When to haggle, when to hold firm",
#      Riley and Zeckhauser. The Quarterly Journal of Economics, 1983, and
# 
#    "Multidimensional Incentive Compatibility and Mechanism Design", 
#      McAfee and McMillan. The Journal of Economic Theory, 1988.
# 
#    Source: John Thanassoulis <john.thanassoulis@nuffield.oxford.ac.uk>
# 
#    Standard finite-differences are used to ap[proximate derivatives, and 
#    1- and 2-D trapezoidal rules to approximate integrals
# 
#    SIF input: Nick Gould, February 2001
# 
#    classification = "C-CLQR2-AN-V-V"
# 
#    The parameter a
# 
#           Alternative values for the SIF file parameters:
# RE A                   5.0            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'NUFFIELD'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['A'] = float(5.0);  #  SIF file default value
        else:
            v_['A'] = float(args[0])
# IE N                   10            $-PARAMETER
# IE N                   20            $-PARAMETER
# IE N                   30            $-PARAMETER
# IE N                   40            $-PARAMETER
        if nargin<2:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[1])
# IE N                   100           $-PARAMETER
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['N-1'] = -1+v_['N']
        v_['RN'] = float(v_['N'])
        v_['H'] = 1.0/v_['RN']
        v_['1/H'] = v_['RN']
        v_['-1/H'] = -1.0*v_['1/H']
        v_['H**2'] = v_['H']*v_['H']
        v_['1/H**2'] = v_['1/H']*v_['1/H']
        v_['-2/H**2'] = -2.0*v_['1/H**2']
        v_['1/H**4'] = v_['1/H**2']*v_['1/H**2']
        v_['A+1'] = 1.0+v_['A']
        v_['-A-1'] = -1.0*v_['A+1']
        v_['C2'] = 3.0*v_['H']
        v_['C3'] = 0.5*v_['C2']
        v_['C4'] = v_['C3']+v_['A']
        v_['C1'] = v_['C3']+v_['-A-1']
        v_['C5'] = -1.0+v_['C3']
        v_['C5'] = 0.5*v_['C5']
        v_['C6'] = 0.5*v_['C3']
        v_['C6'] = v_['C6']+v_['-A-1']
        v_['C6'] = 0.5*v_['C6']
        v_['C1'] = v_['C1']*v_['H']
        v_['C2'] = v_['C2']*v_['H']
        v_['C3'] = v_['C3']*v_['H']
        v_['C4'] = v_['C4']*v_['H']
        v_['C5'] = v_['C5']*v_['H']
        v_['C6'] = v_['C6']*v_['H']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(I)+1):
                [iv,ix_,_] = jtu.s2mpj_ii('V'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'V'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for J in range(int(v_['1']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(J)]])
            valA = jtu.append(valA,float(v_['C1']))
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['C2']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(I)]])
            valA = jtu.append(valA,float(v_['C3']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['0']))]])
            valA = jtu.append(valA,float(v_['C4']))
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['0']))]])
        valA = jtu.append(valA,float(v_['C5']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['C6']))
        for I in range(int(v_['0']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            for J in range(int(v_['0']),int(I)+1):
                [ig,ig_,_] = jtu.s2mpj_ii('VX'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'VX'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-1/H']))
                [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'VV'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-1/H']))
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('VX'+str(int(v_['N']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'VX'+str(int(v_['N']))+','+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(J)]])
            valA = jtu.append(valA,float(v_['1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('VX'+str(int(v_['N']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'VX'+str(int(v_['N']))+','+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['N-1']))+','+str(J)]])
            valA = jtu.append(valA,float(v_['-1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(J)]])
            valA = jtu.append(valA,float(v_['1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['N-1']))+','+str(J)]])
            valA = jtu.append(valA,float(v_['-1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VX'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'VX'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VX'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'VX'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(v_['-1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(v_['-1/H']))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            for J in range(int(v_['0']),int(v_['I-1'])+1):
                v_['J+1'] = 1+J
                [ig,ig_,_] = jtu.s2mpj_ii('VY'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'VY'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(v_['1/H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-1/H']))
                [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'VV'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(v_['1/H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-1/H']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('VY'+str(I)+','+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'VY'+str(I)+','+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(I)]])
            valA = jtu.append(valA,float(v_['1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(I)]])
            valA = jtu.append(valA,float(v_['-1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(I)+','+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'VV'+str(I)+','+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(I)]])
            valA = jtu.append(valA,float(v_['1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(I)]])
            valA = jtu.append(valA,float(v_['-1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VY'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'VY'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VY'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'VY'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(v_['-1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['1/H']))
        [ig,ig_,_] = jtu.s2mpj_ii('VV'+str(int(v_['N']))+','+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'VV'+str(int(v_['N']))+','+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['V'+str(int(v_['N']))+','+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(v_['-1/H']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            for J in range(int(v_['0']),int(v_['I-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('VXX'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'VXX'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H**2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2/H**2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(int(v_['I-1']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H**2']))
        for I in range(int(v_['2']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                v_['J-1'] = -1+J
                v_['J+1'] = 1+J
                [ig,ig_,_] = jtu.s2mpj_ii('VYY'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'VYY'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['J+1']))]])
                valA = jtu.append(valA,float(v_['1/H**2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2/H**2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['J-1']))]])
                valA = jtu.append(valA,float(v_['1/H**2']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('VXX'+str(I)+','+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'VXX'+str(I)+','+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(I)]])
            valA = jtu.append(valA,float(v_['1/H**2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(I)]])
            valA = jtu.append(valA,float(v_['-2/H**2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['1/H**2']))
            [ig,ig_,_] = jtu.s2mpj_ii('VYY'+str(I)+','+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'VYY'+str(I)+','+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(int(v_['I+1']))+','+str(I)]])
            valA = jtu.append(valA,float(v_['1/H**2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(I)]])
            valA = jtu.append(valA,float(v_['-2/H**2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['V'+str(I)+','+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['1/H**2']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            for J in range(int(v_['1']),int(I)+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(I)+','+str(J))
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
        for I in range(int(v_['0']),int(v_['N'])+1):
            for J in range(int(v_['0']),int(I)+1):
                self.gconst = jtu.arrset(self.gconst,ig_['VV'+str(I)+','+str(J)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['V'+str(int(v_['0']))+','+str(int(v_['0']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['V'+str(int(v_['0']))+','+str(int(v_['0']))]]), 0.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eCONVEX', iet_)
        elftv = jtu.loaset(elftv,it,0,'VIP1J')
        elftv = jtu.loaset(elftv,it,1,'VIJP1')
        elftv = jtu.loaset(elftv,it,2,'VIJ')
        elftv = jtu.loaset(elftv,it,3,'VIM1J')
        elftv = jtu.loaset(elftv,it,4,'VIJM1')
        elftv = jtu.loaset(elftv,it,5,'VIPJP')
        elftv = jtu.loaset(elftv,it,6,'VIPJM')
        elftv = jtu.loaset(elftv,it,7,'VIMJM')
        elftv = jtu.loaset(elftv,it,8,'VIMJP')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            v_['I-1'] = -1+I
            for J in range(int(v_['1']),int(I)+1):
                v_['J+1'] = 1+J
                v_['J-1'] = -1+J
                ename = 'C'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eCONVEX')
                ielftype = jtu.arrset(ielftype,ie,iet_["eCONVEX"])
                self.x0 = jnp.zeros((self.n,1))
                vname = 'V'+str(int(v_['I+1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIP1J')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(int(v_['I-1']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIM1J')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(I)+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIJP1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(I)+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIJM1')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIJ')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(int(v_['I+1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIPJP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(int(v_['I-1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIMJM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(int(v_['I+1']))+','+str(int(v_['J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIPJM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'V'+str(int(v_['I-1']))+','+str(int(v_['J+1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='VIMJP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            for J in range(int(v_['1']),int(I)+1):
                ig = ig_['C'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/H**4']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solutions (may be local!)
# LO SOLTN               -2.512312500   $ (n=10)
# LO SOLTN               -2.512359371   $ (n=20)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLQR2-AN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eCONVEX(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((3,9))
        IV_ = jnp.zeros(3)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]-2)
        U_ = jtu.np_like_set(U_, jnp.array([0,3]), U_[0,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]-2)
        U_ = jtu.np_like_set(U_, jnp.array([1,4]), U_[1,4]+1)
        U_ = jtu.np_like_set(U_, jnp.array([2,5]), U_[2,5]+2.500000e-01)
        U_ = jtu.np_like_set(U_, jnp.array([2,7]), U_[2,7]+2.500000e-01)
        U_ = jtu.np_like_set(U_, jnp.array([2,8]), U_[2,8]-2.500000e-01)
        U_ = jtu.np_like_set(U_, jnp.array([2,6]), U_[2,6]-2.500000e-01)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 2, U_[2:3,:].dot(EV_))
        f_   = IV_[0]*IV_[1]-IV_[2]*IV_[2]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ = jtu.np_like_set(g_, 2, -2.0*IV_[2])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -2.0)
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

