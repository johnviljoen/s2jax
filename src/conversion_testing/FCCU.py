from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class FCCU:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : FCCU
#    *********
# 
#    A simple data reconciliation for a fluid catalytic cracker.
# 
#                      +--------------------------+
#                      | FCCU data reconciliation |
#     +-----------+    +--------------------------+
#  1) | Flowsheet |                      Off_gas                      |------->
#     +-----------+                     |----------->                 | Propane
#                              MF_ohd   |                  |-------->F7
#                              |------>F4<-------|         |          |
#                              |        |        |         |DC3_feed  | Butane
#  Feed      Effluent          ^        |        |         |          |------->
#  ------->F1-------->F2......>|        |---------------->F5                  
#           ^                  v       DC4_feed  |         |DC4_btms  |------->
#           |                  |                 |         |          |  LCN
#           |                  |                 |<-------F6-------->F8
#           |                  | HCN              Lean_oil   C8spl_fd |  MCN
#           |                  |-------->                             |------->
#           |                  | LCO          
#           |                  |-------->              
#           |                  | HCO          
#           |                  |-------->              
#           |                  | MF_btms     
#           |                  v 
#           |<----------------F3-------->
#              Dec_recy         Decant
#     +--------------------+
#  2) | Objective function |
#     +--------------------+
#    Obj = sum 1->i [W_i*(C_flow_i - M_flow_i)**2]
#                           
#     Where: W_i       Weight on term i of objective function
#            C_flow_i  Computed flow i (a variable for this problem)
#            M_flow_i  Measrued flow i (a constant for this problem)
#     +-------------+
#  3) | Constraints |
#     +-------------+
#     These represent the linear mass balances around each
#     node, where a node (Fx) represents a single unit operation
#     in a fluid catalytics cracker.
#     +---------------+
#  4) | Initial point |
#     +---------------+
#     Feed       1.0
#     Effluent   1.0
#     MF_ohd     1.0
#     HCN        1.0
#     LCO        1.0
#     HCO        1.0
#     MF_btms    1.0
#     Decant     1.0
#     Dec_recy   1.0
#     Off_gas    1.0
#     DC4_feed   1.0
#     DC3_feed   1.0
#     DC4_btms   1.0
#     Lean_oil   1.0
#     Propane    1.0
#     Butane     1.0
#     C8spl_fd   1.0
#     LCN        1.0
#     MCN        1.0
#     Obj        7.36259000271320D+03
#     +------------------+
#  5) | Optimal solution |
#     +------------------+
#     Feed       3.11639D+01
#     Effluent   3.53528D+01
#     MF_ohd     1.94669D+01
#     HCN        2.94255D+00
#     LCO        4.94255D+00
#     HCO        3.44255D+00
#     MF_btms    4.55828D+00
#     Decant     3.69371D-01
#     Dec_recy   4.18891D+00
#     Off_gas    2.56075D+00
#     DC4_feed   2.41207D+01
#     DC3_feed   5.15601D+00
#     DC4_btms   1.89647D+01
#     Lean_oil   7.21458D+00
#     Propane    2.42801D+00
#     Butane     2.72801D+00
#     C8spl_fd   1.17501D+01
#     LCN        5.87506D+00
#     MCN        5.87506D+00
#     Obj        1.11491D+01
#     +-----------------------------------------------+
#  6) | SPEC.SPC (remove 1st * of every line to use). |
#     +-----------------------------------------------+
# BEGIN
# * maximizer-sought
# *  check-derivatives
#   ignore-derivative-bugs
# * use-scalings
# * print-scalings
#   finite-difference-gradients
# *  exact-second-derivatives-used
# * bfgs-approximate-second-derivatives-used
# * sr1-approximate-second-derivatives-used
#   bandsolver-preconditioned-cg-solver-used   5
# * diagonal-preconditioned-cg-solver-used
# * gill-murray-ponceleon-saunders-preconditioned-cg-solver-used
# * schnabel-eskow-preconditioned-cg-solver-used
# * munksgaards-preconditioned-cg-solver-used
#   exact-cauchy-point-required
# * inexact-cauchy-point-required
# * solve-bqp-accurately
# * two-norm-trust-region
# * gradient-tolerance    1.0D-5
# * constraint-tolerance  1.0D-5
#   trust-region-radius   1.0D+0
#   maximum-number-of-iterations   1000
#   print-level                    1
#   start-printing-at-iteration    0
#   stop-printing-at-iteration     1000
# END
# 
#    Source:
#    W. J. Korchinski, Profimatics, Inc,
#    325 Rolling Oaks Drive, Thousand Oaks, California, USA 91361-1200
#    Telephone: 1-805 496 6661, Fax: 1-805 373 5108
# 
#    SIF input: W. Korchinski, Spring 1993.
# 
#    classification = "C-CSLR2-MN-19-8"
# 
# ***************************************************************
#  PROBLEM SPECIFICATION BEGINS HERE.
#  **********************************
#  **********************************           
# *************************************
#  Define objective function weights. *
# *************************************
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'FCCU'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['W1'] = 0.2
        v_['W2'] = 1.0
        v_['W3'] = 1.0
        v_['W4'] = 0.33333333
        v_['W5'] = 0.33333333
        v_['W6'] = 0.33333333
        v_['W7'] = 1.0
        v_['W8'] = 1.0
        v_['W9'] = 1.0
        v_['W10'] = 1.0
        v_['W11'] = 1.0
        v_['W12'] = 1.0
        v_['W13'] = 1.0
        v_['W14'] = 1.0
        v_['W15'] = 0.33333333
        v_['W16'] = 0.33333333
        v_['W17'] = 1.0
        v_['W18'] = 0.33333333
        v_['W19'] = 0.33333333
        v_['M1'] = 31.0
        v_['M2'] = 36.0
        v_['M3'] = 20.0
        v_['M4'] = 3.0
        v_['M5'] = 5.0
        v_['M6'] = 3.5
        v_['M7'] = 4.2
        v_['M8'] = 0.9
        v_['M9'] = 3.9
        v_['M10'] = 2.2
        v_['M11'] = 22.8
        v_['M12'] = 6.8
        v_['M13'] = 19.0
        v_['M14'] = 8.5
        v_['M15'] = 2.2
        v_['M16'] = 2.5
        v_['M17'] = 10.8
        v_['M18'] = 6.5
        v_['M19'] = 6.5
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('Feed',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Feed')
        [iv,ix_,_] = jtu.s2mpj_ii('Effluent',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Effluent')
        [iv,ix_,_] = jtu.s2mpj_ii('MFuohd',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MFuohd')
        [iv,ix_,_] = jtu.s2mpj_ii('HCN',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'HCN')
        [iv,ix_,_] = jtu.s2mpj_ii('LCO',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'LCO')
        [iv,ix_,_] = jtu.s2mpj_ii('HCO',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'HCO')
        [iv,ix_,_] = jtu.s2mpj_ii('MFubtms',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MFubtms')
        [iv,ix_,_] = jtu.s2mpj_ii('Decant',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Decant')
        [iv,ix_,_] = jtu.s2mpj_ii('Decurecy',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Decurecy')
        [iv,ix_,_] = jtu.s2mpj_ii('Offugas',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Offugas')
        [iv,ix_,_] = jtu.s2mpj_ii('DC4ufeed',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'DC4ufeed')
        [iv,ix_,_] = jtu.s2mpj_ii('DC3ufeed',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'DC3ufeed')
        [iv,ix_,_] = jtu.s2mpj_ii('DC4ubtms',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'DC4ubtms')
        [iv,ix_,_] = jtu.s2mpj_ii('Leanuoil',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Leanuoil')
        [iv,ix_,_] = jtu.s2mpj_ii('Propane',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Propane')
        [iv,ix_,_] = jtu.s2mpj_ii('Butane',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Butane')
        [iv,ix_,_] = jtu.s2mpj_ii('C8splufd',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'C8splufd')
        [iv,ix_,_] = jtu.s2mpj_ii('LCN',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'LCN')
        [iv,ix_,_] = jtu.s2mpj_ii('MCN',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MCN')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('F1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Feed']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Decurecy']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Effluent']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Effluent']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFuohd']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['HCN']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['LCO']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['HCO']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFubtms']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFubtms']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Decant']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Decurecy']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFuohd']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Leanuoil']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Offugas']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ufeed']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ufeed']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC3ufeed']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ubtms']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F6')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ubtms']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Leanuoil']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['C8splufd']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F7',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F7')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC3ufeed']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Propane']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Butane']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('F8',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'F8')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['C8splufd']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['LCN']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MCN']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj1',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Feed']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W1']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj2',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Effluent']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W2']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj3',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFuohd']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W3']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj4',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['HCN']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W4']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj5',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['LCO']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W5']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj6',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['HCO']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W6']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj7',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MFubtms']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W7']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj8',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Decant']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W8']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj9',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Decurecy']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W9']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj10',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Offugas']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W10']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj11',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ufeed']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W11']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj12',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC3ufeed']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W12']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj13',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['DC4ubtms']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W13']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj14',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Leanuoil']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W14']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj15',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Propane']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W15']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj16',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Butane']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W16']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj17',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['C8splufd']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W17']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj18',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['LCN']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W18']))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj19',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MCN']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['W19']))
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
        self.gconst = jtu.arrset(self.gconst,ig_['Obj1'],float(v_['M1']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj2'],float(v_['M2']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj3'],float(v_['M3']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj4'],float(v_['M4']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj5'],float(v_['M5']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj6'],float(v_['M6']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj7'],float(v_['M7']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj8'],float(v_['M8']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj9'],float(v_['M9']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj10'],float(v_['M10']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj11'],float(v_['M11']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj12'],float(v_['M12']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj13'],float(v_['M13']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj14'],float(v_['M14']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj15'],float(v_['M15']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj16'],float(v_['M16']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj17'],float(v_['M17']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj18'],float(v_['M18']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj19'],float(v_['M19']))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['Feed'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Effluent'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MFuohd'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['HCN'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['LCO'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['HCO'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MFubtms'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Decant'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Decurecy'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Offugas'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['DC4ufeed'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['DC3ufeed'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['DC4ubtms'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Leanuoil'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Propane'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Butane'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['C8splufd'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['LCN'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MCN'], float(1.0))
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
        ig = ig_['Obj1']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj2']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj3']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj4']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj5']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj6']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj7']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj8']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj9']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj10']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj11']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj12']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj13']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj14']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj15']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj16']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj17']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj18']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj19']
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
        self.pbclass   = "C-CSLR2-MN-19-8"
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

