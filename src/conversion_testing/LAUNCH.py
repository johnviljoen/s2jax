from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LAUNCH:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    The objective function to be minimized represents the total cost of
#    the development and launching of a 3 stages space launching vehicle.
#    Constraints are imposed on physical interrelations between the variables
#    and performance.
# 
#    The problem is highly non-convex. 
# 
#    Source:
#    B. Rush, J. Bracken and G. McCormick,
#    "A nonliner programming model for launch vehicle design and costing",
#    Operations Research, pp. 185-210, 1967.
# 
#    SIF input: P. Driscoll, Virginia Tech., April 1993,
#               corrected and simplified by Ph. L. Toint, May 1993.
# 
#    classification = "C-COOR2-MY-25-28"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LAUNCH'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('AW1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'AW1')
        [iv,ix_,_] = jtu.s2mpj_ii('IW1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'IW1')
        [iv,ix_,_] = jtu.s2mpj_ii('MF1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MF1')
        [iv,ix_,_] = jtu.s2mpj_ii('TT1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'TT1')
        [iv,ix_,_] = jtu.s2mpj_ii('PW1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PW1')
        [iv,ix_,_] = jtu.s2mpj_ii('ET1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ET1')
        [iv,ix_,_] = jtu.s2mpj_ii('S1L',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'S1L')
        [iv,ix_,_] = jtu.s2mpj_ii('AW2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'AW2')
        [iv,ix_,_] = jtu.s2mpj_ii('IW2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'IW2')
        [iv,ix_,_] = jtu.s2mpj_ii('MF2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MF2')
        [iv,ix_,_] = jtu.s2mpj_ii('TT2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'TT2')
        [iv,ix_,_] = jtu.s2mpj_ii('PW2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PW2')
        [iv,ix_,_] = jtu.s2mpj_ii('ET2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ET2')
        [iv,ix_,_] = jtu.s2mpj_ii('S2L',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'S2L')
        [iv,ix_,_] = jtu.s2mpj_ii('AW3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'AW3')
        [iv,ix_,_] = jtu.s2mpj_ii('IW3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'IW3')
        [iv,ix_,_] = jtu.s2mpj_ii('MF3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'MF3')
        [iv,ix_,_] = jtu.s2mpj_ii('TT3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'TT3')
        [iv,ix_,_] = jtu.s2mpj_ii('PW3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PW3')
        [iv,ix_,_] = jtu.s2mpj_ii('ET3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ET3')
        [iv,ix_,_] = jtu.s2mpj_ii('S3L',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'S3L')
        [iv,ix_,_] = jtu.s2mpj_ii('INW',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'INW')
        [iv,ix_,_] = jtu.s2mpj_ii('BT1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'BT1')
        [iv,ix_,_] = jtu.s2mpj_ii('BT2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'BT2')
        [iv,ix_,_] = jtu.s2mpj_ii('BT3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'BT3')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('STA1',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET1']])
        valA = jtu.append(valA,float(0.0002587))
        self.gscale = jtu.arrset(self.gscale,ig,float(1.0e+8))
        [ig,ig_,_] = jtu.s2mpj_ii('STA2',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET2']])
        valA = jtu.append(valA,float(0.0002587))
        self.gscale = jtu.arrset(self.gscale,ig,float(1.0e+8))
        [ig,ig_,_] = jtu.s2mpj_ii('STA3',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET3']])
        valA = jtu.append(valA,float(0.001958))
        self.gscale = jtu.arrset(self.gscale,ig,float(1.0e+8))
        [ig,ig_,_] = jtu.s2mpj_ii('INST',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(47.040096))
        self.gscale = jtu.arrset(self.gscale,ig,float(1.0e+8))
        [ig,ig_,_] = jtu.s2mpj_ii('LAUN',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(0.003))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(0.003))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(0.003))
        self.gscale = jtu.arrset(self.gscale,ig,float(39215686.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AW1']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(0.6))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AW2']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(0.7))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AW3']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET1']])
        valA = jtu.append(valA,float(5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT1']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET2']])
        valA = jtu.append(valA,float(5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT2']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGTH6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SGTH6')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ET3']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI1A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SGSI1A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-12.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI1B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SGSI1B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-17.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI2A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SGSI2A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-10.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI2B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SGSI2B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-13.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI3A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SGSI3A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-7.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SGSI3B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SGSI3B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-10.0))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW1A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'TTIW1A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-1.2))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-1.2))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW1B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'TTIW1B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-1.4))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-1.4))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW2A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'TTIW2A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-0.6))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-0.6))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-0.6))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-0.6))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-0.6))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW2B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'TTIW2B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-0.75))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-0.75))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-0.75))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-0.75))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-0.75))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW3A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'TTIW3A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-0.7))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-0.7))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-0.7))
        [ig,ig_,_] = jtu.s2mpj_ii('TTIW3B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'TTIW3B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['TT3']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-0.9))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-0.9))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-0.9))
        [ig,ig_,_] = jtu.s2mpj_ii('SMF1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SMF1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MF1']])
        valA = jtu.append(valA,float(20.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW1']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SMF2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SMF2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MF2']])
        valA = jtu.append(valA,float(20.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SMF3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'SMF3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['MF3']])
        valA = jtu.append(valA,float(20.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['IW3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['INW']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI1A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SI1A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(-240.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI1B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SI1B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW1']])
        valA = jtu.append(valA,float(-290.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI2A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SI2A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-240.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI2B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SI2B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW2']])
        valA = jtu.append(valA,float(-290.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI3A',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'SI3A')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-340.0))
        [ig,ig_,_] = jtu.s2mpj_ii('SI3B',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SI3B')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PW3']])
        valA = jtu.append(valA,float(-375.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GLGCON',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'GLGCON')
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
        self.gconst = jtu.arrset(self.gconst,ig_['STA1'],float(247.963))
        self.gconst = jtu.arrset(self.gconst,ig_['STA2'],float(247.963))
        self.gconst = jtu.arrset(self.gconst,ig_['STA3'],float(32.591))
        self.gconst = jtu.arrset(self.gconst,ig_['INST'],float(35.5))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW1A'],float(24.0))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW1B'],float(28.0))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW2A'],float(12.0))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW2B'],float(15.0))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW3A'],float(14.0))
        self.gconst = jtu.arrset(self.gconst,ig_['TTIW3B'],float(18.0))
        self.gconst = jtu.arrset(self.gconst,ig_['SMF1'],float(20.0))
        self.gconst = jtu.arrset(self.gconst,ig_['SMF2'],float(20.0))
        self.gconst = jtu.arrset(self.gconst,ig_['SMF3'],float(20.0))
        self.gconst = jtu.arrset(self.gconst,ig_['GLGCON'],float(35000.0))
        #%%%%%%%%%%%%%%%%%%%%  RANGES %%%%%%%%%%%%%%%%%%%%%%
        grange = jnp.full((ngrp,1),None)
        grange = jtu.np_like_set(grange, legrps, jnp.full((self.nle,1),float('inf')))
        grange = jtu.np_like_set(grange, gegrps, jnp.full((self.nge,1),float('inf')))
        grange = jtu.arrset(grange,ig_['GLGCON'],float(15000.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),1.0e-8)
        self.xupper = jnp.full((self.n,1),1.0e+4)
        self.xlower = jtu.np_like_set(self.xlower, ix_['S1L'], 125.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['S1L'], 150.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['S2L'], 75.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['S2L'], 100.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['S3L'], 50.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['S3L'], 70.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['MF1'], 0.25)
        self.xupper = jtu.np_like_set(self.xupper, ix_['MF1'], 0.30)
        self.xlower = jtu.np_like_set(self.xlower, ix_['MF2'], 0.24)
        self.xupper = jtu.np_like_set(self.xupper, ix_['MF2'], 0.29)
        self.xlower = jtu.np_like_set(self.xlower, ix_['MF3'], 0.16)
        self.xupper = jtu.np_like_set(self.xupper, ix_['MF3'], 0.21)
        self.xlower = jtu.np_like_set(self.xlower, ix_['INW'], 2.5)
        self.xupper = jtu.np_like_set(self.xupper, ix_['INW'], 4.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['AW1'], float(68.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['IW1'], float(136.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MF1'], float(0.29988744))
        self.x0 = jtu.np_like_set(self.x0, ix_['TT1'], float(3733.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['PW1'], float(2177.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['ET1'], float(746.6))
        self.x0 = jtu.np_like_set(self.x0, ix_['S1L'], float(125.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['AW2'], float(28.2))
        self.x0 = jtu.np_like_set(self.x0, ix_['IW2'], float(47.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MF2'], float(0.28939109))
        self.x0 = jtu.np_like_set(self.x0, ix_['TT2'], float(480.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['PW2'], float(566.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['ET2'], float(96.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['S2L'], float(75.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['AW3'], float(11.2))
        self.x0 = jtu.np_like_set(self.x0, ix_['IW3'], float(16.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['MF3'], float(0.20980926))
        self.x0 = jtu.np_like_set(self.x0, ix_['TT3'], float(129.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['PW3'], float(145.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['ET3'], float(129.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['S3L'], float(50.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['INW'], float(2.5))
        self.x0 = jtu.np_like_set(self.x0, ix_['BT1'], float(155.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['BT2'], float(314.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['BT3'], float(403.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD1', iet_)
        elftv = jtu.loaset(elftv,it,0,'VA')
        elftv = jtu.loaset(elftv,it,1,'VB')
        elftv = jtu.loaset(elftv,it,2,'VC')
        elftv = jtu.loaset(elftv,it,3,'VD')
        elftv = jtu.loaset(elftv,it,4,'VE')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePOWER', iet_)
        elftv = jtu.loaset(elftv,it,0,'XX')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'PWR')
        elftp = jtu.loaset(elftp,it,1,'SC')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD2', iet_)
        elftv = jtu.loaset(elftv,it,0,'VA')
        elftv = jtu.loaset(elftv,it,1,'VB')
        elftv = jtu.loaset(elftv,it,2,'VC')
        elftv = jtu.loaset(elftv,it,3,'VD')
        [it,iet_,_] = jtu.s2mpj_ii( 'eX7Y', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y1')
        elftv = jtu.loaset(elftv,it,2,'Y2')
        elftv = jtu.loaset(elftv,it,3,'Y3')
        elftv = jtu.loaset(elftv,it,4,'Y4')
        elftv = jtu.loaset(elftv,it,5,'Y5')
        elftv = jtu.loaset(elftv,it,6,'Y6')
        elftv = jtu.loaset(elftv,it,7,'Y7')
        [it,iet_,_] = jtu.s2mpj_ii( 'eX5Y', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y1')
        elftv = jtu.loaset(elftv,it,2,'Y2')
        elftv = jtu.loaset(elftv,it,3,'Y3')
        elftv = jtu.loaset(elftv,it,4,'Y4')
        elftv = jtu.loaset(elftv,it,5,'Y5')
        [it,iet_,_] = jtu.s2mpj_ii( 'eX3Y', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y1')
        elftv = jtu.loaset(elftv,it,2,'Y2')
        elftv = jtu.loaset(elftv,it,3,'Y3')
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        [it,iet_,_] = jtu.s2mpj_ii( 'eBIG1', iet_)
        elftv = jtu.loaset(elftv,it,0,'LH')
        elftv = jtu.loaset(elftv,it,1,'TH')
        elftv = jtu.loaset(elftv,it,2,'LL')
        elftv = jtu.loaset(elftv,it,3,'V1')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'XPROD1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD1')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD1"])
        vname = 'AW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'TT1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VE')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XPF'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.146))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPG'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.648))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPROD2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD2')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD2"])
        vname = 'AW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'S1L'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XPL'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.736))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPM'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.229))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPROD3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD1')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD1"])
        vname = 'AW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'TT2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VE')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'X2PF'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.146))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'X2PG'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.648))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPROD4'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD2')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD2"])
        vname = 'AW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'S2L'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'X2PL'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.736))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'X2PM'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.229))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPROD5'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD1')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD1"])
        vname = 'AW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'TT3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VE')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XQF'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.539))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XQG'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.772))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1000.0))
        ename = 'XPROD6'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD2')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD2"])
        vname = 'AW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VA')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VB')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VC')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'S3L'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='VD')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XQL'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.33))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(100.0))
        ename = 'XQM'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
        vname = 'ET3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PWR')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.498))
        posep = jnp.where(elftp[ielftype[ie]]=='SC')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(100.0))
        ename = 'SMFE1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eX7Y')
        ielftype = jtu.arrset(ielftype,ie,iet_["eX7Y"])
        vname = 'MF1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y5')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y6')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'INW'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y7')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'SMFE2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eX5Y')
        ielftype = jtu.arrset(ielftype,ie,iet_["eX5Y"])
        vname = 'MF2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'INW'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y5')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'SMFE3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eX3Y')
        ielftype = jtu.arrset(ielftype,ie,iet_["eX3Y"])
        vname = 'MF3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'IW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'INW'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'TT1BT1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
        ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
        vname = 'TT1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'TT2BT2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
        ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
        vname = 'TT2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'TT3BT3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
        ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
        vname = 'TT3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XBIG11'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBIG1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBIG1"])
        vname = 'TT1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='TH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LL')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XBIG12'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBIG1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBIG1"])
        vname = 'TT2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='TH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LL')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XBIG13'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eBIG1')
        ielftype = jtu.arrset(ielftype,ie,iet_["eBIG1"])
        vname = 'TT3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'BT3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='TH')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PW3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='LL')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'MF3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-8),float(1.0e+4),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gSUMM',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['STA1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5272.77))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPF'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(160.909))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPG'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(282.874))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.64570846))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPL'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(31.136196))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPM'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(12.092112))
        ig = ig_['STA2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5272.77))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['X2PF'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(160.909))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['X2PG'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(282.874))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.64570846))
        ig = ig_['STA1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['X2PL'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(31.136196))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['X2PM'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(12.092112))
        ig = ig_['STA3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD5'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5272.77))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XQF'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(181.806))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XQG'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(232.57))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XPROD6'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.49783215))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XQL'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-0.22424514))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XQM'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(20.708238))
        ig = ig_['LAUN']
        self.grftype = jtu.arrset(self.grftype,ig,'gSUMM')
        ig = ig_['SMF1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SMFE1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SMF2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SMFE2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SMF3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SMFE3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI1A']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT1BT1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI1B']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT1BT1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI2A']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT2BT2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI2B']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT2BT2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI3A']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT3BT3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['SI3B']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TT3BT3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['GLGCON']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XBIG11'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-32.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XBIG12'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-32.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XBIG13'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-32.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.arange(self.nle), grange[legrps])
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nge), grange[gegrps])
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-MY-25-28"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eBIG1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        LG = jnp.log(EV_[3])
        f_   = (EV_[0]*EV_[1]*LG)/EV_[2]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 3, (EV_[0]*EV_[1])/(EV_[3]*EV_[2]))
            g_ = jtu.np_like_set(g_, 0, (EV_[1]*LG)/EV_[2])
            g_ = jtu.np_like_set(g_, 1, (EV_[0]*LG)/EV_[2])
            g_ = jtu.np_like_set(g_, 2, -(EV_[0]*EV_[1]*LG)/(EV_[2]**2))
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -(EV_[0]*EV_[1])/(EV_[2]*EV_[3]**2))
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), EV_[1]/(EV_[3]*EV_[2]))
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), H_[3,0])
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), EV_[0]/(EV_[3]*EV_[2]))
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), H_[3,1])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), -(EV_[0]*EV_[1])/(EV_[2]**2*EV_[3]))
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), H_[3,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), LG/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), -EV_[1]*LG/EV_[2]**2)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), -EV_[0]*LG/EV_[2]**2)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 2.0*(EV_[0]*EV_[1]*LG)/(EV_[2]**3.0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePROD1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EA = 1.2781
        VA0 = EV_[0]**EA
        VA1 = EA*EV_[0]**(EA-1.0)
        VA2 = EA*(EA-1.0)*EV_[0]**(EA-2.0)
        EB = -0.1959
        VB0 = EV_[1]**EB
        VB1 = EB*EV_[1]**(EB-1.0)
        VB2 = EB*(EB-1.0)*EV_[1]**(EB-2.0)
        EC = 2.4242
        VC0 = EV_[2]**EC
        VC1 = EC*EV_[2]**(EC-1.0)
        VC2 = EC*(EC-1.0)*EV_[2]**(EC-2.0)
        ED = 0.38745
        VD0 = EV_[3]**ED
        VD1 = ED*EV_[3]**(ED-1.0)
        VD2 = ED*(ED-1.0)*EV_[3]**(ED-2.0)
        EE = 0.9904
        VE0 = EV_[4]**EE
        VE1 = EE*EV_[4]**(EE-1.0)
        VE2 = EE*(EE-1.0)*EV_[4]**(EE-2.0)
        f_   = VA0*VB0*VC0*VD0*VE0
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, VA1*VB0*VC0*VD0*VE0)
            g_ = jtu.np_like_set(g_, 1, VA0*VB1*VC0*VD0*VE0)
            g_ = jtu.np_like_set(g_, 2, VA0*VB0*VC1*VD0*VE0)
            g_ = jtu.np_like_set(g_, 3, VA0*VB0*VC0*VD1*VE0)
            g_ = jtu.np_like_set(g_, 4, VA0*VB0*VC0*VD0*VE1)
            if nargout>2:
                H_ = jnp.zeros((5,5))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), VA2*VB0*VC0*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), VA1*VB1*VC0*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), VA1*VB0*VC1*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), VA1*VB0*VC0*VD1*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), VA1*VB0*VC0*VD0*VE1)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), VA0*VB2*VC0*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), VA0*VB1*VC1*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), VA0*VB1*VC0*VD1*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), VA0*VB1*VC0*VD0*VE1)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), VA0*VB0*VC2*VD0*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), VA0*VB0*VC1*VD1*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), VA0*VB0*VC1*VD0*VE1)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), VA0*VB0*VC0*VD2*VE0)
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), VA0*VB0*VC0*VD1*VE1)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), VA0*VB0*VC0*VD0*VE2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePROD2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EA = 0.3322
        VA0 = EV_[0]**EA
        VA1 = EA*EV_[0]**(EA-1.0)
        VA2 = EA*(EA-1.0)*EV_[0]**(EA-2.0)
        EB = -1.5935
        VB0 = EV_[1]**EB
        VB1 = EB*EV_[1]**(EB-1.0)
        VB2 = EB*(EB-1.0)*EV_[1]**(EB-2.0)
        EC = 0.2363
        VC0 = EV_[2]**EC
        VC1 = EC*EV_[2]**(EC-1.0)
        VC2 = EC*(EC-1.0)*EV_[2]**(EC-2.0)
        ED = 0.1079
        VD0 = EV_[3]**ED
        VD1 = ED*EV_[3]**(ED-1.0)
        VD2 = ED*(ED-1.0)*EV_[3]**(ED-2.0)
        f_   = VA0*VB0*VC0*VD0
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, VA1*VB0*VC0*VD0)
            g_ = jtu.np_like_set(g_, 1, VA0*VB1*VC0*VD0)
            g_ = jtu.np_like_set(g_, 2, VA0*VB0*VC1*VD0)
            g_ = jtu.np_like_set(g_, 3, VA0*VB0*VC0*VD1)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), VA2*VB0*VC0*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), VA1*VB1*VC0*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), VA1*VB0*VC1*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), VA1*VB0*VC0*VD1)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), VA0*VB2*VC0*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), VA0*VB1*VC1*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), VA0*VB1*VC0*VD1)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), VA0*VB0*VC2*VD0)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), VA0*VB0*VC1*VD1)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), VA0*VB0*VC0*VD2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePOWER(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        SCPWR = self.elpar[iel_][0]/(self.elpar[iel_][1]**self.elpar[iel_][0])
        f_   = (EV_[0]/self.elpar[iel_][1])**self.elpar[iel_][0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, SCPWR*EV_[0]**(self.elpar[iel_][0]-1.0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), SCPWR*(self.elpar[iel_][0]-1.0)*EV_[0]**(self.elpar[iel_][0]-2.0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eX7Y(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,8))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,4]), U_[1,4]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,5]), U_[1,5]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,6]), U_[1,6]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,7]), U_[1,7]+1)
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

    @staticmethod
    def eX5Y(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,6))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,4]), U_[1,4]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,5]), U_[1,5]+1)
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

    @staticmethod
    def eX3Y(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,4))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
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

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
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

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSUMM(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_**0.460
        if nargout>1:
            g_ = 0.460*GVAR_**(-0.540)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = -0.2484*GVAR_**(-1.540)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

