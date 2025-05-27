from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class NET1:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : NET1
#    *********
# 
#    A gas network problem for the south-east of England.
# 
#     SIF input: Sybille Schachler, Oxford, August 1992.
#    classification = "C-COOI2-RN-48-57"
# 
#    ...Problem size parameters
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'NET1'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['NNOD'] = 22
        v_['NPIP'] = 17
        v_['NCMP'] = 3
        v_['NSRC'] = 2
        v_['CSTART'] = 18
        v_['CEND'] = 20
        v_['SSTART'] = 21
        v_['SEND'] = 22
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('NOP17',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP17')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL11',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL11')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP9',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP9')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL10',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL10')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP16',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP16')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL16',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL16')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP19',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP19')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL17',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL17')
        [iv,ix_,_] = jtu.s2mpj_ii('SFL22',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'SFL22')
        [iv,ix_,_] = jtu.s2mpj_ii('SBV22',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'SBV22')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP18',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP18')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP4')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL13',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL13')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL5')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP11',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP11')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL8')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP6',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP6')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP12',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP12')
        [iv,ix_,_] = jtu.s2mpj_ii('CFL19',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CFL19')
        [iv,ix_,_] = jtu.s2mpj_ii('CBV19',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CBV19')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL7',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL7')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP5')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL6',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL6')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP8')
        [iv,ix_,_] = jtu.s2mpj_ii('CFL20',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CFL20')
        [iv,ix_,_] = jtu.s2mpj_ii('CBV20',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CBV20')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP7',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP7')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL9',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL9')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP21',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP21')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL2')
        [iv,ix_,_] = jtu.s2mpj_ii('SFL21',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'SFL21')
        [iv,ix_,_] = jtu.s2mpj_ii('SBV21',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'SBV21')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP1')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL1')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP14',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP14')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL12',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL12')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP10',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP10')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL3')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP2')
        [iv,ix_,_] = jtu.s2mpj_ii('CFL18',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CFL18')
        [iv,ix_,_] = jtu.s2mpj_ii('CBV18',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'CBV18')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP3')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL4')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP15',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP15')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL15',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL15')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP20',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP20')
        [iv,ix_,_] = jtu.s2mpj_ii('PFL14',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PFL14')
        [iv,ix_,_] = jtu.s2mpj_ii('NOP13',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'NOP13')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('MBE1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL1']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL2']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL21']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL3']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL18']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL4']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL18']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL5']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL22']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL6']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL7']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL19']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE6')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL8']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL19']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE7',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE7')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL9']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL20']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE8',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE8')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL6']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL20']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE9',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE9')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL10']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL11']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE10',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE10')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL3']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL12']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE11',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE11')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL5']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL8']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL13']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE12',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE12')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL7']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE13',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE13')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL14']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE14',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE14')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL1']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL12']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE15',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE15')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL4']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL15']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE16',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE16')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL10']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL16']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE17',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE17')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL11']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE18',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL13']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL17']])
        valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE19',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL16']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL17']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE20',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL14']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL15']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MBE21',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBE21')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL2']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PFL9']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('MCR18',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'MCR18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP3']])
        valA = jtu.append(valA,float(1.00000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP2']])
        valA = jtu.append(valA,float(-1.40000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV18']])
        valA = jtu.append(valA,float(0.00000))
        [ig,ig_,_] = jtu.s2mpj_ii('MCR19',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'MCR19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP6']])
        valA = jtu.append(valA,float(1.00000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP5']])
        valA = jtu.append(valA,float(-1.40000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV19']])
        valA = jtu.append(valA,float(0.00000))
        [ig,ig_,_] = jtu.s2mpj_ii('MCR20',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'MCR20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP8']])
        valA = jtu.append(valA,float(1.00000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP7']])
        valA = jtu.append(valA,float(-1.40000000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV20']])
        valA = jtu.append(valA,float(0.00000))
        [ig,ig_,_] = jtu.s2mpj_ii('CLF18',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLF18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV18']])
        valA = jtu.append(valA,float(1.00000e+04))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL18']])
        valA = jtu.append(valA,float(-1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('CLF19',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLF19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV19']])
        valA = jtu.append(valA,float(1.00000e+04))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL19']])
        valA = jtu.append(valA,float(-1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('CLF20',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLF20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV20']])
        valA = jtu.append(valA,float(1.00000e+04))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CFL20']])
        valA = jtu.append(valA,float(-1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('SLF21',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SLF21')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SBV21']])
        valA = jtu.append(valA,float(0.00000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL21']])
        valA = jtu.append(valA,float(-1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('SUF21',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SUF21')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SBV21']])
        valA = jtu.append(valA,float(-3.00000e+03))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL21']])
        valA = jtu.append(valA,float(1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('SLF22',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SLF22')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SBV22']])
        valA = jtu.append(valA,float(0.00000))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL22']])
        valA = jtu.append(valA,float(-1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('SUF22',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'SUF22')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SBV22']])
        valA = jtu.append(valA,float(-1.06000e+02))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SFL22']])
        valA = jtu.append(valA,float(1.00000000))
        [ig,ig_,_] = jtu.s2mpj_ii('CLP18',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLP18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP3']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP2']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV18']])
        valA = jtu.append(valA,float(-4.72000e+02))
        [ig,ig_,_] = jtu.s2mpj_ii('CUP18',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CUP18')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP3']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP2']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('CLP19',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLP19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP6']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP5']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV19']])
        valA = jtu.append(valA,float(-3.45000e+02))
        [ig,ig_,_] = jtu.s2mpj_ii('CUP19',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CUP19')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP6']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP5']])
        valA = jtu.append(valA,float(1))
        [ig,ig_,_] = jtu.s2mpj_ii('CLP20',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CLP20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP8']])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP7']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['CBV20']])
        valA = jtu.append(valA,float(-5.75000e+02))
        [ig,ig_,_] = jtu.s2mpj_ii('CUP20',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'CUP20')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP8']])
        valA = jtu.append(valA,float(-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['NOP7']])
        valA = jtu.append(valA,float(1))
        for i in range(int(v_['1']),int(v_['NPIP'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('PDE'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'PDE'+str(i))
            self.gscale = jtu.arrset(self.gscale,ig,float(1.00000e+03))
        for i in range(int(v_['CSTART']),int(v_['CEND'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('HPCON'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'HPCON'+str(i))
            self.gscale = jtu.arrset(self.gscale,ig,float(70.00000000))
        for i in range(int(v_['CSTART']),int(v_['CEND'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('HPOBJ'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(0.03500000))
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
        self.gconst = jtu.arrset(self.gconst,ig_['MBE1'],float(6.65680000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE4'],float(1.96100000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE9'],float(3.72060e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE10'],float(47.17000000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE11'],float(1.60060e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE12'],float(4.25060e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE13'],float(5.30000e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE14'],float(24.16800000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE15'],float(2.54400000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE16'],float(89.14600000))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE17'],float(4.92900e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE20'],float(4.64280e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['MBE21'],float(1.48400e+02))
        self.gconst = jtu.arrset(self.gconst,ig_['CLF18'],float(1.00000e+04))
        self.gconst = jtu.arrset(self.gconst,ig_['CLF19'],float(1.00000e+04))
        self.gconst = jtu.arrset(self.gconst,ig_['CLF20'],float(1.00000e+04))
        self.gconst = jtu.arrset(self.gconst,ig_['HPCON18'],float(2.07000e+04))
        self.gconst = jtu.arrset(self.gconst,ig_['HPCON19'],float(2.07000e+04))
        self.gconst = jtu.arrset(self.gconst,ig_['HPCON20'],float(4.14000e+04))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['SFL21'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['SFL21'], 3.00000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['SFL22'], 0.00000)
        self.xupper = jtu.np_like_set(self.xupper, ix_['SFL22'], 1.06000e+02)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP1'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP1'], 1.01500e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP2'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP2'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP3'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP3'], 9.72000e+02)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP4'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP4'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP5'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP5'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP6'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP6'], 8.45000e+02)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP7'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP7'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP8'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP8'], 1.07500e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP9'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP9'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP10'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP10'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP11'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP11'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP12'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP12'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP13'], 5.80000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP13'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP14'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP14'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP15'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP15'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP16'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP16'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP17'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP17'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP18'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP18'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP19'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP19'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP20'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP20'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['NOP21'], 5.00000e+02)
        self.xupper = jtu.np_like_set(self.xupper, ix_['NOP21'], 1.10000e+03)
        self.xlower = jtu.np_like_set(self.xlower, ix_['CBV18'], 0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['CBV18'], 0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['CBV19'], 1)
        self.xupper = jtu.np_like_set(self.xupper, ix_['CBV19'], 1)
        self.xlower = jtu.np_like_set(self.xlower, ix_['CBV20'], 1)
        self.xupper = jtu.np_like_set(self.xupper, ix_['CBV20'], 1)
        self.xlower = jtu.np_like_set(self.xlower, ix_['SBV21'], 1)
        self.xupper = jtu.np_like_set(self.xupper, ix_['SBV21'], 1)
        self.xlower = jtu.np_like_set(self.xlower, ix_['SBV22'], 1)
        self.xupper = jtu.np_like_set(self.xupper, ix_['SBV22'], 1)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(5.00000e+02))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eA0PANHAN', iet_)
        elftv = jtu.loaset(elftv,it,0,'PIN')
        elftv = jtu.loaset(elftv,it,1,'POUT')
        elftv = jtu.loaset(elftv,it,2,'FLOW')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'PIPRES')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA1MAXHP', iet_)
        elftv = jtu.loaset(elftv,it,0,'PIN')
        elftv = jtu.loaset(elftv,it,1,'POUT')
        elftv = jtu.loaset(elftv,it,2,'FLOW')
        elftv = jtu.loaset(elftv,it,3,'CBV')
        elftp = jtu.loaset(elftp,it,0,'IPL')
        elftp = jtu.loaset(elftp,it,1,'OPL')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA2HPFUN', iet_)
        elftv = jtu.loaset(elftv,it,0,'PIN')
        elftv = jtu.loaset(elftv,it,1,'POUT')
        elftv = jtu.loaset(elftv,it,2,'FLOW')
        elftv = jtu.loaset(elftv,it,3,'CBV')
        elftp = jtu.loaset(elftp,it,0,'IPL')
        elftp = jtu.loaset(elftp,it,1,'OPL')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'PANH1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP14'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.62131268))
        ename = 'PANH2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP21'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.31605264))
        ename = 'PANH3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.13104611))
        ename = 'PANH4'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP15'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.12796251))
        ename = 'PANH5'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP11'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.78624623))
        ename = 'PANH6'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.84948702))
        ename = 'PANH7'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP12'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.13696026))
        ename = 'PANH8'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP11'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.25900862))
        ename = 'PANH9'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP21'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.32838618))
        ename = 'PANH10'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP16'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.33657520))
        ename = 'PANH11'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP17'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL11'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.61512113))
        ename = 'PANH12'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP14'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL12'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.51339271))
        ename = 'PANH13'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP11'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL13'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.20890923))
        ename = 'PANH14'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP13'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL14'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.15474706))
        ename = 'PANH15'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP15'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL15'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.26980036))
        ename = 'PANH16'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP16'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL16'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.04255562))
        ename = 'PANH17'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA0PANHAN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA0PANHAN"])
        vname = 'NOP18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PFL17'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='PIPRES')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.12570329))
        ename = 'HPMAX18'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA1MAXHP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA1MAXHP"])
        vname = 'NOP2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        ename = 'HPMAX19'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA1MAXHP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA1MAXHP"])
        vname = 'NOP5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        ename = 'HPMAX20'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA1MAXHP')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA1MAXHP"])
        vname = 'NOP7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        ename = 'HPFUN18'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA2HPFUN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA2HPFUN"])
        vname = 'NOP2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV18'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        ename = 'HPFUN19'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA2HPFUN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA2HPFUN"])
        vname = 'NOP5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV19'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        ename = 'HPFUN20'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eA2HPFUN')
        ielftype = jtu.arrset(ielftype,ie,iet_["eA2HPFUN"])
        vname = 'NOP7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='PIN')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'NOP8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='POUT')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CFL20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='FLOW')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'CBV20'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(5.00000e+02))
        posev = jnp.where(elftv[ielftype[ie]]=='CBV')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='IPL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        posep = jnp.where(elftp[ielftype[ie]]=='OPL')[0]
        jtu.loaset(self.elpar,ie,posep[0],float(0.00000))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for i in range(int(v_['1']),int(v_['NPIP'])+1):
            ig = ig_['PDE'+str(i)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PANH'+str(i)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for i in range(int(v_['CSTART']),int(v_['CEND'])+1):
            ig = ig_['HPCON'+str(i)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['HPMAX'+str(i)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for i in range(int(v_['CSTART']),int(v_['CEND'])+1):
            ig = ig_['HPOBJ'+str(i)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['HPFUN'+str(i)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOI2-RN-48-57"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eA0PANHAN(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A0FLEX = 1.8539e0
        A0FGT0 = EV_[2]>=0.0e0
        if A0FGT0!=0:
            A0HFLO  = (
              -self.elpar[iel_][0]*A0FLEX*(A0FLEX-1.0e0)*EV_[2]**(A0FLEX-2.0e0))
        if A0FGT0==0:
            A0HFLO  = (
              A0FLEX*(A0FLEX-1.0e0)*self.elpar[iel_][0]*jnp.absolute(EV_[2])**(A0FLEX-2.0e0))
        f_    = (
              EV_[0]*EV_[0]-EV_[1]*EV_[1]-self.elpar[iel_][0]*EV_[2]*jnp.absolute(EV_[2])**(A0FLEX-1.0e0))
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, -2.0e0*EV_[1])
            g_ = jtu.np_like_set(g_, 0, 2.0e0*EV_[0])
            g_ = jtu.np_like_set(g_, 2, -self.elpar[iel_][0]*A0FLEX*jnp.absolute(EV_[2])**(A0FLEX-1.0e0))
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -2.0e0)
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e0)
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), A0HFLO)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA1MAXHP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A1BETA = 0.23077e0
        A1HFAC = 203.712e0
        A1PSUC = EV_[0]-self.elpar[iel_][0]*EV_[3]
        A1PDIS = EV_[1]+self.elpar[iel_][1]*EV_[3]
        A1CRB = (A1PDIS/A1PSUC)**A1BETA
        A1PROD = A1BETA*A1HFAC*A1CRB*EV_[2]
        A1GPIN = -A1PROD/A1PSUC
        A1GPOU = A1PROD/A1PDIS
        A1GFLO = A1HFAC*(A1CRB-1.0e0)
        A1GCBV = -self.elpar[iel_][0]*A1GPIN+self.elpar[iel_][1]*A1GPOU
        A1HII = A1PROD*(A1BETA+1.0e0)/(A1PSUC**2)
        A1HIO = -A1PROD*A1BETA/(A1PSUC*A1PDIS)
        A1HOO = A1PROD*(A1BETA-1.0e0)/(A1PDIS**2)
        A1HIC = -self.elpar[iel_][0]*A1HII+self.elpar[iel_][1]*A1HIO
        A1HOC = -self.elpar[iel_][0]*A1HIO+self.elpar[iel_][1]*A1HOO
        f_   = EV_[2]*A1GFLO
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, A1GPIN)
            g_ = jtu.np_like_set(g_, 1, A1GPOU)
            g_ = jtu.np_like_set(g_, 2, A1GFLO)
            g_ = jtu.np_like_set(g_, 3, A1GCBV)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), A1HII)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), A1HIO)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), A1HOO)
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), A1GPIN/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), A1GPOU/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), A1HIC)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), A1HOC)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), A1GCBV/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -self.elpar[iel_][0]*A1HIC+self.elpar[iel_][1]*A1HOC)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA2HPFUN(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A2BETA = 0.23077e0
        A2HFAC = 203.712e0
        A2PSUC = EV_[0]-self.elpar[iel_][0]*EV_[3]
        A2PDIS = EV_[1]+self.elpar[iel_][1]*EV_[3]
        A2CRB = (A2PDIS/A2PSUC)**A2BETA
        A2PROD = A2BETA*A2HFAC*A2CRB*EV_[2]
        A2GPIN = -A2PROD/A2PSUC
        A2GPOU = A2PROD/A2PDIS
        A2GFLO = A2HFAC*(A2CRB-1.0e0)
        A2GCBV = -self.elpar[iel_][0]*A2GPIN+self.elpar[iel_][1]*A2GPOU
        A2HII = A2PROD*(A2BETA+1.0e0)/(A2PSUC**2)
        A2HIO = -A2PROD*A2BETA/(A2PSUC*A2PDIS)
        A2HOO = A2PROD*(A2BETA-1.0e0)/(A2PDIS**2)
        A2HIC = -self.elpar[iel_][0]*A2HII+self.elpar[iel_][1]*A2HIO
        A2HOC = -self.elpar[iel_][0]*A2HIO+self.elpar[iel_][1]*A2HOO
        f_   = EV_[2]*A2GFLO
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, A2GPIN)
            g_ = jtu.np_like_set(g_, 1, A2GPOU)
            g_ = jtu.np_like_set(g_, 2, A2GFLO)
            g_ = jtu.np_like_set(g_, 3, A2GCBV)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), A2HII)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), A2HIO)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), A2HOO)
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), A2GPIN/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), A2GPOU/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), A2HIC)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), A2HOC)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), A2GCBV/EV_[2])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -self.elpar[iel_][0]*A2HIC+self.elpar[iel_][1]*A2HOC)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

