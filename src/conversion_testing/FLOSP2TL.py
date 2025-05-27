from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class FLOSP2TL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : FLOSP2TL
#    *********
# 
#    A  two-dimensional base  flow  problem in an inclined enclosure.
# 
#    Temperature constant at y = +/- 1 boundary conditions
#    Low Reynold's number
# 
#    The flow is considered in a square of length 2,  centered on the
#    origin and aligned with the x-y axes. The square is divided into
#    4 n ** 2  sub-squares,  each of  length 1 / n.  The differential
#    equation is replaced by  discrete nonlinear equations at each of 
#    the grid points. 
# 
#    The differential equation relates the vorticity, temperature and
#    a stream function.
#    
#    Source: 
#    J. N. Shadid
#    "Experimental and computational study of the stability
#    of Natural convection flow in an inclined enclosure",
#    Ph. D. Thesis, University of Minnesota, 1989,
#    problem SP2 (pp.128-130), 
# 
#    SIF input: Nick Gould, August 1993.
# 
#    classification = "C-CNQR2-MY-V-V"
# 
#    Half the number of discretization intervals
#    Number of variables = 3(2M+1)**2 
# 
#           Alternative values for the SIF file parameters:
# IE M                   1              $-PARAMETER n=27
# IE M                   2              $-PARAMETER n=75
# IE M                   5              $-PARAMETER n=363     original value
# IE M                   8              $-PARAMETER n=867
# IE M                   10             $-PARAMETER n=1323
# IE M                   15             $-PARAMETER n=2883
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'FLOSP2TL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['M'] = int(1);  #  SIF file default value
        else:
            v_['M'] = int(args[0])
        if nargin<2:
            v_['RA'] = float(1.0e+3);  #  SIF file default value
        else:
            v_['RA'] = float(args[1])
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['AX'] = 1.0
        v_['THETA'] = 0.5*v_['PI']
        v_['A1'] = 0.0
        v_['A2'] = 1.0
        v_['A3'] = 0.0
        v_['B1'] = 0.0
        v_['B2'] = 1.0
        v_['B3'] = 1.0
        v_['F1'] = 1.0
        v_['F2'] = 0.0
        v_['F3'] = 0.0
        v_['G1'] = 1.0
        v_['G2'] = 0.0
        v_['G3'] = 0.0
        v_['M-1'] = -1+v_['M']
        v_['-M'] = -1*v_['M']
        v_['-M+1'] = -1*v_['M-1']
        v_['1/H'] = float(v_['M'])
        v_['-1/H'] = -1.0*v_['1/H']
        v_['2/H'] = 2.0*v_['1/H']
        v_['-2/H'] = -2.0*v_['1/H']
        v_['H'] = 1.0/v_['1/H']
        v_['H2'] = v_['H']*v_['H']
        v_['1/H2'] = v_['1/H']*v_['1/H']
        v_['-2/H2'] = -2.0*v_['1/H2']
        v_['1/2H'] = 0.5*v_['1/H']
        v_['-1/2H'] = -0.5*v_['1/H']
        v_['AXX'] = v_['AX']*v_['AX']
        v_['SINTHETA'] = jnp.sin(v_['THETA'])
        v_['COSTHETA'] = jnp.cos(v_['THETA'])
        v_['PI1'] = v_['AX']*v_['RA']
        v_['PI1'] = v_['PI1']*v_['COSTHETA']
        v_['PI1'] = -0.5*v_['PI1']
        v_['-PI1'] = -1.0*v_['PI1']
        v_['PI2'] = v_['AXX']*v_['RA']
        v_['PI2'] = v_['PI2']*v_['SINTHETA']
        v_['PI2'] = 0.5*v_['PI2']
        v_['-PI2'] = -1.0*v_['PI2']
        v_['2A1'] = 2.0*v_['A1']
        v_['2B1'] = 2.0*v_['B1']
        v_['2F1'] = 2.0*v_['F1']
        v_['2G1'] = 2.0*v_['G1']
        v_['2F1/AX'] = v_['2F1']/v_['AX']
        v_['2G1/AX'] = v_['2G1']/v_['AX']
        v_['AX/2'] = 0.5*v_['AX']
        v_['AXX/2'] = 0.5*v_['AXX']
        v_['AXX/4'] = 0.25*v_['AXX']
        v_['2AX'] = 2.0*v_['AX']
        v_['2AXX'] = 2.0*v_['AXX']
        v_['2/AX'] = 2.0/v_['AX']
        v_['2/AXH'] = v_['2/H']/v_['AX']
        v_['-2/AXH'] = -1.0*v_['2/AXH']
        v_['PI1/2H'] = v_['PI1']*v_['1/2H']
        v_['-PI1/2H'] = v_['PI1']*v_['-1/2H']
        v_['PI2/2H'] = v_['PI2']*v_['1/2H']
        v_['-PI2/2H'] = v_['PI2']*v_['-1/2H']
        v_['2A1/H'] = v_['2A1']*v_['1/H']
        v_['-2A1/H'] = v_['2A1']*v_['-1/H']
        v_['2B1/H'] = v_['2B1']*v_['1/H']
        v_['-2B1/H'] = v_['2B1']*v_['-1/H']
        v_['2F1/AXH'] = v_['2F1/AX']*v_['1/H']
        v_['-2F1/AXH'] = v_['2F1/AX']*v_['-1/H']
        v_['2G1/AXH'] = v_['2G1/AX']*v_['1/H']
        v_['-2G1/AXH'] = v_['2G1/AX']*v_['-1/H']
        v_['AX/H2'] = v_['AX']*v_['1/H2']
        v_['-AX/H2'] = -1.0*v_['AX/H2']
        v_['AX/4H2'] = 0.25*v_['AX/H2']
        v_['-AX/4H2'] = -0.25*v_['AX/H2']
        v_['AXX/H2'] = v_['AXX']*v_['1/H2']
        v_['-2AXX/H2'] = -2.0*v_['AXX/H2']
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
        for J in range(int(v_['-M']),int(v_['M'])+1):
            for I in range(int(v_['-M']),int(v_['M'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('OM'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'OM'+str(I)+','+str(J))
                [iv,ix_,_] = jtu.s2mpj_ii('PH'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'PH'+str(I)+','+str(J))
                [iv,ix_,_] = jtu.s2mpj_ii('PS'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'PS'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for J in range(int(v_['-M+1']),int(v_['M-1'])+1):
            v_['J+'] = 1+J
            v_['J-'] = -1+J
            for I in range(int(v_['-M+1']),int(v_['M-1'])+1):
                v_['I+'] = 1+I
                v_['I-'] = -1+I
                [ig,ig_,_] = jtu.s2mpj_ii('S'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'S'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(int(v_['I+']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(int(v_['I-']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(I)+','+str(int(v_['J+']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(I)+','+str(int(v_['J-']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(int(v_['I+']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['-PI1/2H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(int(v_['I-']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['PI1/2H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(int(v_['J+']))]])
                valA = jtu.append(valA,float(v_['-PI2/2H']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(int(v_['J-']))]])
                valA = jtu.append(valA,float(v_['PI2/2H']))
                [ig,ig_,_] = jtu.s2mpj_ii('V'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'V'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(int(v_['I+']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(int(v_['I-']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(I)+','+str(int(v_['J+']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PS'+str(I)+','+str(int(v_['J-']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['OM'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['AXX/4']))
                [ig,ig_,_] = jtu.s2mpj_ii('E'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'E'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(int(v_['I+']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(int(v_['I-']))+','+str(J)]])
                valA = jtu.append(valA,float(v_['1/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(v_['-2AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(int(v_['J+']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['PH'+str(I)+','+str(int(v_['J-']))]])
                valA = jtu.append(valA,float(v_['AXX/H2']))
        for K in range(int(v_['-M']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['M']))]])
            valA = jtu.append(valA,float(v_['2A1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['M-1']))]])
            valA = jtu.append(valA,float(v_['-2A1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['M']))]])
            valA = jtu.append(valA,float(v_['A2']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['-M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['-M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['-M+1']))]])
            valA = jtu.append(valA,float(v_['2B1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['-M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['-M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['-M']))]])
            valA = jtu.append(valA,float(v_['-2B1/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(K)+','+str(int(v_['-M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(K)+','+str(int(v_['-M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(K)+','+str(int(v_['-M']))]])
            valA = jtu.append(valA,float(v_['B2']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['2F1/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['M-1']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['-2F1/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['F2']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['-M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['-M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['-M+1']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['2G1/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['-M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['-M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['-M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['-2G1/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('T'+str(int(v_['-M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'T'+str(int(v_['-M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PH'+str(int(v_['-M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['G2']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(K)+','+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(K)+','+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(K)+','+str(int(v_['M']))]])
            valA = jtu.append(valA,float(v_['-2/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(K)+','+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(K)+','+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(K)+','+str(int(v_['M-1']))]])
            valA = jtu.append(valA,float(v_['2/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(K)+','+str(int(v_['-M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(K)+','+str(int(v_['-M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(K)+','+str(int(v_['-M+1']))]])
            valA = jtu.append(valA,float(v_['2/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(K)+','+str(int(v_['-M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(K)+','+str(int(v_['-M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(K)+','+str(int(v_['-M']))]])
            valA = jtu.append(valA,float(v_['-2/H']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(int(v_['M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(int(v_['M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(int(v_['M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['-2/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(int(v_['M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(int(v_['M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(int(v_['M-1']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['2/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(int(v_['-M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(int(v_['-M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(int(v_['-M+1']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['2/AXH']))
            [ig,ig_,_] = jtu.s2mpj_ii('V'+str(int(v_['-M']))+','+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'V'+str(int(v_['-M']))+','+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['PS'+str(int(v_['-M']))+','+str(K)]])
            valA = jtu.append(valA,float(v_['-2/AXH']))
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
        for K in range(int(v_['-M']),int(v_['M'])+1):
            self.gconst  = (                   jtu.arrset(self.gconst,ig_['T'+str(K)+','+str(int(v_['M']))],float(v_['A3'])))
            self.gconst  = (                   jtu.arrset(self.gconst,ig_['T'+str(K)+','+str(int(v_['-M']))],float(v_['B3'])))
            self.gconst  = (                   jtu.arrset(self.gconst,ig_['T'+str(int(v_['M']))+','+str(K)],float(v_['F3'])))
            self.gconst  = (                   jtu.arrset(self.gconst,ig_['T'+str(int(v_['-M']))+','+str(K)],float(v_['G3'])))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        for K in range(int(v_['-M']),int(v_['M'])+1):
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['PS'+str(K)+','+str(int(v_['-M']))]]), 1.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['PS'+str(K)+','+str(int(v_['-M']))]]), 1.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['PS'+str(int(v_['-M']))+','+str(K)]]), 1.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['PS'+str(int(v_['-M']))+','+str(K)]]), 1.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['PS'+str(K)+','+str(int(v_['M']))]]), 1.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['PS'+str(K)+','+str(int(v_['M']))]]), 1.0)
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['PS'+str(int(v_['M']))+','+str(K)]]), 1.0)
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['PS'+str(int(v_['M']))+','+str(K)]]), 1.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'PSIM')
        elftv = jtu.loaset(elftv,it,1,'PSIP')
        elftv = jtu.loaset(elftv,it,2,'PHIM')
        elftv = jtu.loaset(elftv,it,3,'PHIP')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for J in range(int(v_['-M+1']),int(v_['M-1'])+1):
            v_['J+'] = 1+J
            v_['J-'] = -1+J
            for I in range(int(v_['-M+1']),int(v_['M-1'])+1):
                v_['I+'] = 1+I
                v_['I-'] = -1+I
                ename = 'E'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
                ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
                self.x0 = jnp.zeros((self.n,1))
                vname = 'PS'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PSIP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PS'+str(I)+','+str(int(v_['J-']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PSIM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PH'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PHIP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PH'+str(int(v_['I-']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PHIM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'F'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
                ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
                vname = 'PS'+str(int(v_['I+']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PSIP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PS'+str(int(v_['I-']))+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PSIM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PH'+str(I)+','+str(int(v_['J+']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PHIP')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'PH'+str(I)+','+str(int(v_['J-']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='PHIM')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for J in range(int(v_['-M+1']),int(v_['M-1'])+1):
            for I in range(int(v_['-M+1']),int(v_['M-1'])+1):
                ig = ig_['E'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-AX/4H2']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['F'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['AX/4H2']))
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
        self.pbclass   = "C-CNQR2-MY-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,4))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[0]*IV_[1]
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

