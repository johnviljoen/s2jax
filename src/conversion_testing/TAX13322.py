from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class TAX13322:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : TAX13322
#    --------
# 
#    An optimal income tax model with multidimensional taxpayer types,
#    due to Judd, Ma, Saunders & Su
# 
#    Source:
#    Kenneth L. Judd, Ma,  Michael A. Saunders and Che-Lin Su
#    "Optimal Income Taxation with Multidimensional Taxpayer Types"
#    Working Paper, Hoover Institute, Stanford University, 2017
# 
#    SIF input: Nick Gould, July 2018 based on the AMPL model pTAX5Dncl
# 
#    "If ever there was an example that exhibited the stupidity of SIF,
#     this is it. NIMG"
# 
#    classification = "C-COOR2-MN-72-1261"
# 
#    parameters
# 
#           Alternative values for the SIF file parameters:
# IE NA                  1              $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'TAX13322'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['NA'] = int(1);  #  SIF file default value
        else:
            v_['NA'] = int(args[0])
# IE NB                  3              $-PARAMETER
        if nargin<2:
            v_['NB'] = int(3);  #  SIF file default value
        else:
            v_['NB'] = int(args[1])
# IE NC                  3              $-PARAMETER
        if nargin<3:
            v_['NC'] = int(3);  #  SIF file default value
        else:
            v_['NC'] = int(args[2])
# IE ND                  2              $-PARAMETER
        if nargin<4:
            v_['ND'] = int(2);  #  SIF file default value
        else:
            v_['ND'] = int(args[3])
# IE NE                  2              $-PARAMETER
        if nargin<5:
            v_['NE'] = int(2);  #  SIF file default value
        else:
            v_['NE'] = int(args[4])
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['6'] = 6
        v_['ONE'] = 1.0e0
        v_['TWO'] = 2.0e0
        v_['THREE'] = 3.0e0
        v_['NBD'] = v_['NB']*v_['ND']
        v_['NCE'] = v_['NC']*v_['NE']
        v_['NP'] = v_['NBD']*v_['NCE']
        v_['NP'] = v_['NP']*v_['NA']
        v_['NPM1'] = -1+v_['NP']
        v_['M'] = v_['NP']*v_['NPM1']
        v_['OMEGA1'] = v_['ONE']/v_['TWO']
        v_['OMEGA2'] = v_['TWO']/v_['THREE']
        v_['THETA1'] = v_['ONE']/v_['THREE']
        v_['THETA2'] = v_['ONE']/v_['TWO']
        v_['THETA3'] = v_['TWO']/v_['THREE']
        v_['PSI1'] = 1.0e0
        v_['PSI2'] = 1.5e0
        v_['W1'] = 2.0e0
        v_['W2'] = 2.5e0
        v_['W3'] = 3.0e0
        v_['W4'] = 3.5e0
        v_['W5'] = 4.0e0
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['LAM'+str(I)+','+str(P)+','+str(Q)] = 1.0e+0
        v_['Q'] = v_['1']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA1']
        v_['Q'] = v_['2']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA2']
        v_['Q'] = v_['3']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA1']
        v_['Q'] = v_['4']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA2']
        v_['Q'] = v_['5']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA1']
        v_['Q'] = v_['6']
        v_['RA'+str(int(v_['Q']))] = v_['ONE']/v_['OMEGA2']
        for I in range(int(v_['1']),int(v_['NA'])+1):
            v_['LOGW'] = jnp.log(v_['W'+str(I)])
            v_['P'] = v_['1']
            v_['-THETA'] = -1.0*v_['THETA1']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI1']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
            v_['P'] = v_['2']
            v_['-THETA'] = -1.0*v_['THETA1']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI2']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
            v_['P'] = v_['3']
            v_['-THETA'] = -1.0*v_['THETA2']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI1']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
            v_['P'] = v_['4']
            v_['-THETA'] = -1.0*v_['THETA2']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI2']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
            v_['P'] = v_['5']
            v_['-THETA'] = -1.0*v_['THETA3']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI1']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
            v_['P'] = v_['6']
            v_['-THETA'] = -1.0*v_['THETA3']
            v_['RB'] = v_['LOGW']*v_['-THETA']
            v_['RE'] = jnp.exp(v_['RB'])
            v_['RB'] = v_['RE']*v_['PSI2']
            v_['RB'+str(I)+','+str(int(v_['P']))] = v_['RB']/v_['-THETA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    [iv,ix_,_] = jtu.s2mpj_ii('C'+str(I)+','+str(P)+','+str(Q),ix_)
                    self.xnames=jtu.arrset(self.xnames,iv,'C'+str(I)+','+str(P)+','+str(Q))
                    [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(I)+','+str(P)+','+str(Q),ix_)
                    self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(I)+','+str(P)+','+str(Q))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for L in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('I'+str(L),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'I'+str(L))
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['LAMBDA'] = v_['LAM'+str(I)+','+str(P)+','+str(Q)]
                    v_['-LAMBDA'] = -1.0e0*v_['LAMBDA']
                    [ig,ig_,_] = jtu.s2mpj_ii('T',ig_)
                    gtype = jtu.arrset(gtype,ig,'>=')
                    cnames = jtu.arrset(cnames,ig,'T')
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['Y'+str(I)+','+str(P)+','+str(Q)]])
                    valA = jtu.append(valA,float(v_['LAMBDA']))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['C'+str(I)+','+str(P)+','+str(Q)]])
                    valA = jtu.append(valA,float(v_['-LAMBDA']))
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
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['C'+str(I)+','+str(P)+','+str(Q)]]), 0.1e0)
                    self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(I)+','+str(P)+','+str(Q)]]), 0.1e0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.1e0))
        self.y0 = jnp.full((self.m,1),float(0.1e0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eA1', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA2', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA3', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA4', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA5', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eA6', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eB1', iet_)
        elftv = jtu.loaset(elftv,it,0,'Y')
        [it,iet_,_] = jtu.s2mpj_ii( 'eB2', iet_)
        elftv = jtu.loaset(elftv,it,0,'Y')
        [it,iet_,_] = jtu.s2mpj_ii( 'eB3', iet_)
        elftv = jtu.loaset(elftv,it,0,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A1-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA1"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A2-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA2')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA2"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A3-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA3')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA3"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A4-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA4')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA4"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A5-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA5')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA5"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'A6-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eA6')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eA6"])
                    vname = 'C'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'B1-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eB1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eB1"])
                    vname = 'Y'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'B2-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eB2')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eB2"])
                    vname = 'Y'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                for Q in range(int(v_['1']),int(v_['NCE'])+1):
                    ename = 'B3-'+str(I)+','+str(P)+','+str(Q)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'eB3')
                    ielftype = jtu.arrset(ielftype,ie,iet_["eB3"])
                    vname = 'Y'+str(I)+','+str(P)+','+str(Q)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.1e0))
                    posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['NA'])+1):
            for P in range(int(v_['1']),int(v_['NBD'])+1):
                v_['Q'] = v_['1']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                ig = ig_['OBJ']
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['Q'] = v_['2']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['Q'] = v_['3']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['Q'] = v_['4']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['Q'] = v_['5']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['Q'] = v_['6']
                v_['R'] = (v_['RA'+str(int(v_['Q']))]*v_['LAM'+str(I)+','+str(P)+','+
                     str(int(v_['Q']))])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(P)+','+str(int(v_['Q']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
        for I in range(int(v_['1']),int(v_['NA'])+1):
            v_['LOGW'] = jnp.log(v_['W'+str(I)])
            for Q in range(int(v_['1']),int(v_['NCE'])+1):
                v_['P'] = v_['1']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                ig = ig_['OBJ']
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['P'] = v_['2']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['P'] = v_['3']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['P'] = v_['4']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['P'] = v_['5']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
                v_['P'] = v_['6']
                v_['R'] = (v_['RB'+str(I)+','+str(int(v_['P']))]*v_['LAM'+str(I)+','+
                     str(int(v_['P']))+','+str(Q)])
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(Q)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['R']))
        v_['L'] = v_['0']
        for I in range(int(v_['1']),int(v_['NA'])+1):
            v_['I+1'] = 1+I
            v_['I-1'] = -1+I
            v_['LOGW'] = jnp.log(v_['W'+str(I)])
            v_['P'] = v_['1']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['P'] = v_['2']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['P'] = v_['3']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['P'] = v_['4']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['P'] = v_['5']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['P'] = v_['6']
            v_['P+1'] = 1+v_['P']
            v_['P-1'] = -1+v_['P']
            v_['RB'] = v_['RB'+str(I)+','+str(int(v_['P']))]
            v_['-RB'] = -1.0*v_['RB']
            v_['RA'] = v_['RA'+str(int(v_['1']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['2']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A1-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['1']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['2']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['1'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['3']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A2-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['2']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['3']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['2'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['4']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['3']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['4']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['3'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['5']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A4-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['4']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['5']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['4'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['6']),int(v_['NCE'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A5-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['5']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            v_['RA'] = v_['RA'+str(int(v_['6']))]
            v_['-RA'] = -1.0*v_['RA']
            for S in range(int(v_['1']),int(v_['P-1'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for S in range(int(v_['P+1']),int(v_['NBD'])+1):
                for R in range(int(v_['1']),int(v_['NA'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['1']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['2']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['3']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['4']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['5']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
                    v_['L'] = v_['L']+v_['1']
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(S)+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['1']),int(v_['I-1'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for R in range(int(v_['I+1']),int(v_['NA'])+1):
                for T in range(int(v_['1']),int(v_['NCE'])+1):
                    v_['L'] = v_['L']+v_['1']
                    ig = ig_['I'+str(int(v_['L']))]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(R)+','+str(int(v_['P']))+','+str(T)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
            for T in range(int(v_['1']),int(v_['5'])+1):
                v_['L'] = v_['L']+v_['1']
                ig = ig_['I'+str(int(v_['L']))]
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['A6-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RA']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(int(v_['6']))]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['RB']))
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['B3-'+str(I)+','+str(int(v_['P']))+','+str(T)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-RB']))
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-MN-72-1261"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(self):

        import jax.numpy as jnp
        self.efpar = jnp.array([]);
        self.efpar = jtu.arrset( self.efpar,0,0.1e0)
        return pbm

    @staticmethod
    def eA1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 0.0e0
        OMEGA = 1.0e0/2.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 0.0e0
        OMEGA = 2.0e0/3.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA3(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 1.0e0
        OMEGA = 1.0e0/2.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA4(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 1.0e0
        OMEGA = 2.0e0/3.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA5(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 1.5e0
        OMEGA = 1.0e0/2.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eA6(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        ALPHA = 1.5e0
        OMEGA = 2.0e0/3.0e0
        OM1 = OMEGA-1.0e0
        OM2 = OMEGA-2.0e0
        CMA = EV_[0]-ALPHA
        BIG = CMA>=self.efpar[0]
        if BIG!=0:
            F = CMA**OMEGA
        if BIG!=0:
            G = OMEGA*CMA**OM1
        if BIG!=0:
            H = OMEGA*OM1*CMA**OM2
        if BIG==0:
            C1 = OMEGA*(2.0e0-OMEGA)*self.efpar[0]**OM1
        if BIG==0:
            C2 = OMEGA*OM1*self.efpar[0]**OM2
        if BIG==0:
            F = ((1.0e0-1.5e0*OMEGA+0.5e0*OMEGA*OMEGA)*self.efpar[0]**OMEGA+
             C1*CMA+0.5e0*C2*CMA*CMA)
        if BIG==0:
            G = C1+C2*CMA
        if BIG==0:
            H = C2
        f_   = F
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, G)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), H)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eB1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        THETA = 1.0e0/3.0e0
        f_   = EV_[0]**THETA
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, THETA*EV_[0]**(THETA-1.0e0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), THETA*(THETA-1.0e0)*EV_[0]**(THETA-2.0e0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eB2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        THETA = 1.0e0/2.0e0
        f_   = EV_[0]**THETA
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, THETA*EV_[0]**(THETA-1.0e0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), THETA*(THETA-1.0e0)*EV_[0]**(THETA-2.0e0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eB3(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        THETA = 2.0e0/3.0e0
        f_   = EV_[0]**THETA
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, THETA*EV_[0]**(THETA-1.0e0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), THETA*(THETA-1.0e0)*EV_[0]**(THETA-2.0e0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

