import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class AIRCRFTA:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : AIRCRFTA
#    *********
# 
#    The aircraft stability problem by Rheinboldt, as a function
#    of the elevator, aileron and rudder deflection controls.
# 
#    Source: Problem 9 in
#    J.J. More',"A collection of nonlinear model problems"
#    Proceedings of the AMS-SIAM Summer Seminar on the Computational
#    Solution of Nonlinear Systems of Equations, Colorado, 1988.
#    Argonne National Laboratory MCS-P60-0289, 1989.
# 
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CNOR2-RN-8-5"
# 
#    Values for the controls
#    1) Elevator
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'AIRCRFTA'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['ELVVAL'] = 0.1
        v_['AILVAL'] = 0.0
        v_['RUDVAL'] = 0.0
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('ROLLRATE',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ROLLRATE')
        [iv,ix_,_] = jtu.s2mpj_ii('PITCHRAT',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'PITCHRAT')
        [iv,ix_,_] = jtu.s2mpj_ii('YAWRATE',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'YAWRATE')
        [iv,ix_,_] = jtu.s2mpj_ii('ATTCKANG',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ATTCKANG')
        [iv,ix_,_] = jtu.s2mpj_ii('SSLIPANG',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'SSLIPANG')
        [iv,ix_,_] = jtu.s2mpj_ii('ELEVATOR',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'ELEVATOR')
        [iv,ix_,_] = jtu.s2mpj_ii('AILERON',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'AILERON')
        [iv,ix_,_] = jtu.s2mpj_ii('RUDDERDF',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'RUDDERDF')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('G1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'G1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ROLLRATE']])
        valA = jtu.append(valA,float(-3.933))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PITCHRAT']])
        valA = jtu.append(valA,float(0.107))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['YAWRATE']])
        valA = jtu.append(valA,float(0.126))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SSLIPANG']])
        valA = jtu.append(valA,float(-9.99))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AILERON']])
        valA = jtu.append(valA,float(-45.83))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['RUDDERDF']])
        valA = jtu.append(valA,float(-7.64))
        [ig,ig_,_] = jtu.s2mpj_ii('G2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'G2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PITCHRAT']])
        valA = jtu.append(valA,float(-0.987))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ATTCKANG']])
        valA = jtu.append(valA,float(-22.95))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ELEVATOR']])
        valA = jtu.append(valA,float(-28.37))
        [ig,ig_,_] = jtu.s2mpj_ii('G3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'G3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ROLLRATE']])
        valA = jtu.append(valA,float(0.002))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['YAWRATE']])
        valA = jtu.append(valA,float(-0.235))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SSLIPANG']])
        valA = jtu.append(valA,float(5.67))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AILERON']])
        valA = jtu.append(valA,float(-0.921))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['RUDDERDF']])
        valA = jtu.append(valA,float(-6.51))
        [ig,ig_,_] = jtu.s2mpj_ii('G4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'G4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['PITCHRAT']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ATTCKANG']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['ELEVATOR']])
        valA = jtu.append(valA,float(-1.168))
        [ig,ig_,_] = jtu.s2mpj_ii('G5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'G5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['YAWRATE']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['SSLIPANG']])
        valA = jtu.append(valA,float(-0.196))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['AILERON']])
        valA = jtu.append(valA,float(-0.0071))
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
        self.xlower = jtu.np_like_set(self.xlower, ix_['ELEVATOR'], v_['ELVVAL'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['ELEVATOR'], v_['ELVVAL'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['AILERON'], v_['AILVAL'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['AILERON'], v_['AILVAL'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['RUDDERDF'], v_['RUDVAL'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['RUDDERDF'], v_['RUDVAL'])
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.0))
        if('ELEVATOR' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['ELEVATOR'], float(v_['ELVVAL']))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['ELEVATOR']),float(v_['ELVVAL'])))
        if('AILERON' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['AILERON'], float(v_['AILVAL']))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['AILERON']),float(v_['AILVAL'])))
        if('RUDDERDF' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['RUDDERDF'], float(v_['RUDVAL']))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['RUDDERDF']),float(v_['RUDVAL'])))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'E1A'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'PITCHRAT'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'YAWRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E1B'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'YAWRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'ATTCKANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E1C'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'ATTCKANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'SSLIPANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E1D'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'PITCHRAT'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'ATTCKANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E2A'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'ROLLRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'YAWRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E2B'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'ROLLRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'SSLIPANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E3A'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'ROLLRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'PITCHRAT'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E3B'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_['en2PR'])
        vname = 'ROLLRATE'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'ATTCKANG'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['G1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1A'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-0.727))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1B'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(8.39))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1C'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-684.4))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1D'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(63.5))
        ig = ig_['G2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2A'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.949))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2B'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.173))
        ig = ig_['G3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3A'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-0.716))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3B'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.578))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1D'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.132))
        ig = ig_['G4']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2B'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['G5']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3B'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
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
        self.pbclass   = "C-CNOR2-RN-8-5"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

