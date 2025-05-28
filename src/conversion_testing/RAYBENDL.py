import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class RAYBENDL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A ray bending problem.  A ray across a inhomogeneous 2D medium is
#    represented by a piecewise linear curve whose knots can be chosen.  
#    The problem is then to optimize the positions of these knots in order 
#    to obtain a ray path corresponding to the minimum travel time from 
#    source to receiver,  according to Fermat principle.
# 
#    The problem becomes harder and harder when the dimesnion increases
#    because the knots are getting closer and closer and the objective
#    has a nondifferentiable kink when two knots coincide.  The difficulty
#    is less apparent when exact second derivatives are not used.
# 
#    Source: a test example in
#    T.J. Moser, G. Nolet and R. Snieder,
#    "Ray bending revisited",
#    Bulletin of the Seism. Society of America 21(1).
# 
#    SIF input: Ph Toint, Dec 1991.
# 
#    classification = "C-COXR2-MY-V-0"
# 
#    number of  knots  ( >= 4 )
#    ( n = 2( NKNOTS - 1 ) ) 
# 
#           Alternative values for the SIF file parameters:
# IE NKNOTS              4              $-PARAMETER n = 6
# IE NKNOTS              11             $-PARAMETER n = 20
# IE NKNOTS              21             $-PARAMETER n = 40     original value
# IE NKNOTS              32             $-PARAMETER n = 62
# IE NKNOTS              64             $-PARAMETER n = 126
# IE NKNOTS              512            $-PARAMETER n = 1022
# IE NKNOTS              1024           $-PARAMETER n = 2046
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'RAYBENDL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['NKNOTS'] = int(4);  #  SIF file default value
        else:
            v_['NKNOTS'] = int(args[0])
        v_['XSRC'] = 0.0
        v_['ZSRC'] = 0.0
        v_['XRCV'] = 100.0
        v_['ZRCV'] = 100.0
        v_['NK-1'] = -1+v_['NKNOTS']
        v_['NK-2'] = -2+v_['NKNOTS']
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['NKNOTS'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('Z'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Z'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['NKNOTS'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('TIME'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(int(v_['0']))], v_['XSRC'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['0']))], v_['XSRC'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['Z'+str(int(v_['0']))], v_['ZSRC'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['Z'+str(int(v_['0']))], v_['ZSRC'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(int(v_['NKNOTS']))], v_['XRCV'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['NKNOTS']))], v_['XRCV'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['Z'+str(int(v_['NKNOTS']))], v_['ZRCV'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['Z'+str(int(v_['NKNOTS']))], v_['ZRCV'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        v_['XRANGE'] = v_['XRCV']-v_['XSRC']
        v_['ZRANGE'] = v_['ZRCV']-v_['ZSRC']
        v_['RKNOTS'] = float(v_['NKNOTS'])
        for I in range(int(v_['0']),int(v_['NKNOTS'])+1):
            v_['REALI'] = float(I)
            v_['FRAC'] = v_['REALI']/v_['RKNOTS']
            v_['XINCR'] = v_['FRAC']*v_['XRANGE']
            v_['ZINCR'] = v_['FRAC']*v_['ZRANGE']
            v_['XC'] = v_['XSRC']+v_['XINCR']
            v_['ZC'] = v_['ZSRC']+v_['ZINCR']
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['XC']))
            self.x0 = jtu.np_like_set(self.x0, ix_['Z'+str(I)], float(v_['ZC']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eTT', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        elftv = jtu.loaset(elftv,it,2,'Z1')
        elftv = jtu.loaset(elftv,it,3,'Z2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['NKNOTS'])+1):
            v_['I-1'] = -1+I
            ename = 'T'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'eTT')
                ielftype = jtu.arrset(ielftype,ie,iet_['eTT'])
            vname = 'X'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Z'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Z1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Z'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Z2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['NKNOTS'])+1):
            ig = ig_['TIME'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['T'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#   Solution of the continuous problem
# LO RAYBENDL            96.2424
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COXR2-MY-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(self):

        import jax.numpy as jnp
        self.efpar = jnp.array([])
        self.efpar = jtu.arrset( self.efpar,0,0.01)
        return pbm

    @staticmethod
    def eTT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((3,4))
        IV_ = jnp.zeros(3)
        U_ = jtu.np_like_set(U_, jnp.array([2,0]), U_[2,0]-1)
        U_ = jtu.np_like_set(U_, jnp.array([2,1]), U_[2,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 2, U_[2:3,:].dot(EV_))
        C0 = 1.0+self.efpar[0]*IV_[0]
        C1 = 1.0+self.efpar[0]*IV_[1]
        DCDZ = self.efpar[0]
        V = 1.0/C1+1.0/C0
        VDZ0 = -DCDZ/(C0*C0)
        VDZ1 = -DCDZ/(C1*C1)
        VDZ0Z0 = 2.0*DCDZ*DCDZ/C0**3
        VDZ1Z1 = 2.0*DCDZ*DCDZ/C1**3
        DZ1 = IV_[1]-IV_[0]
        R = jnp.sqrt(IV_[2]*IV_[2]+DZ1*DZ1)
        RDX = IV_[2]/R
        RDZ1 = DZ1/R
        RDZ0 = -RDZ1
        RDXDX = (1.0-IV_[2]*IV_[2]/(R*R))/R
        RDXZ1 = -IV_[2]*DZ1/R**3
        RDXZ0 = -RDXZ1
        RDZ1Z1 = (1.0-DZ1*DZ1/(R*R))/R
        RDZ0Z0 = RDZ1Z1
        RDZ0Z1 = -RDZ1Z1
        f_   = V*R
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 2, V*RDX)
            g_ = jtu.np_like_set(g_, 0, V*RDZ0+VDZ0*R)
            g_ = jtu.np_like_set(g_, 1, V*RDZ1+VDZ1*R)
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), V*RDXDX)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), VDZ0*RDX+V*RDXZ0)
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), VDZ1*RDX+V*RDXZ1)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), V*RDZ0Z0+VDZ0Z0*R+2.0*VDZ0*RDZ0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), V*RDZ0Z1+VDZ1*RDZ0+VDZ0*RDZ1)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), V*RDZ1Z1+VDZ1Z1*R+2.0*VDZ1*RDZ1)
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

