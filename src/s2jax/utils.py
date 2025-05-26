from pprint import pprint
import numpy as np
import re

# used by LEVYM.py, LEVYMONT8C.py
class structtype():
    def __str__(self):
        pprint(vars(self))
        return ''
    pass
    def __repr__(self):
        pprint(vars(self))
        return ''
    
# Computes the effective index name in List, or add name to List if not in there already. Return
# the index of name in List and new = 1 if the List has been enlarged or 0 if name was already
# present in List at the call.

def s2mpj_ii( name, List ):
    
    if name in List:
       idx = List[ name ]
       new = 0
    else:
       idx = len( List)
       List[ name ] = idx
       new = 1
    return idx, List, new

# Get the index of a nonlinear variable.  This implies adding it to the variables' dictionary ix_
# if it is a new one, and adjusting the bounds, start point and types according to their default
# setting.
def s2mpj_nlx( self, name, List, getxnames=None, xlowdef=None, xuppdef=None, x0def=None ):

    iv, List, newvar = s2mpj_ii( name, List );
    if( newvar ):
        self.n = self.n + 1;
        if getxnames:
            self.xnames = arrset( self.xnames, iv, name )
        if hasattr( self, "xlower" ):
            thelen = len( self.xlower )
            if ( iv <= thelen ):
                self.xlower = np.append( self.xlower, np.full( (iv-thelen+1,1), float(0.0) ), 0 )
            if not xlowdef is None:
                self.xlower[iv] 
        if hasattr( self, "xupper" ):
            thelen = len( self.xupper )
            if ( iv <= thelen ):
                self.xupper = np.append( self.xupper, np.full( (iv-thelen+1,1), float('Inf') ), 0 )
            if not xuppdef is None:
                self.xupper[iv] 
        try:
            self.xtype  = arrset( self.xtype, iv, 'r' )
        except:
            pass
        thelen = len( self.x0 )
        if ( iv <= thelen ):
            self.x0 = np.append( self.x0, np.full( (iv-thelen+1,1), 0.0 ), 0 )
        if not x0def is None:
            self.x0[iv] =  x0def
    return iv, List

# An emulation of the Matlab find() function for everything that can ne enumerated
def find( lst, condition ):
    return np.array([i for i, elem in enumerate(lst) if condition(elem)])

# Set the elements indexed by index of an np.array (arr) to value.
def arrset( arr, index, value ):
    if isinstance( index, np.ndarray):
        maxind = np.max( index )
    else:
        maxind = index
    if len(arr) <= maxind:
        arr= np.append( arr, np.full( maxind - len( arr ) + 1, None ) )
    arr[index] = value
    return arr

# Set the (i,j)-th element of a list of arrays (loa) to value.
def loaset( loa, i, j, value ):
    if len(loa) <= i:
       loa.extend( [None] * ( i - len( loa ) + 1 ) )
    if loa[i] is None:
       loa[i]= np.full( j + 1, None )
    if len(loa[i]) <= j:
       loa[i]= np.append(loa[i],np.full( j - len( loa[i] ) + 1, None ) )
    loa[i][j] = value
    return loa

# This tool consider all problems in list_of_python_problems (whose files are in
# the ./python_problems directory) and selects those whose SIF classification matches
# that given by the input string classif. Matching is in the sense of  regular expressions
# (regexp).
# If varargin is empty (i.e. only classif is used as input argument), the function prints
# the list of matching problems on standard output. Otherwise, the list is output in the
# file whose name is a string passed as varargin{1} (Further components of varargin are
# ignored).
#
# If the input string is 'help'  or 'h', a message is printed on the standard output
# describing the SIF classification scheme and an brief explanation of how to use the tool.
#
# Thanks to Greta Malaspina (Firenze) for an inital implementation in Matlab.
def s2mpjlib_select( classif, *args ):

    if classif in [ "help", "h" ]:
    
        print( "  " )
        print( " === The classification scheme ===" )
        print( "  " )
        print( " A problem is classified by a string of the form" )
        print( "    X-XXXXr-XX-n-m" )
        print( " The first character in the string identifies the problem collection" )
        print( " from which the problem is extracted. Possible values are" )
        print( "    C the CUTEst collection;" )
        print( "    S the SPARCO collection; and" )
        print( "    N none of the above." )
        print( " The character immediately following the first hyphen defines the type" )
        print( " of variables occurring in the problem. Its possible values are" )
        print( "    C the problem has continuous variables only;" )
        print( "    I the problem has integer variables only;" )
        print( "    B the problem has binary variables only; and" )
        print( "    M the problem has variables of different types." )
        print( " The second character after the first hyphen defines the type" )
        print( " of the problem''s objective function. Its possible values are" )
        print( "    N no objective function is defined;" )
        print( "    C the objective function is constant;" )
        print( "    L the objective function is linear;" )
        print( "    Q the objective function is quadratic;" )
        print( "    S the objective function is a sum of squares; and" )
        print( "    O the objective function is none of the above." )
        print( " The third character after the first hyphen defines the type of" )
        print( " constraints of the problem. Its possible values are" )
        print( "    U the problem is unconstrained;" )
        print( "    X the problem’s only constraints are fixed variables;" )
        print( "    B the problem’s only constraints are bounds on the variables;" )
        print( "    N the problem’s constraints represent the adjacency matrix of a (linear)" )
        print( "      network;" )
        print( "    L the problem’s constraints are linear;" )
        print( "    Q the problem’s constraints are quadratic; and" )
        print( "    O the problem’s constraints are more general than any of the above alone." )
        print( " The fourth character after the first hyphen indicates the smoothness of" )
        print( " the problem. There are two possible choices" )
        print( "    R the problem is regular, that is, its first and second derivatives " )
        print( "      exist and are continuous everywhere; or" )
        print( "    I the problem is irregular." )
        print( " The integer (r) which corresponds to the fourth character of the string is" )
        print( " the degree of the highest derivatives provided analytically within the problem" )
        print( " description. It is restricted to being one of the single characters O, 1, or 2." )
        print( " The character immediately following the second hyphen indicates the primary" )
        print( " origin and/or interest of the problem. Its possible values are" )
        print( "    A the problem is academic, that is, has been constructed specifically by" )
        print( "      researchers to test one or more algorithms;" )
        print( "    M the problem is part of a modeling exercise where the actual value of the" )
        print( "      solution is not used in a genuine practical application; and" )
        print( "    R the problem’s solution is (or has been) actually used in a real")
        print( "      application for purposes other than testing algorithms." )
        print( " The next character in the string indicates whether or not the problem" )
        print( " description contains explicit internal variables. There are two possible" )
        print( " values, namely," )
        print( "    Y the problem description contains explicit internal variables; or" )
        print( "    N the problem description does not contain any explicit internal variables." )
        print( " The symbol(s) between the third and fourth hyphen indicate the number of" )
        print( " variables in the problem. Possible values are" )
        print( "    V the number of variables in the problem can be chosen by the user; or" )
        print( "    n a positive integer giving the actual (fixed) number of problem variables." )
        print( " The symbol(s) after the fourth hyphen indicate the number of constraints" )
        print( " (other than fixed variables and bounds) in the problem. Note that fixed" )
        print( " variables are not considered as general constraints here. The two possible" )
        print( " values are" )
        print( "    V the number of constraints in the problem can be chosen by the user; or" )
        print( "    m a nonnegative integer giving the actual (fixed) number of constraints." )
        print( "  " )
        print( " === Using the problem selection tool ===" )
        print( "  " )
        print( " In order to use the selection too, you should first open Python in the parent" )
        print( " of the directory containing the Python problem files, then import the library" )
        print( " by issuing the command" )
        print( "    from s2jax.reference import *" )
        print( " or, more specifiaclly," )
        print( "    from s2mpjlib import s2mpjlib_select")
        print( " The select tool may then be called with its first argument being a string ")
        print( " which specifies the class of problems of interest.  This string is constructed" )
        print( " by replacing by a dot each character in the classification string for which" )
        print( " all possible values are acceptable (the dot is a wildcard character)." )
        print( " For instance" )
        print( "    s2mpjlib_select( \"C-CSU..-..-2-0\" ) ")
        print( " lists all CUTEst unconstrained ""sum-of-squares"" problems in two continuous" )
        print( " variables, while " )
        print( "    s2mpjlib_select( ""C-C....-..-V-V"" ) " )
        print( " lists all CUTEst problems with variable number of continuous variables and" )
        print( " variable number of constraints." )
        print( " The classification strings \"unconstrained\", \"bound-constrained\", " )
        print( " \"fixed-variables\", \"general-constraints\", \"variable-n\" and " )
        print( " \"variable-m\" are also allowed." )
        print( " NOTE: any regular expression may be used as the first argument of select " )
        print( "       to specify the problem class, so that, for instance, the previous " )
        print( "       selection can also be achieved by s2mpjlib_select( \"C-C.*V-V\" ) ")
        print( " Writing the list of selected problems to a file is obtained by specifying" )
        print( " the name of the file as a second argument of select, as in ")
        print( "    s2mpjlib_select( \"C-C....-..-V-V\", filename )" )

    else:
    
        #  Modify the filter to cope with fixed numbers of variables/constraints with more
        #  than one digit.

        if classif == "unconstrained":
            classif = ".-..U.*"
        elif classif == "bound-constrained":
            classif = ".-..B.*"
        elif classif == "fixed-variables":
            classif = ".-..X.*"
        elif classif == "general-constraints":
            classif = ".-..[LNQO].*"
        elif classif == "variable-n":
            classif = ".-..B..-..-V-[V0-9]*"
        elif classif == "variable-m":
            classif = ".-..B..-..-[V0-9]*-V"
        else:
            lencl = len( classif )
            if lencl > 11 and classif[11] == ".":
                oclassif = classif
                classif  = classif[0:11] + "[V0-9]*"
                if lencl> 12:
                    classif = classif + oclassif[12:lencl]
            lenclm1 = len( classif ) - 1
            if classif[ lenclm1 ] == ".":
                classif = classif[0:lenclm1] + "[V0-9]*"
        filter_pattern = f'classification = .*{classif}'
      
        list_of_problems = "./list_of_python_problems"
        python_problems  = "./python_problems/"

        if len(args) > 0:
            fid = open( args[0], "w" )
        else:
            fid = None

        filter_pattern = f'classification = .*{classif}'
        with open( list_of_problems, 'r' ) as f:
            allprobs = f.readlines()

        for theprob in allprobs:
            theprob = theprob.strip()
            problem_path = os.path.join( python_problems, theprob )
            if os.path.isfile( problem_path ):
                with open( problem_path, 'r' ) as prob_file:
                    content = prob_file.read()
                if re.search( filter_pattern, content ):
                    if fid:
                        fid.write(f'{theprob}\n')
                    else:
                        print( theprob )

        if fid:
            fid.close()

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # dynamically allocate memory like pytorch does
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_log_compiles", True) # will print out recompilations
    jax.config.update('jax_default_matmul_precision', "default")
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

