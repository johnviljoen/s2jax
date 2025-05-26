"""
script to go through all s2mpj produced problems and convert them to jax/equinox
compatible format.
"""

import re
import os

# import test_ACOPP14 file as a string
def load_problem_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

file = "/home/john/Documents/s2jax/src/python_problems_old/ACOPP14.py"
code = load_problem_file(file)

# add a couple imports at the top
code = 'import s2jax.jax_utils as jtu\n' + code
code = 'from jax.experimental.sparse import BCOO, BCSR\n' + code

# a few ctrl-f replacements
code = re.sub(
    r'^(\s*)import\s+numpy\s+as\s+np\s*$',
    r'\1import jax.numpy as jnp',
    code, flags=re.M,
)
code = re.sub(r'\bnp\b', 'jnp', code)
code = re.sub(r'\bjnp\.append\b', 'jtu.append', code)
code = re.sub(r'\barrset\b', 'jtu.arrset', code)
code = re.sub(r'\bfind\b', 'jtu.find', code)
code = re.sub(r'\bloaset\b', 'jtu.loaset', code)
code = re.sub(r'\bs2mpj_ii\b', 'jtu.s2mpj_ii', code)
code = re.sub(r'\bs2mpj_nlx\b', 'jtu.s2mpj_nlx', code)

# change import from s2mpjlib import * to from s2jax import *
# test = """

# from s2mpjlib import *
# """
pattern = re.compile(r"^\s*from\s+s2mpjlib\s+import\s+\*\s*$", re.M)
code = pattern.sub("from s2jax.utils import *", code, count=1)  # only first match

# change class  ACOPP14(CUTEst_problem): -> class ACOPP14:
# test = """
# class ACOPP14(CUTEst_problem):
# """
pattern  = re.compile(r"""
    ^(?P<indent>\s*)          # leading spaces/tabs
    class\s+                  # 'class' plus at least one space
    (?P<name>[A-Za-z_]\w*)    # class name
    \s*\([^)]*\)\s*           # any base‑class list in (...)
    :                         # trailing colon
    """, re.M | re.VERBOSE)

code = pattern.sub(r"\g<indent>class \g<name>:", code)

# change variable[index] = value -> variable = variable.at[index].set(value)
# test = """
#         self.x0[ix_['P2']] = float(0.4)
# """
pattern = re.compile(r"""
    ^(?P<indent>\s*)            # leading spaces/tabs
    (?P<var>[^\[\s]+)           # variable name
    \[
        (?P<idx>(?!\s*['"]).*?) # ← do *not* match if index begins with ' or " - this is a dictionary key
    \]
    \s*=\s*                     # equals sign
    (?P<rhs>.+)$                # right‑hand side
    """, re.M | re.VERBOSE)

replacement = r"\g<indent>\g<var> = \g<var>.at[\g<idx>].set(\g<rhs>)"

code = pattern.sub(replacement, code)

# turn  csr_matrix((val,(ir,ic)),shape=(…))  →  BCSR.from_bcoo(BCOO((val,jnp.array((ir,ic)).T),shape=(…)))
csr_pat = re.compile(r"""
    csr_matrix\(                       # literal text
        \s*\(                          #  (
            (?P<val>[^,]+?)            #  valA
            \s*,\s*                    #  ,
            \(\s*(?P<ir>[^,]+?)\s*,\s* #  irA ,
                 (?P<ic>[^\)]+?)\s*\)  #  icA )
        \)\s*,\s*                      #  ),
        (?P<shape>shape\s*=\s*\([^\)]*\)) # shape=(…)
    \s*\)                              # )
    """, re.VERBOSE)

csr_repl = (
    r"BCSR.from_bcoo(BCOO((\g<val>, "
    r"jnp.array((\g<ir>,\g<ic>)).T), "
    r"\g<shape>))"
)

code = csr_pat.sub(csr_repl, code)

# write this to a new file
output_file = "/home/john/Documents/s2jax/src/python_problems/test_ACOPP14_jax.py"
with open(output_file, 'w') as file:
    file.write(code)

pass