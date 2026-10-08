"""Optional exact C implementation of the existing grouped integer DFS.

This changes neither assignment domains nor pruning/tie order or solver work.
Coefficient rounding, CP-SAT, physical replay and all outputs stay in Python.
Only round3's isolated solver is patched; frozen round2 is never mutated.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

_LIB = None
_I = ctypes.c_int64
_U = ctypes.c_uint64
_MAX = (1 << 63)-1


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source=Path(__file__).with_suffix('.c')
    identity=hashlib.sha256(source.read_bytes()+b'gcc -O3 -std=c99 -fPIC -shared').hexdigest()[:20]
    directory=source.parent/'.native'; directory.mkdir(exist_ok=True)
    dest=directory/('enum_'+identity+'.so')
    if not dest.exists():
        temp=directory/(dest.name+'.'+str(os.getpid())+'.tmp')
        subprocess.run(['gcc','-O3','-std=c99','-fPIC','-shared',str(source),'-o',str(temp)],check=True,
                       capture_output=True,text=True)
        os.replace(temp,dest)
    lib=ctypes.CDLL(str(dest))
    ptr=ctypes.POINTER(_I)
    lib.exact_visit.argtypes=[ctypes.c_int,ctypes.c_int,*([ptr]*8),
                             ctypes.POINTER(_U),ctypes.POINTER(_U),ctypes.POINTER(_U)]
    lib.exact_visit.restype=ctypes.c_int
    _LIB=lib
    return lib


def _python_visit(alternatives, suffix, deps, initial):
    """Overflow/compiler fallback with the original traversal exactly."""
    ng=len(alternatives);nr=len(initial)
    best=float('inf');bestchoices=None;nodes=leaves=pruned=0;selected=[]
    def visit(g,loads,dep):
        nonlocal best,bestchoices,nodes,leaves,pruned
        nodes+=1
        lower=max([dep,deps[g],*(loads[j]+suffix[g][j] for j in range(nr))],default=0)
        if lower>=best:
            pruned+=1;return
        if g==ng:
            leaves+=1;best=lower;bestchoices=tuple(selected);return
        for ns,add,d in alternatives[g]:
            selected.append(ns)
            visit(g+1,[loads[j]+add[j] for j in range(nr)],max(dep,d))
            selected.pop()
    visit(0,initial,0)
    return best,bestchoices,nodes,leaves,pruned


def exact_visit(alternatives, suffix, deps, initial):
    ng=len(alternatives);nr=len(initial)
    numbers=[*initial,*deps,*(v for row in suffix for v in row)]
    numbers += [v for group in alternatives for _,loads,d in group for v in (*loads,d)]
    fits=(ng<=512 and nr<=128 and all(isinstance(x,int) and 0<=x<_MAX for x in numbers))
    if fits:
        fits=all(initial[j]+sum(max(a[1][j] for a in group) for group in alternatives)<_MAX
                 for j in range(nr))
    if not fits:
        return _python_visit(alternatives,suffix,deps,initial)
    try:
        lib=_load_library()
    except (OSError,subprocess.CalledProcessError):
        return _python_visit(alternatives,suffix,deps,initial)
    offsets=[0]
    for group in alternatives:offsets.append(offsets[-1]+len(group))
    def array(values):return (_I*max(1,len(values)))(*values)
    off=array(offsets)
    add=array([x for group in alternatives for _,loads,_ in group for x in loads])
    alt_dep=array([d for group in alternatives for _,_,d in group])
    suf=array([x for row in suffix for x in row]);dep=array(deps);ini=array(initial)
    path=(_I*max(1,ng))();best=_I();nodes=_U();leaves=_U();pruned=_U()
    code=lib.exact_visit(ng,nr,off,add,alt_dep,suf,dep,ini,path,ctypes.byref(best),
                         ctypes.byref(nodes),ctypes.byref(leaves),ctypes.byref(pruned))
    if code:
        return _python_visit(alternatives,suffix,deps,initial)
    choices=tuple(alternatives[g][path[g]][0] for g in range(ng))
    return best.value,choices,nodes.value,leaves.value,pruned.value


def enable_exact_native_enum():
    """Install an equivalent DFS in the local isolated module, explicitly."""
    from . import optimizer
    module=optimizer._solver
    if getattr(module,'_round3_native_enum',False):return
    source=(Path(__file__).resolve().parents[1]/'round2/optimizer.py').read_text()
    start=source.index('def _enumerate_assignment(')
    stop=source.index('\ndef _solve_fixed_t',start)
    function=source[start:stop]
    old_start=function.index('    best=math.inf;bestchoices=None;')
    old_stop=function.index('    owners=[None]*len(table)',old_start)
    replacement='    best,bestchoices,nodes,leaves,pruned=_round3_exact_visit(alternatives,suffix,deps,initial)\n'
    function=function[:old_start]+replacement+function[old_stop:]
    module._round3_exact_visit=exact_visit
    module._round3_python_enum=module._enumerate_assignment
    exec(compile(function,str(Path(__file__)), 'exec'),module.__dict__)
    module._round3_native_enum=True


def disable_exact_native_enum():
    from . import optimizer
    module=optimizer._solver
    if getattr(module,'_round3_native_enum',False):
        module._enumerate_assignment=module._round3_python_enum
        module._round3_native_enum=False
