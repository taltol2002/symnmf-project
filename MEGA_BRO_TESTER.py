"""
MEGA BRO TESTER - SymNMF Final Project
Only tests what the assignment requires. No synthetic edge cases.
"""
import subprocess, os, time, math, random

P="\033[92mPASS\033[0m"; F="\033[91mFAIL\033[0m"; W="\033[93mWARN\033[0m"
B="\033[1m"; R="\033[0m"
tt=pt=0; fl=[]

def run(c,t=120):
    try: r=subprocess.run(c,capture_output=True,text=True,timeout=t,shell=True); return r.stdout.strip(),r.stderr.strip(),r.returncode
    except: return "","TIMEOUT",-1

def chk(n,ok,d=""):
    global tt,pt,fl; tt+=1
    if ok: pt+=1; print(f"  [{P}] {n}")
    else: fl.append(n); print(f"  [{F}] {n}"); d and print(f"         {d[:200]}")

def sec(t): print(f"\n{B}{'='*70}\n  {t}\n{'='*70}{R}")

def pm(t):
    r=[]
    for l in t.strip().split('\n'):
        if l.strip(): r.append([float(x) for x in l.strip().split(',')])
    return r

def meq(a,b,tol=1e-3):
    if len(a)!=len(b): return False
    for i in range(len(a)):
        if len(a[i])!=len(b[i]): return False
        for j in range(len(a[i])):
            if abs(a[i][j]-b[i][j])>tol: return False
    return True

def issym(m):
    n=len(m)
    for i in range(n):
        for j in range(n):
            if abs(m[i][j]-m[j][i])>1e-6: return False
    return True

def c4dp(t):
    for l in t.strip().split('\n'):
        for v in l.split(','):
            if '.' not in v.strip() or len(v.strip().split('.')[1])!=4: return False
    return True

def mkf(n,d):
    with open(n,'w') as f:
        for r in d: f.write(','.join(str(x) for x in r)+'\n')

def has_vg():
    _,_,c=run("which valgrind"); return c==0

def msym(X):
    n=len(X);d=len(X[0]);A=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i!=j: A[i][j]=math.exp(-sum((X[i][k]-X[j][k])**2 for k in range(d))/2.0)
    return A

def mddg(X):
    A=msym(X);n=len(A);D=[[0.0]*n for _ in range(n)]
    for i in range(n): D[i][i]=sum(A[i])
    return D

def mnorm(X):
    A=msym(X);n=len(A);dg=[sum(A[i]) for i in range(n)]
    W=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dn=math.sqrt(dg[i])*math.sqrt(dg[j])
            W[i][j]=A[i][j]/dn if dn>0 else 0.0
    return W

DS={
    "8pts_2d":[[1.0,2.0],[1.2,1.8],[0.8,2.2],[1.1,1.9],[5.0,5.0],[5.3,4.8],[4.9,5.2],[5.1,4.9]],
    "20pts_2d":[[1.0,2.0],[1.2,1.8],[0.8,2.2],[1.1,1.9],[0.9,2.1],[1.3,1.7],[1.0,1.8],
                [5.0,5.0],[5.3,4.8],[4.9,5.2],[5.1,4.9],[5.2,5.1],[4.8,5.3],[5.0,4.7],
                [9.0,1.0],[9.2,0.8],[8.8,1.2],[9.1,0.9],[8.9,1.1],[9.3,0.7]],
}
random.seed(777)
DS["50pts_5d"]=[[random.gauss(0,1) for _ in range(5)] for _ in range(50)]


def t1_build():
    sec("1. BUILD & FILES")
    run("make clean 2>/dev/null"); run("rm -f symnmfmodule.cpython-*.so"); run("rm -rf build __pycache__")
    o,e,c=run("make"); chk("make compiles",c==0,e)
    chk("symnmf executable",os.path.exists("symnmf"))
    run("make clean"); o,e,c=run("make 2>&1"); chk("no warnings","warning" not in(o+e).lower())
    o,e,c=run("python3 setup.py build_ext --inplace"); chk("setup.py builds",c==0,e[:200])
    chk(".so created",any('symnmfmodule' in f and f.endswith('.so') for f in os.listdir('.')))
    for f in ['symnmf.py','symnmf.c','symnmfmodule.c','symnmf.h','analysis.py','setup.py','Makefile','kmeans.py']:
        chk(f"file: {f}",os.path.exists(f))


def t2_c_math():
    sec("2. C PROGRAM — sym/ddg/norm + MATH VERIFICATION")
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data); n=len(data)
        print(f"\n  --- {name} ({n}pts) ---")
        t0=time.time()
        for goal,mfn in [("sym",msym),("ddg",mddg),("norm",mnorm)]:
            o,e,c=run(f"./symnmf {goal} {fn}")
            chk(f"{goal}: runs",c==0 and o!="",e[:80])
            if not o: continue
            m=pm(o)
            chk(f"{goal}: {n}x{n}",len(m)==n and all(len(r)==n for r in m))
            chk(f"{goal}: %.4f format",c4dp(o))
            chk(f"{goal}: matches manual math",meq(m,mfn(data)))
            if goal=="sym":
                chk("sym: diagonal=0",all(abs(m[i][i])<1e-6 for i in range(n)))
                chk("sym: symmetric",issym(m))
            elif goal=="ddg":
                chk("ddg: off-diag=0",all(abs(m[i][j])<1e-6 for i in range(n) for j in range(n) if i!=j))
                chk("ddg: diagonal>0",all(m[i][i]>0 for i in range(n)))
            elif goal=="norm":
                chk("norm: diagonal=0",all(abs(m[i][i])<1e-6 for i in range(n)))
                chk("norm: symmetric",issym(m))
        print(f"  ({time.time()-t0:.2f}s)")
        os.remove(fn)


def t3_c_errors():
    sec("3. C PROGRAM — ERROR HANDLING")
    mkf("_e.txt",[[1.0,2.0],[3.0,4.0]])
    for d,cmd in [("no args","./symnmf"),("one arg","./symnmf sym"),
                  ("invalid goal","./symnmf invalid _e.txt"),("symnmf in C","./symnmf symnmf _e.txt"),
                  ("missing file","./symnmf sym nonexistent.txt"),("too many args","./symnmf sym ddg _e.txt")]:
        o,_,_=run(cmd); chk(f"{d}: error msg","An Error Has Occurred" in o)
    os.remove("_e.txt")


def t4_python_funcs():
    sec("4. PYTHON — sym/ddg/norm/symnmf")
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data); n=len(data)
        print(f"\n  --- {name} ---")
        t0=time.time()
        for g in ["sym","ddg","norm"]:
            o,e,c=run(f"python3 symnmf.py 2 {g} {fn}")
            chk(f"py {g}: runs",c==0 and o!="",e[:80])
            if o: chk(f"py {g}: %.4f",c4dp(o))
        for k in [2,3]:
            if k>=n: continue
            o,e,c=run(f"python3 symnmf.py {k} symnmf {fn}")
            chk(f"py symnmf k={k}: runs",c==0 and o!="",e[:80])
            if not o: continue
            m=pm(o)
            chk(f"  {n}rows x {k}cols",len(m)==n and all(len(r)==k for r in m))
            chk(f"  values>=0",all(m[i][j]>=-1e-10 for i in range(len(m)) for j in range(len(m[0]))))
            chk(f"  %.4f",c4dp(o))
        print(f"  ({time.time()-t0:.2f}s)")
        os.remove(fn)


def t5_python_errors():
    sec("5. PYTHON — ERROR HANDLING")
    mkf("_pe.txt",[[1.0,2.0],[3.0,4.0],[5.0,6.0]])
    for d,cmd in [("no args","python3 symnmf.py"),("too few args","python3 symnmf.py 2 sym"),
                  ("invalid goal","python3 symnmf.py 2 invalid _pe.txt"),
                  ("k=abc","python3 symnmf.py abc sym _pe.txt"),
                  ("missing file","python3 symnmf.py 2 sym nonexistent.txt")]:
        o,e,_=run(cmd); chk(f"{d}","An Error Has Occurred" in o+e)
    os.remove("_pe.txt")


def t6_c_vs_python():
    sec("6. C vs PYTHON — SAME OUTPUT")
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data)
        for g in ["sym","ddg","norm"]:
            co,_,cc=run(f"./symnmf {g} {fn}"); po,_,pc=run(f"python3 symnmf.py 2 {g} {fn}")
            if cc==0 and pc==0 and co and po:
                chk(f"{name} {g}: C==Python",meq(pm(co),pm(po)))
        os.remove(fn)


def t7_analysis():
    sec("7. ANALYSIS.PY")
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data)
        t0=time.time()
        for k in [2,3]:
            o,e,c=run(f"python3 analysis.py {k} {fn}")
            chk(f"analysis {k} {name}: runs",c==0 and o!="",e[:80])
            if not o: continue
            ls=o.strip().split('\n')
            chk(f"  2 lines",len(ls)==2)
            has_n=any(l.startswith('nmf:') for l in ls)
            has_k=any(l.startswith('kmeans:') for l in ls)
            chk(f"  has nmf:",has_n); chk(f"  has kmeans:",has_k)
            for l in ls:
                if ':' in l:
                    lb=l.split(':')[0].strip(); v=float(l.split(':')[1])
                    chk(f"  {lb} in [-1,1]",-1<=v<=1)
        print(f"  ({time.time()-t0:.2f}s)")
        os.remove(fn)


def t8_reference():
    sec("8. REFERENCE RESULTS (21 values)")
    ref={('input_1.txt',2):(0.8856,0.8856),('input_1.txt',3):(0.5957,0.5957),
         ('input_1.txt',4):(0.4944,0.5154),('input_1.txt',5):(0.2866,0.4556),
         ('input_1.txt',6):(0.2859,0.4619),('input_1.txt',7):(0.2192,0.4611),
         ('input_1.txt',8):(0.4522,0.4571),('input_1.txt',9):(0.2316,0.4583),
         ('input_1.txt',10):(0.2215,0.4596),
         ('input_2.txt',2):(0.1938,0.3481),('input_2.txt',3):(0.2714,0.4789),
         ('input_2.txt',4):(0.3416,0.4875),('input_2.txt',5):(0.4775,0.5725),
         ('input_2.txt',6):(0.6272,0.6820),('input_2.txt',7):(0.5469,0.7491),
         ('input_2.txt',8):(0.6637,0.6475),('input_2.txt',9):(0.5449,0.5625),
         ('input_2.txt',10):(0.2127,0.4820),
         ('input_3.txt',2):(0.0878,0.3128),('input_3.txt',3):(0.0715,0.3395),
         ('input_3.txt',4):(0.1439,0.4091)}
    tested=0
    for (f,k),(rn,rk) in ref.items():
        if not os.path.exists(f): continue
        tested+=1; o,e,c=run(f"python3 analysis.py {k} {f}")
        if c!=0: chk(f"k={k} {f}",False,e[:80]); continue
        ns=ks=None
        for l in o.strip().split('\n'):
            if l.startswith('nmf:'): ns=float(l.split(':')[1])
            elif l.startswith('kmeans:'): ks=float(l.split(':')[1])
        chk(f"k={k} {f}: nmf={ns} ref={rn}",ns is not None and abs(ns-rn)<0.001)
        chk(f"k={k} {f}: km={ks} ref={rk}",ks is not None and abs(ks-rk)<0.001)
    if tested==0: print(f"  [{W}] No reference files (input_1/2/3.txt)")


def t9_valgrind():
    sec("9. VALGRIND — MEMORY LEAKS")
    if not has_vg(): print(f"  [{W}] valgrind not installed"); return
    vg="valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=99"
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data)
        print(f"\n  --- {name} ---")
        for g in ["sym","ddg","norm"]:
            t0=time.time()
            _,e,_=run(f"{vg} ./symnmf {g} {fn}",180)
            nl="no leaks are possible" in e or "All heap blocks were freed" in e or "definitely lost: 0 bytes" in e
            ne="ERROR SUMMARY: 0 errors" in e
            chk(f"vg {g}: no leaks",nl); chk(f"vg {g}: no errors",ne)
            print(f"    ({time.time()-t0:.1f}s)")
        os.remove(fn)
    print(f"\n  --- error paths ---")
    mkf("_ve.txt",[[1.0,2.0],[3.0,4.0]])
    for d,cmd in [("missing file",f"{vg} ./symnmf sym nope.txt"),
                  ("bad goal",f"{vg} ./symnmf invalid _ve.txt"),
                  ("no args",f"{vg} ./symnmf")]:
        _,e,_=run(cmd,60)
        nl="no leaks are possible" in e or "All heap blocks were freed" in e or "definitely lost: 0 bytes" in e
        chk(f"vg error ({d}): no leaks",nl)
    os.remove("_ve.txt")


def t10_quality():
    sec("10. CODE QUALITY")
    with open("symnmf.c") as f: cc=f.read()
    chk('C: %.4f','"%.4f"' in cc)
    chk('C: error msg','"An Error Has Occurred' in cc)
    with open("symnmfmodule.c") as f: ml=f.readlines()
    chk("module: PY_SSIZE_T_CLEAN",ml[0].strip()=="#define PY_SSIZE_T_CLEAN")
    chk("module: Python.h",any("Python.h" in l for l in ml[:3]))
    chk("module: name symnmfmodule","symnmfmodule" in ''.join(ml[-5:]))
    with open("Makefile") as f: mk=f.read()
    for flag in ["-ansi","-Wall","-Wextra","-Werror","-pedantic-errors"]:
        chk(f"Makefile: {flag}",flag in mk)
    with open("symnmf.py") as f: py=f.read()
    chk("py: seed(1234)","np.random.seed(1234)" in py)
    chk("py: __name__","__name__" in py)
    chk("py: import symnmfmodule","symnmfmodule" in py)
    with open("analysis.py") as f: an=f.read()
    chk("analysis: seed(1234)","np.random.seed(1234)" in an)
    chk("analysis: silhouette_score","silhouette_score" in an)
    chk("analysis: __name__","__name__" in an)


def t11_repro():
    sec("11. REPRODUCIBILITY")
    mkf("_rp.txt",DS["8pts_2d"])
    o1,_,_=run("python3 symnmf.py 2 symnmf _rp.txt")
    o2,_,_=run("python3 symnmf.py 2 symnmf _rp.txt")
    chk("symnmf: same output twice",o1==o2 and o1!="")
    o1,_,_=run("python3 analysis.py 2 _rp.txt")
    o2,_,_=run("python3 analysis.py 2 _rp.txt")
    chk("analysis: same output twice",o1==o2 and o1!="")
    os.remove("_rp.txt")


def t12_timing():
    sec("12. PERFORMANCE")
    for name,data in DS.items():
        fn=f"_t_{name}.txt"; mkf(fn,data); n=len(data)
        t0=time.time(); run(f"./symnmf sym {fn}"); print(f"  C sym {name} ({n}pts): {time.time()-t0:.3f}s")
        t0=time.time(); run(f"./symnmf norm {fn}"); print(f"  C norm {name} ({n}pts): {time.time()-t0:.3f}s")
        if n>=4:
            t0=time.time(); run(f"python3 symnmf.py 2 symnmf {fn}"); print(f"  Py symnmf {name} ({n}pts): {time.time()-t0:.3f}s")
        os.remove(fn)
    for f in ['input_1.txt','input_2.txt','input_3.txt']:
        if not os.path.exists(f): continue
        with open(f) as fh: n=sum(1 for l in fh if l.strip())
        t0=time.time(); run(f"./symnmf norm {f}"); print(f"  C norm {f} ({n}pts): {time.time()-t0:.3f}s")
        t0=time.time(); run(f"python3 analysis.py 2 {f}"); print(f"  Py analysis {f} ({n}pts): {time.time()-t0:.3f}s")


def main():
    global tt,pt,fl
    start=time.time()
    print(f"\n{B}{'#'*70}\n#{'MEGA BRO TESTER':^68}#\n{'#'*70}{R}")
    for n in ['input_1.txt','input_2.txt','input_3.txt']:
        if not os.path.exists(n) and os.path.exists(f'tests/{n}'): run(f"cp tests/{n} .")
    t1_build(); t2_c_math(); t3_c_errors(); t4_python_funcs(); t5_python_errors()
    t6_c_vs_python(); t7_analysis(); t8_reference(); t9_valgrind(); t10_quality()
    t11_repro(); t12_timing()
    elapsed=time.time()-start
    print(f"\n{B}{'#'*70}")
    print(f"  {pt}/{tt} passed ({elapsed:.1f}s)")
    if pt==tt: print(f"  \033[92m*** ALL {tt} TESTS PASSED ***\033[0m")
    else:
        print(f"  \033[91m{tt-pt} FAILED:\033[0m")
        for f in fl: print(f"    - {f}")
    print(f"{'#'*70}{R}")

if __name__=='__main__': main()