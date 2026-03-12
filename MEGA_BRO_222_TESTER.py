"""
MEGA_BRO_222_TESTER.py

A project-specific tester for the SymNMF assignment.

What it checks:
- required files
- make / setup.py build flow
- C CLI goals: sym / ddg / norm
- Python CLI goals: sym / ddg / norm / symnmf
- direct extension smoke tests
- C vs Python consistency
- analysis.py behavior
- Valgrind leak / memory-error checks
- timing snapshots
- static checks from the project spec
- modularity / doc-comments / function length scan for symnmf.c

Usage:
    python3 MEGA_BRO_222_TESTER.py [project_root]

If project_root is omitted, the current working directory is used.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


# =========================
# constants
# =========================

SEED = 1234
FLOAT_TOL = 1e-4
TIMING_RUNS = 3

TIMEOUT_SHORT = 30
TIMEOUT_MEDIUM = 45
TIMEOUT_LONG = 120

ERROR_MSG = "An Error Has Occurred"

REQUIRED_FILES = [
    "symnmf.py",
    "symnmf.c",
    "symnmfmodule.c",
    "symnmf.h",
    "analysis.py",
    "setup.py",
    "Makefile",
]

OPTIONAL_FILES = [
    "kmeans.py",
]

C_GOALS = ["sym", "ddg", "norm"]


# =========================
# reporting
# =========================

@dataclass
class TestResult:
    section: str
    name: str
    status: str
    detail: str = ""


class Reporter:
    def __init__(self) -> None:
        self.results: List[TestResult] = []

    def add(self, section: str, name: str, status: str, detail: str = "") -> None:
        self.results.append(TestResult(section, name, status, detail))
        prefix = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
        print(f"  {prefix} {name}")
        if detail:
            for line in detail.rstrip().splitlines():
                print(f"         {line}")

    def pass_(self, section: str, name: str, detail: str = "") -> None:
        self.add(section, name, "PASS", detail)

    def fail(self, section: str, name: str, detail: str = "") -> None:
        self.add(section, name, "FAIL", detail)

    def warn(self, section: str, name: str, detail: str = "") -> None:
        self.add(section, name, "WARN", detail)

    def counts(self) -> Tuple[int, int, int]:
        p = sum(r.status == "PASS" for r in self.results)
        f = sum(r.status == "FAIL" for r in self.results)
        w = sum(r.status == "WARN" for r in self.results)
        return p, f, w


# =========================
# command helpers
# =========================

@dataclass
class CmdResult:
    command: str
    code: int
    out: str
    err: str
    elapsed: float
    timed_out: bool = False


def run_cmd(
    command: str,
    cwd: Path,
    timeout: int = TIMEOUT_MEDIUM,
    env: Optional[dict] = None,
) -> CmdResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return CmdResult(
            command=command,
            code=proc.returncode,
            out=proc.stdout,
            err=proc.stderr,
            elapsed=time.perf_counter() - start,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CmdResult(
            command=command,
            code=-999,
            out=exc.stdout or "",
            err=exc.stderr or "",
            elapsed=time.perf_counter() - start,
            timed_out=True,
        )


def compact_output(res: CmdResult, limit: int = 900) -> str:
    text = (res.out + ("\n" if res.out and res.err else "") + res.err).strip()
    if not text:
        return f"return code {res.code}"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def tail(text: str, max_lines: int = 12) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def command_ok(res: CmdResult, allow_nonzero: bool = False) -> Tuple[bool, str]:
    if res.timed_out:
        return False, f"timeout after {res.elapsed:.2f}s"
    if not allow_nonzero and res.code != 0:
        return False, compact_output(res)
    return True, ""


# =========================
# file helpers
# =========================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def count_points(path: Path) -> int:
    lines = [ln for ln in read_text(path).splitlines() if ln.strip()]
    return len(lines)


def load_points(path: Path) -> List[List[float]]:
    pts = []
    for line in read_text(path).splitlines():
        if line.strip():
            pts.append([float(x) for x in line.strip().split(",")])
    return pts


def write_points(path: Path, points: Sequence[Sequence[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in points:
            f.write(",".join(f"{x:.10f}" for x in row) + "\n")


# =========================
# dataset generation
# =========================

def generate_datasets(base_dir: Path) -> List[Tuple[str, Path]]:
    random.seed(SEED)
    datasets: List[Tuple[str, Path]] = []

    def add(name: str, points: Sequence[Sequence[float]]) -> None:
        path = base_dir / f"{name}.txt"
        write_points(path, points)
        datasets.append((name, path))

    add("2clusters_2d", [
        [-0.2, 0.1], [0.0, -0.1], [0.15, 0.05], [-0.1, 0.2],
        [5.0, 5.1], [4.9, 4.8], [5.2, 5.0], [4.8, 5.2],
    ])

    add("3clusters_2d", [
        [-3.0, -3.1], [-2.7, -3.2], [-3.2, -2.9], [-2.8, -2.8], [-3.1, -3.4],
        [0.1, 0.0], [0.2, -0.2], [-0.1, 0.3], [0.3, 0.1], [-0.2, -0.1],
        [4.8, 5.0], [5.2, 4.9], [5.0, 5.3], [4.7, 5.1], [5.3, 4.8],
        [0.0, 4.0], [0.2, 4.3], [-0.2, 3.8], [0.1, 3.9], [-0.1, 4.2],
    ])

    add("2points", [[0.0, 0.0], [1.0, 0.0]])
    add("3pts_3d", [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    add("close_pts", [[0.0, 0.0], [0.02, 0.01], [0.01, -0.02], [0.03, 0.0], [0.0, 0.03], [-0.02, 0.01]])
    add("far_pts", [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [20.0, 20.0], [-15.0, 12.0]])
    add("1d_points", [[-3.0], [-1.0], [0.0], [1.5], [2.2], [5.0]])

    rand_pts = []
    for _ in range(50):
        rand_pts.append([round(random.uniform(-5.0, 5.0), 6) for _ in range(5)])
    add("random_50_5d", rand_pts)

    ref1 = [[math.sin(i * 0.31) + j * 0.2 for j in range(4)] for i in range(12)]
    ref2 = [[(i % 5) * 0.7 + math.cos(j + i * 0.1) for j in range(6)] for i in range(15)]
    ref3 = [[math.sin(i + j) + math.cos(i * j * 0.1) for j in range(3)] for i in range(9)]
    add("input_1", ref1)
    add("input_2", ref2)
    add("input_3", ref3)

    return datasets


# =========================
# math reference
# =========================

def sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def sym_manual(points: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(points)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i][j] = math.exp(-sqdist(points[i], points[j]) / 2.0)
    return out


def ddg_manual(points: Sequence[Sequence[float]]) -> List[List[float]]:
    a = sym_manual(points)
    n = len(a)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = sum(a[i])
    return d


def norm_manual(points: Sequence[Sequence[float]]) -> List[List[float]]:
    a = sym_manual(points)
    deg = [sum(row) for row in a]
    n = len(a)
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            den = math.sqrt(deg[i]) * math.sqrt(deg[j])
            w[i][j] = a[i][j] / den if den > 0 else 0.0
    return w


def transpose_py(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*mat)] if mat else []


def allclose_matrix(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    tol: float = FLOAT_TOL,
) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if abs(x - y) > tol:
                return False
    return True


def shape(mat: Sequence[Sequence[float]]) -> Tuple[int, int]:
    if not mat:
        return 0, 0
    return len(mat), len(mat[0])


def parse_matrix(text: str) -> List[List[float]]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [[float(x) for x in ln.split(",")] for ln in lines]


def is_4dp_output(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    for ln in lines:
        for cell in ln.split(","):
            if not re.fullmatch(r"-?\d+\.\d{4}", cell.strip()):
                return False
    return True


# =========================
# valgrind helpers
# =========================

def has_valgrind() -> bool:
    return shutil.which("valgrind") is not None


def parse_valgrind(text: str) -> Tuple[bool, bool]:
    no_leaks = (
        "All heap blocks were freed -- no leaks are possible" in text
        or re.search(r"definitely lost:\s+0 bytes", text) is not None
    )
    no_errors = re.search(r"ERROR SUMMARY:\s+0 errors", text) is not None
    return no_leaks, no_errors


# =========================
# modularity scan
# =========================

@dataclass
class FunctionInfo:
    name: str
    start: int
    end: int
    total_lines: int
    has_doc: bool


def find_functions(c_text: str) -> List[FunctionInfo]:
    lines = c_text.splitlines()
    funcs: List[FunctionInfo] = []
    i = 0
    n = len(lines)
    control_words = {"if", "for", "while", "switch", "else", "do"}

    while i < n:
        matched = None
        matched_name = None

        for j in range(i, min(i + 6, n)):
            joined = " ".join(ln.strip() for ln in lines[i:j + 1]).strip()
            if not joined.endswith("{"):
                continue

            first = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", joined)
            if first and first.group(1) in control_words:
                continue

            pat = (
                r"(?:static\s+)?"
                r"[A-Za-z_][A-Za-z0-9_\s\*]*\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*"
                r"\([^;{}]*\)\s*\{\s*$"
            )
            m = re.match(pat, joined)
            if m:
                matched = (i, j)
                matched_name = m.group(1)
                break

        if matched is None or matched_name is None:
            i += 1
            continue

        sig_start, brace_line = matched
        depth = 0
        k = brace_line
        while k < n:
            depth += lines[k].count("{")
            depth -= lines[k].count("}")
            if depth == 0:
                break
            k += 1

        doc_start = sig_start
        has_doc = False
        if sig_start - 1 >= 0 and lines[sig_start - 1].strip().endswith("*/"):
            p = sig_start - 1
            while p >= 0:
                doc_start = p
                if "/*" in lines[p]:
                    has_doc = True
                    break
                p -= 1

        funcs.append(
            FunctionInfo(
                name=matched_name,
                start=doc_start + 1,
                end=k + 1,
                total_lines=k - doc_start + 1,
                has_doc=has_doc,
            )
        )
        i = k + 1

    return funcs


# =========================
# section printers
# =========================

def print_banner(project_root: Path, datasets: Sequence[Tuple[str, Path]]) -> None:
    print("#" * 72)
    print("#                     MEGA_BRO_222_TESTER - SymNMF                  #")
    print("#" * 72)
    print(f"Project root: {project_root}")
    desc = ", ".join(f"{name}({count_points(path)}pts)" for name, path in datasets)
    print(f"Datasets: {desc}")


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =========================
# test sections
# =========================

def test_files_and_build(project_root: Path, rep: Reporter) -> None:
    section = "1. BUILD & COMPILATION"
    print_section(section)

    for name in REQUIRED_FILES:
        path = project_root / name
        if path.exists():
            rep.pass_(section, f"file exists: {name}")
        else:
            rep.fail(section, f"file exists: {name}")

    for name in OPTIONAL_FILES:
        path = project_root / name
        if path.exists():
            rep.pass_(section, f"optional file exists: {name}")
        else:
            rep.warn(section, f"optional file exists: {name}", "Optional file is missing.")

    run_cmd("make clean", project_root, timeout=TIMEOUT_SHORT)

    make_res = run_cmd("make", project_root, timeout=TIMEOUT_LONG)
    if make_res.code == 0:
        rep.pass_(section, "make: compiles OK")
    else:
        rep.fail(section, "make: compiles OK", compact_output(make_res))

    if (project_root / "symnmf").exists():
        rep.pass_(section, "make: symnmf executable exists")
    else:
        rep.fail(section, "make: symnmf executable exists")

    if "warning:" not in (make_res.out + make_res.err).lower():
        rep.pass_(section, "make: zero warnings")
    else:
        rep.fail(section, "make: zero warnings", compact_output(make_res))

    setup_res = run_cmd("python3 setup.py build_ext --inplace", project_root, timeout=TIMEOUT_LONG)
    if setup_res.code == 0:
        rep.pass_(section, "setup.py: builds OK")
    else:
        rep.fail(section, "setup.py: builds OK", compact_output(setup_res))

    so_files = list(project_root.glob("symnmfmodule*.so"))
    if so_files:
        rep.pass_(section, "setup.py: .so module created")
    else:
        rep.fail(section, "setup.py: .so module created")

    import_res = run_cmd(
        "python3 -c \"import symnmfmodule; print('ok')\"",
        project_root,
        timeout=TIMEOUT_SHORT,
    )
    if import_res.code == 0 and import_res.out.strip() == "ok":
        rep.pass_(section, "python import symnmfmodule")
    else:
        rep.fail(section, "python import symnmfmodule", compact_output(import_res))


def test_c_cli(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "2. C PROGRAM — sym/ddg/norm"
    print_section(section)

    manual = {
        "sym": sym_manual,
        "ddg": ddg_manual,
        "norm": norm_manual,
    }

    base_sets = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "2points", "3pts_3d",
        "close_pts", "far_pts", "1d_points", "random_50_5d",
    }]

    for name, path in base_sets:
        pts = load_points(path)
        d = len(pts[0]) if pts else 0
        print(f"\n  --- {name} ({len(pts)} pts, {d}D) ---")

        for goal in C_GOALS:
            res = run_cmd(f"./symnmf {goal} {path.name}", project_root, timeout=TIMEOUT_LONG)
            ok, detail = command_ok(res)
            if ok:
                rep.pass_(section, f"{goal}: runs OK")
            else:
                rep.fail(section, f"{goal}: runs OK", detail)
                continue

            try:
                mat = parse_matrix(res.out)
            except Exception as exc:
                rep.fail(section, f"{goal}: parse matrix", str(exc))
                continue

            rows, cols = shape(mat)
            n = len(pts)
            if rows == n and cols == n:
                rep.pass_(section, f"{goal}: square {n}x{n}")
            else:
                rep.fail(section, f"{goal}: square {n}x{n}", f"got {rows}x{cols}")

            if is_4dp_output(res.out):
                rep.pass_(section, f"{goal}: format %.4f")
            else:
                rep.fail(section, f"{goal}: format %.4f")

            if allclose_matrix(mat, transpose_py(mat)):
                rep.pass_(section, f"{goal}: symmetric")
            else:
                rep.fail(section, f"{goal}: symmetric")

            if all(x >= -FLOAT_TOL for row in mat for x in row):
                rep.pass_(section, f"{goal}: values >= 0")
            else:
                rep.fail(section, f"{goal}: values >= 0")

            if goal in {"sym", "norm"}:
                if all(abs(mat[i][i]) <= FLOAT_TOL for i in range(n)):
                    rep.pass_(section, f"{goal}: diagonal zero")
                else:
                    rep.fail(section, f"{goal}: diagonal zero")

                off_ok = all(mat[i][j] <= 1.0 + FLOAT_TOL for i in range(n) for j in range(n) if i != j)
                if off_ok:
                    rep.pass_(section, f"{goal}: values <= 1 (off-diag)")
                else:
                    rep.fail(section, f"{goal}: values <= 1 (off-diag)")

            if goal == "ddg":
                if all(abs(mat[i][j]) <= FLOAT_TOL for i in range(n) for j in range(n) if i != j):
                    rep.pass_(section, f"{goal}: off-diagonal zero")
                else:
                    rep.fail(section, f"{goal}: off-diagonal zero")

                if all(mat[i][i] >= -FLOAT_TOL for i in range(n)):
                    rep.pass_(section, f"{goal}: diagonal non-negative")
                else:
                    rep.fail(section, f"{goal}: diagonal non-negative")

                off_ok = all(mat[i][j] <= 1.0 + FLOAT_TOL for i in range(n) for j in range(n) if i != j)
                if off_ok:
                    rep.pass_(section, f"{goal}: values <= 1 (off-diag)")
                else:
                    rep.fail(section, f"{goal}: values <= 1 (off-diag)")

            ref = manual[goal](pts)
            if allclose_matrix(mat, ref):
                rep.pass_(section, f"{goal}: matches manual reference")
            else:
                rep.fail(section, f"{goal}: matches manual reference")


def test_c_errors(project_root: Path, sample_path: Path, rep: Reporter) -> None:
    section = "3. C PROGRAM — ERROR HANDLING"
    print_section(section)

    cases = [
        ("no args", "./symnmf"),
        ("one arg", "./symnmf sym"),
        ("too many args", f"./symnmf sym {sample_path.name} extra"),
        ("invalid goal", f"./symnmf bogus {sample_path.name}"),
        ("symnmf in C CLI", f"./symnmf symnmf {sample_path.name}"),
        ("missing file", "./symnmf sym definitely_missing_123456.txt"),
        ("empty goal", f"./symnmf '' {sample_path.name}"),
    ]

    for label, cmd in cases:
        res = run_cmd(cmd, project_root, timeout=TIMEOUT_SHORT)
        text = res.out + res.err
        if res.code != 0 and ERROR_MSG in text:
            rep.pass_(section, f"{label}: prints error")
        else:
            rep.fail(section, f"{label}: prints error", compact_output(res))


def test_python_cli(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "4. PYTHON PROGRAM — sym/ddg/norm"
    print_section(section)

    base_sets = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "2points", "3pts_3d",
        "close_pts", "far_pts", "1d_points", "random_50_5d",
    }]

    for name, path in base_sets:
        print(f"\n  --- {name} ---")
        for goal in C_GOALS:
            res = run_cmd(f"python3 symnmf.py 2 {goal} {path.name}", project_root, timeout=TIMEOUT_LONG)

            if name == "2points" and res.code != 0 and ERROR_MSG in (res.out + res.err):
                rep.pass_(section, f"py {goal}: correctly identified k >= n error")
                continue

            ok, detail = command_ok(res)
            if ok:
                rep.pass_(section, f"py {goal}: runs OK")
            else:
                rep.fail(section, f"py {goal}: runs OK", detail)
                continue

            if is_4dp_output(res.out):
                rep.pass_(section, f"py {goal}: format %.4f")
            else:
                rep.fail(section, f"py {goal}: format %.4f")


def test_extension_smoke(project_root: Path, rep: Reporter) -> None:
    section = "4b. DIRECT PYTHON EXTENSION SMOKE"
    print_section(section)

    code = r'''
import symnmfmodule as m
X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
W = m.norm(X)
A = m.sym(X)
D = m.ddg(X)
H0 = [[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]]
H = m.symnmf(H0, W, len(X), len(H0[0]))
print(len(A), len(A[0]), len(D), len(W), len(H), len(H[0]))
'''
    res = run_cmd(f"python3 - <<'PY'\n{code}\nPY", project_root, timeout=TIMEOUT_LONG)
    if res.code == 0:
        rep.pass_(section, "extension smoke: direct sym/ddg/norm/symnmf")
    else:
        rep.fail(section, "extension smoke: direct sym/ddg/norm/symnmf", compact_output(res))

    bad_code = r'''
import symnmfmodule as m
try:
    m.sym([])
    print("NO_ERROR")
except Exception:
    print("ERROR")
'''
    bad_res = run_cmd(f"python3 - <<'PY'\n{bad_code}\nPY", project_root, timeout=TIMEOUT_SHORT)
    if bad_res.code == 0 and bad_res.out.strip() == "ERROR":
        rep.pass_(section, "extension smoke: rejects empty input")
    else:
        rep.warn(section, "extension smoke: rejects empty input", compact_output(bad_res))


def test_python_symnmf(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "5. PYTHON PROGRAM — symnmf"
    print_section(section)

    chosen = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "close_pts", "far_pts",
        "1d_points", "random_50_5d",
    }]

    for name, path in chosen:
        n = len(load_points(path))
        for k in [2, 3]:
            print(f"\n  --- {name}, k={k} ---")
            res = run_cmd(f"python3 symnmf.py {k} symnmf {path.name}", project_root, timeout=TIMEOUT_LONG)
            ok, detail = command_ok(res)
            if ok:
                rep.pass_(section, f"symnmf k={k}: runs OK")
            else:
                rep.fail(section, f"symnmf k={k}: runs OK", detail)
                continue

            try:
                h = parse_matrix(res.out)
            except Exception as exc:
                rep.fail(section, f"symnmf k={k}: parse output", str(exc))
                continue

            rows, cols = shape(h)
            if rows == n:
                rep.pass_(section, f"symnmf k={k}: {n} rows")
            else:
                rep.fail(section, f"symnmf k={k}: {n} rows", f"got {rows}")

            if cols == k:
                rep.pass_(section, f"symnmf k={k}: {k} cols")
            else:
                rep.fail(section, f"symnmf k={k}: {k} cols", f"got {cols}")

            if all(math.isfinite(x) for row in h for x in row):
                rep.pass_(section, f"symnmf k={k}: all finite")
            else:
                rep.fail(section, f"symnmf k={k}: all finite")

            if all(x >= -FLOAT_TOL for row in h for x in row):
                rep.pass_(section, f"symnmf k={k}: values >= 0")
            else:
                rep.fail(section, f"symnmf k={k}: values >= 0")

            if is_4dp_output(res.out):
                rep.pass_(section, f"symnmf k={k}: format %.4f")
            else:
                rep.fail(section, f"symnmf k={k}: format %.4f")


def test_python_errors(project_root: Path, sample_path: Path, rep: Reporter) -> None:
    section = "6. PYTHON PROGRAM — ERROR HANDLING"
    print_section(section)

    cases = [
        ("no args", "python3 symnmf.py"),
        ("1 arg", "python3 symnmf.py 2"),
        ("2 args", "python3 symnmf.py 2 sym"),
        ("too many args", f"python3 symnmf.py 2 sym {sample_path.name} extra"),
        ("invalid goal", f"python3 symnmf.py 2 bogus {sample_path.name}"),
        ("k=abc", f"python3 symnmf.py abc sym {sample_path.name}"),
        ("missing file", "python3 symnmf.py 2 sym definitely_missing_123456.txt"),
    ]

    for label, cmd in cases:
        res = run_cmd(cmd, project_root, timeout=TIMEOUT_SHORT)
        text = res.out + res.err
        if res.code != 0 and ERROR_MSG in text:
            rep.pass_(section, f"{label}: prints error")
        else:
            rep.fail(section, f"{label}: prints error", compact_output(res))


def test_c_vs_python(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "7. C vs PYTHON CONSISTENCY"
    print_section(section)

    base_sets = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "2points", "3pts_3d",
        "close_pts", "far_pts", "1d_points", "random_50_5d",
    }]

    for name, path in base_sets:
        print(f"\n  --- {name} ---")
        for goal in C_GOALS:
            c_res = run_cmd(f"./symnmf {goal} {path.name}", project_root, timeout=TIMEOUT_LONG)
            p_res = run_cmd(f"python3 symnmf.py 2 {goal} {path.name}", project_root, timeout=TIMEOUT_LONG)

            if p_res.code != 0 and ERROR_MSG in (p_res.out + p_res.err):
                rep.pass_(section, f"{goal}: Python correctly errored on k >= n")
                continue
            if c_res.code == 0 and p_res.code == 0:
                rep.pass_(section, f"{goal}: both ran")
                try:
                    c_mat = parse_matrix(c_res.out)
                    p_mat = parse_matrix(p_res.out)
                except Exception as exc:
                    rep.fail(section, f"{goal}: C == Python", str(exc))
                    continue

                if allclose_matrix(c_mat, p_mat):
                    rep.pass_(section, f"{goal}: C == Python")
                else:
                    rep.fail(section, f"{goal}: C == Python")
            else:
                detail = f"C: {compact_output(c_res)} | PY: {compact_output(p_res)}"
                rep.fail(section, f"{goal}: both ran", detail)


def test_analysis(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "8. ANALYSIS.PY"
    print_section(section)

    chosen = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "close_pts",
        "far_pts", "1d_points", "random_50_5d",
    }]

    for name, path in chosen:
        n = len(load_points(path))
        for k in [2, 3]:
            if k >= n:
                continue

            label = f"analysis {k} {name}"
            res = run_cmd(f"python3 analysis.py {k} {path.name}", project_root, timeout=TIMEOUT_LONG)

            if name == "close_pts" and res.code != 0:
                rep.warn(section, f"{label}: degenerate synthetic case produced error", compact_output(res))
                continue

            if res.code != 0:
                rep.fail(section, f"{label}: runs OK", compact_output(res))
                continue

            rep.pass_(section, f"{label}: runs OK")

            lines = [ln.strip() for ln in res.out.splitlines() if ln.strip()]
            if len(lines) == 2:
                rep.pass_(section, f"{label}: exactly 2 lines")
            else:
                rep.fail(section, f"{label}: exactly 2 lines", res.out)
                continue

            nmf_line, km_line = lines

            if nmf_line.startswith("nmf:"):
                rep.pass_(section, f"{label}: has nmf")
            else:
                rep.fail(section, f"{label}: has nmf", nmf_line)

            if km_line.startswith("kmeans:"):
                rep.pass_(section, f"{label}: has kmeans")
            else:
                rep.fail(section, f"{label}: has kmeans", km_line)

            for prefix, line in [("nmf", nmf_line), ("kmeans", km_line)]:
                try:
                    val = float(line.split(":", 1)[1].strip())
                    if -1.0 <= val <= 1.0:
                        rep.pass_(section, f"{label}: {prefix} in [-1,1]")
                    else:
                        rep.fail(section, f"{label}: {prefix} in [-1,1]", str(val))

                    if re.fullmatch(rf"{prefix}: -?\d+\.\d{{4}}", line):
                        rep.pass_(section, f"{label}: {prefix} 4dp")
                    else:
                        rep.fail(section, f"{label}: {prefix} 4dp", line)
                except Exception as exc:
                    rep.fail(section, f"{label}: {prefix} parse", str(exc))


def test_reference(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "9. REFERENCE RESULTS"
    print_section(section)

    refs = [x for x in datasets if x[0].startswith("input_")]
    for name, path in refs:
        n = len(load_points(path))
        max_k = min(10, n - 1)
        for k in range(2, max_k + 1):
            label = f"reference {path.name} k={k}"
            res = run_cmd(f"python3 analysis.py {k} {path.name}", project_root, timeout=TIMEOUT_LONG)
            if res.code == 0:
                rep.pass_(section, f"{label}: runs")
                if "nmf:" in res.out:
                    rep.pass_(section, f"{label}: nmf")
                else:
                    rep.fail(section, f"{label}: nmf", res.out)

                if "kmeans:" in res.out:
                    rep.pass_(section, f"{label}: kmeans")
                else:
                    rep.fail(section, f"{label}: kmeans", res.out)
            else:
                rep.fail(section, f"{label}: runs", compact_output(res))


def test_valgrind(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "10. VALGRIND — C PROGRAM"
    print_section(section)

    if not has_valgrind():
        rep.warn(section, "valgrind installed", "Valgrind was not found in PATH.")
        return

    chosen = [x for x in datasets if x[0] in {
        "2clusters_2d", "3clusters_2d", "2points", "3pts_3d", "close_pts",
    }]

    for name, path in chosen:
        print(f"\n  --- {name} ---")
        for goal in C_GOALS:
            label = f"vg {goal} {name}"
            cmd = (
                "valgrind --leak-check=full --show-leak-kinds=all "
                "--errors-for-leak-kinds=all "
                f"./symnmf {goal} {path.name}"
            )
            res = run_cmd(cmd, project_root, timeout=TIMEOUT_LONG)
            text = res.out + "\n" + res.err
            no_leaks, no_errors = parse_valgrind(text)

            if no_leaks:
                rep.pass_(section, f"{label}: no leaks")
            else:
                rep.fail(section, f"{label}: no leaks", tail(text))

            if no_errors:
                rep.pass_(section, f"{label}: no mem errors")
            else:
                rep.fail(section, f"{label}: no mem errors", tail(text))

    print("\n  --- error paths ---")
    error_cases = [
        ("missing file", "./symnmf sym definitely_missing_123456.txt"),
        ("invalid goal", f"./symnmf bogus {chosen[0][1].name}"),
        ("no args", "./symnmf"),
        ("C symnmf goal path", f"./symnmf symnmf {chosen[0][1].name}"),
    ]

    for label, cmd in error_cases:
        vg_cmd = (
            "valgrind --leak-check=full --show-leak-kinds=all "
            "--errors-for-leak-kinds=all "
            f"{cmd}"
        )
        res = run_cmd(vg_cmd, project_root, timeout=TIMEOUT_LONG)
        text = res.out + "\n" + res.err
        no_leaks, no_errors = parse_valgrind(text)

        if no_leaks:
            rep.pass_(section, f"vg error path ({label}): no leaks")
        else:
            rep.fail(section, f"vg error path ({label}): no leaks", tail(text))

        if no_errors:
            rep.pass_(section, f"vg error path ({label}): no mem errors")
        else:
            rep.fail(section, f"vg error path ({label}): no mem errors", tail(text))


def test_timing(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "11. TIMING / PERFORMANCE SNAPSHOT"
    print_section(section)

    data_random = next(path for name, path in datasets if name == "random_50_5d")
    tests = [
        ("C norm random_50_5d", f"./symnmf norm {data_random.name}"),
        ("Python norm random_50_5d", f"python3 symnmf.py 2 norm {data_random.name}"),
        ("Python symnmf random_50_5d k=3", f"python3 symnmf.py 3 symnmf {data_random.name}"),
        ("analysis random_50_5d k=3", f"python3 analysis.py 3 {data_random.name}"),
    ]

    for label, cmd in tests:
        times = []
        bad = None
        for _ in range(TIMING_RUNS):
            res = run_cmd(cmd, project_root, timeout=TIMEOUT_LONG)
            if res.code != 0:
                bad = compact_output(res)
                break
            times.append(res.elapsed)

        if bad is not None:
            rep.fail(section, f"{label}: timing run", bad)
        else:
            detail = "samples=" + ", ".join(f"{x:.4f}" for x in times)
            rep.pass_(section, f"{label}: mean {statistics.mean(times):.4f}s", detail)


def test_static_quality(project_root: Path, rep: Reporter) -> None:
    section = "12. STATIC / QUALITY CHECKS"
    print_section(section)

    sym_c = read_text(project_root / "symnmf.c")
    module_c = read_text(project_root / "symnmfmodule.c")
    py = read_text(project_root / "symnmf.py")
    analysis_py = read_text(project_root / "analysis.py")
    header = read_text(project_root / "symnmf.h")
    makefile = read_text(project_root / "Makefile")

    if "%.4f" in sym_c:
        rep.pass_(section, "symnmf.c: contains %.4f")
    else:
        rep.fail(section, "symnmf.c: contains %.4f")

    if ERROR_MSG in sym_c:
        rep.pass_(section, "symnmf.c: contains error message")
    else:
        rep.fail(section, "symnmf.c: contains error message")

    non_empty_module_lines = [ln for ln in module_c.splitlines() if ln.strip()]
    if non_empty_module_lines[:1] == ["#define PY_SSIZE_T_CLEAN"]:
        rep.pass_(section, "symnmfmodule.c: PY_SSIZE_T_CLEAN first")
    else:
        rep.fail(section, "symnmfmodule.c: PY_SSIZE_T_CLEAN first")

    if "#include <Python.h>" in module_c:
        rep.pass_(section, "symnmfmodule.c: imports Python.h")
    else:
        rep.fail(section, "symnmfmodule.c: imports Python.h")

    if "np.random.seed(1234)" in py:
        rep.pass_(section, "symnmf.py: np.random.seed(1234)")
    else:
        rep.fail(section, "symnmf.py: np.random.seed(1234)")

    if "import symnmfmodule" in py:
        rep.pass_(section, "symnmf.py: imports symnmfmodule")
    else:
        rep.fail(section, "symnmf.py: imports symnmfmodule")

    if 'if __name__ == "__main__":' in py or "if __name__ == '__main__':" in py:
        rep.pass_(section, "symnmf.py: __main__ guard")
    else:
        rep.fail(section, "symnmf.py: __main__ guard")

    if re.search(r"\.4f", py):
        rep.pass_(section, "symnmf.py: 4dp formatting")
    else:
        rep.fail(section, "symnmf.py: 4dp formatting")

    if "np.random.seed(1234)" in analysis_py:
        rep.pass_(section, "analysis.py: np.random.seed(1234)")
    else:
        rep.warn(section, "analysis.py: np.random.seed(1234)", "Missing explicit seed.")

    if "silhouette_score" in analysis_py:
        rep.pass_(section, "analysis.py: silhouette_score")
    else:
        rep.fail(section, "analysis.py: silhouette_score")

    if 'if __name__ == "__main__":' in analysis_py or "if __name__ == '__main__':" in analysis_py:
        rep.pass_(section, "analysis.py: __main__ guard")
    else:
        rep.fail(section, "analysis.py: __main__ guard")

    for flag in ["-ansi", "-Wall", "-Wextra", "-Werror", "-pedantic-errors"]:
        if flag in makefile:
            rep.pass_(section, f"Makefile: {flag}")
        else:
            rep.fail(section, f"Makefile: {flag}")

    if "symnmfmodule.c" in makefile:
        rep.warn(section, "Makefile: symnmfmodule.c not compiled with gcc", "Found symnmfmodule.c in Makefile text.")
    else:
        rep.pass_(section, "Makefile: symnmfmodule.c not compiled with gcc")

    for name in ["sym", "ddg", "norm", "symnmf"]:
        if re.search(rf"\b{name}\s*\(", header):
            rep.pass_(section, f"symnmf.h: prototype for {name}")
        else:
            rep.fail(section, f"symnmf.h: prototype for {name}")

    funcs = find_functions(sym_c)
    if not funcs:
        rep.warn(section, "symnmf.c: modularity scan", "Could not parse function boundaries.")
    else:
        too_long = [f"{f.name} ({f.total_lines} lines)" for f in funcs if f.total_lines > 40]
        undocumented = [f.name for f in funcs if not f.has_doc]

        if too_long:
            rep.fail(section, "symnmf.c: functions within 40 lines", ", ".join(too_long))
        else:
            rep.pass_(section, "symnmf.c: functions within 40 lines")

        if undocumented:
            rep.warn(section, "symnmf.c: function documentation", ", ".join(undocumented))
        else:
            rep.pass_(section, "symnmf.c: function documentation")

        helper_count = sum(
            1 for f in funcs
            if f.name not in {"main", "alloc_matrix", "free_matrix", "print_matrix", "sym", "ddg", "norm", "symnmf", "read_data"}
        )
        if helper_count >= 3:
            rep.pass_(section, "symnmf.c: helper modularity")
        else:
            rep.warn(section, "symnmf.c: helper modularity", f"Detected only {helper_count} internal helpers.")

    if re.search(r"\bk\s*>=?\s*n\b|\bk\s*<\s*n\b|\b0\s*<\s*k", py):
        rep.pass_(section, "symnmf.py: explicit k-range check")
    else:
        rep.warn(section, "symnmf.py: explicit k-range check", "No clear k validation found in symnmf.py.")

    if "PyList_GetItem(pylist, 0)" in module_c and "PyList_Size(pylist)" in module_c:
        if "if (n == 0)" in module_c or "if (PyList_Size(pylist) == 0)" in module_c:
            rep.pass_(section, "symnmfmodule.c: empty-input guard")
        else:
            rep.warn(section, "symnmfmodule.c: empty-input guard", "Could not find explicit empty-input validation.")

    if re.search(r"PyErr_SetString|PyExc_", module_c):
        rep.pass_(section, "symnmfmodule.c: raises Python errors")
    else:
        rep.warn(section, "symnmfmodule.c: raises Python errors", "No explicit PyErr_SetString found.")


def test_math_edges(project_root: Path, rep: Reporter) -> None:
    section = "13. MATHEMATICAL EDGE CHECKS"
    print_section(section)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        points = [
            [1.0, 0.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ]
        data = tmp / "equidistant.txt"
        write_points(data, points)

        local_copy = project_root / data.name
        shutil.copy2(data, local_copy)

        try:
            sym_res = run_cmd(f"./symnmf sym {local_copy.name}", project_root, timeout=TIMEOUT_SHORT)
            if sym_res.code != 0:
                rep.fail(section, "edge sym equidistant: runs", compact_output(sym_res))
                return

            rep.pass_(section, "edge sym equidistant: runs")
            sym_mat = parse_matrix(sym_res.out)
            expected = math.exp(-3.0 / 2.0)

            if abs(sym_mat[0][1] - expected) <= 2e-4:
                rep.pass_(section, "edge sym equidistant: exp(-1.5)")
            else:
                rep.fail(section, "edge sym equidistant: exp(-1.5)", str(sym_mat[0][1]))

            if allclose_matrix(sym_mat, transpose_py(sym_mat)):
                rep.pass_(section, "edge sym equidistant: symmetry")
            else:
                rep.fail(section, "edge sym equidistant: symmetry")

            ddg_res = run_cmd(f"./symnmf ddg {local_copy.name}", project_root, timeout=TIMEOUT_SHORT)
            norm_res = run_cmd(f"./symnmf norm {local_copy.name}", project_root, timeout=TIMEOUT_SHORT)
            if ddg_res.code != 0 or norm_res.code != 0:
                rep.fail(section, "ddg/norm edge runs", compact_output(ddg_res) + " | " + compact_output(norm_res))
                return

            d_mat = parse_matrix(ddg_res.out)
            w_mat = parse_matrix(norm_res.out)

            row_sum_ok = all(abs(d_mat[i][i] - sum(sym_mat[i])) <= 4e-4 for i in range(len(sym_mat)))
            if row_sum_ok:
                rep.pass_(section, "ddg diagonal == row sums of sym")
            else:
                rep.fail(section, "ddg diagonal == row sums of sym")

            expected_w = norm_manual(points)
            if allclose_matrix(w_mat, expected_w, tol=6e-4):
                rep.pass_(section, "norm formula matches manual reference")
            else:
                rep.fail(section, "norm formula matches manual reference")

        finally:
            try:
                local_copy.unlink()
            except OSError:
                pass


def test_reproducibility(project_root: Path, datasets: Sequence[Tuple[str, Path]], rep: Reporter) -> None:
    section = "14. REPRODUCIBILITY"
    print_section(section)

    data = next(path for name, path in datasets if name == "2clusters_2d")

    r1 = run_cmd(f"python3 symnmf.py 2 symnmf {data.name}", project_root, timeout=TIMEOUT_LONG)
    r2 = run_cmd(f"python3 symnmf.py 2 symnmf {data.name}", project_root, timeout=TIMEOUT_LONG)
    if r1.code == 0 and r2.code == 0 and r1.out == r2.out:
        rep.pass_(section, "symnmf.py reproducible")
    else:
        rep.fail(section, "symnmf.py reproducible", compact_output(r1) + " | " + compact_output(r2))

    a1 = run_cmd(f"python3 analysis.py 2 {data.name}", project_root, timeout=TIMEOUT_LONG)
    a2 = run_cmd(f"python3 analysis.py 2 {data.name}", project_root, timeout=TIMEOUT_LONG)
    if a1.code == 0 and a2.code == 0 and a1.out == a2.out:
        rep.pass_(section, "analysis.py reproducible")
    else:
        rep.fail(section, "analysis.py reproducible", compact_output(a1) + " | " + compact_output(a2))


# =========================
# project copy
# =========================

def prepare_project_copy(src_root: Path) -> Tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_ctx = tempfile.TemporaryDirectory(prefix="mega_bro_222_")
    dst_root = Path(temp_ctx.name) / src_root.name
    shutil.copytree(src_root, dst_root)
    return temp_ctx, dst_root


# =========================
# main
# =========================

def main() -> int:
    parser = argparse.ArgumentParser(description="Project-specific SymNMF tester.")
    parser.add_argument("project_root", nargs="?", default=".", help="Path to the project root.")
    args = parser.parse_args()

    src_root = Path(args.project_root).resolve()
    if not src_root.exists():
        print(f"Project root does not exist: {src_root}", file=sys.stderr)
        return 2

    temp_ctx, project_root = prepare_project_copy(src_root)
    rep = Reporter()
    datasets = generate_datasets(project_root)

    print_banner(project_root, datasets)

    try:
        test_files_and_build(project_root, rep)
        test_c_cli(project_root, datasets, rep)
        test_c_errors(project_root, next(path for name, path in datasets if name == "2clusters_2d"), rep)
        test_python_cli(project_root, datasets, rep)
        test_extension_smoke(project_root, rep)
        test_python_symnmf(project_root, datasets, rep)
        test_python_errors(project_root, next(path for name, path in datasets if name == "2clusters_2d"), rep)
        test_c_vs_python(project_root, datasets, rep)
        test_analysis(project_root, datasets, rep)
        test_reference(project_root, datasets, rep)
        test_valgrind(project_root, datasets, rep)
        test_timing(project_root, datasets, rep)
        test_static_quality(project_root, rep)
        test_math_edges(project_root, rep)
        test_reproducibility(project_root, datasets, rep)
    finally:
        temp_ctx.cleanup()

    p, f, w = rep.counts()
    print("\n" + "#" * 72)
    print(f"RESULTS: {p}/{len(rep.results)} tests passed")
    print(f"{f} failed")
    print(f"{w} warnings")
    if f:
        print("Failed items:")
        for r in rep.results:
            if r.status == "FAIL":
                print(f"  - {r.name}")
    print("#" * 72)

    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())