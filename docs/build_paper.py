"""Assemble docs/journal_ieee_access/main.tex from the v1 head/tail and the new parts, then compile.

    .venv/bin/python docs/make_results.py && .venv/bin/python docs/build_p2b.py && .venv/bin/python docs/build_paper.py

Table-row files are inlined (an \\input ending in \\\\ directly before \\bottomrule trips LaTeX).
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "journal_ieee_access")
os.chdir(J)

v1 = open("main_v1_backup.tex").read()
head = v1[: v1.index("\\section{Introduction}")]
tail = v1[v1.index("\\section*{Acknowledgment}"):]

TITLE = ("Cross-Attention Fusion of 3D MRI and 1D Electrophysiology for Vision-Pathway Disorders: "
         "A Controlled Benchmark and a Reproducible Multi-Cohort Evaluation")
head = re.sub(r"\\title\{.*?\}\n", lambda m: "\\title{" + TITLE + "}\n", head, count=1, flags=re.S)
head = head.replace("\\input{numbers}", "\\input{numbers}\n\\input{numbers_results}")
head = head.replace("Cross-Attention Fusion of 3D MRI and Visual Evoked Potentials",
                    "Cross-Attention Fusion of 3D MRI and 1D Electrophysiology")
head = head.replace("{M. Jain: Cross-Attention Fusion of 3D MRI and 1D Electrophysiology}",
                    "{M. Jain: Cross-Attention Fusion of 3D MRI and 1D Electrophysiology}")
abstract = open("parts/abstract.tex").read()
head = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}\n", lambda m: abstract, head, flags=re.S)


def part(n):
    return open(f"parts/{n}.tex").read()


body = (head + part("p1_intro") + "\n\\section{Results}\\label{sec:results}\n" + part("p2b_synthetic") + "\n" +
        part("p2c_real") + "\n" + part("p2a_engineering") + "\n" + part("p3_discussion") + "\n" + tail)


def inline(m):
    path = ("../" if m.group(1).startswith("results_assets") else "") + m.group(1) + ".tex"
    return open(path).read().strip() + "\n"


body = re.sub(r"\\input\{(results_assets/tab_[A-Za-z0-9_]+|tab_params_v2)\}", inline, body)
import shutil, glob
for f in glob.glob("../results_assets/fig_*.png"):
    shutil.copy(f, "figs/")
body = body.replace("results_assets/fig_", "figs/fig_")
open("main.tex", "w").write(body)
r = subprocess.run(["tectonic", "-X", "compile", "main.tex"], capture_output=True, text=True)
errs = [l for l in (r.stderr + r.stdout).splitlines() if l.startswith("error")]
print("compile:", "OK" if r.returncode == 0 else "FAILED", errs[:5])
sys.exit(r.returncode)
