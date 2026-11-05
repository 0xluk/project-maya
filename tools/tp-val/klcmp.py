#!/usr/bin/env python3
"""klcmp.py [--n N | --vocab V] [--json OUT] REF.f32 RUN.f32 [RUN.f32 ...]: compare STRATA_GLM_KL_DUMP files
(n x n_vocab F32 raw logits).  Per RUN vs REF: KL(ref||run) per position in nats - mean / p99 / max - and top-1
agreement %.  Needs numpy.  --json writes the numbers as a list of {ref, run, kl_mean, kl_p99, kl_max, top1, n}."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--vocab", type=int, default=0)
ap.add_argument("--n", type=int, default=0, help="positions (alternative to --vocab)")
ap.add_argument("--json", default="")
a = ap.parse_args()
sz = os.path.getsize(a.files[0]) // 4
if not a.vocab and not a.n:
    b = os.path.basename(a.files[0]).split(".")[0]
    a.n = int(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.files[0]))), b + ".n")).read())
V = a.vocab or sz // a.n


def load(f):
    x = np.fromfile(f, dtype=np.float32).astype(np.float64).reshape(-1, V)
    m = x.max(1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(1, keepdims=True)))


ref = load(a.files[0])
pr = np.exp(ref)
print(f"ref {os.path.basename(a.files[0])}: {ref.shape[0]} pos x {V} vocab")
out = []
for f in a.files[1:]:
    x = load(f)
    if x.shape != ref.shape:
        print(f"{os.path.basename(f)}: shape {x.shape} != {ref.shape}")
        out.append(dict(ref=a.files[0], run=f, error="shape"))
        continue
    finite = bool(np.isfinite(x).all())
    kl = (pr * (ref - x)).sum(1)
    t1 = float((ref.argmax(1) == x.argmax(1)).mean() * 100)
    r = dict(ref=a.files[0], run=f, n=int(ref.shape[0]), kl_mean=float(kl.mean()),
             kl_p99=float(np.percentile(kl, 99)), kl_max=float(kl.max()), top1=t1, finite=finite)
    out.append(r)
    print(f"{os.path.basename(f)}: KL mean {r['kl_mean']:.3e} p99 {r['kl_p99']:.3e} max {r['kl_max']:.3e} "
          f"| top1 agree {t1:.2f}%" + ("" if finite else " | NON-FINITE logits"))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
