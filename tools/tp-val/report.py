#!/usr/bin/env python3
"""report.py RUN_DIR [--baseline DIR]: summary.md + acceptance verdict for one run_ab.sh leg (see ACCEPTANCE.md).
Exit 0 = ACCEPT, 1 = REJECT, 2 = incomplete (a phase is missing)."""
import argparse, json, os, re, sys
from collections import Counter

# ---- thresholds (ACCEPTANCE.md) ----
KL_REF2B_MEAN, KL_REF2B_TOP1, KL_REF2B_P99 = 2.5e-2, 93.0, 0.45
KL_TWIN2_MEAN = 1e-2
SPEED_FLOOR = 0.97          # TP must not be slower than baseline beyond noise (short + long decode)
TURN_CEIL = 1.10            # turn wall may not grow more than 10% (prefill is unchanged by TP)

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--baseline", default="")
a = ap.parse_args()


def load_bench(d):
    f = os.path.join(d, "bench", "bench.json")
    return {r["tag"]: r for r in json.load(open(f))} if os.path.exists(f) else {}


def coherent(path):
    """Cheap automatic checks for a greedy text; a human still reads them (texts are next to bench.json)."""
    t = open(path, encoding="utf-8", errors="replace").read()
    why = []
    if len(t.strip()) < 300:
        why.append("short")
    if "�" in t:
        why.append("U+FFFD")
    pr = sum(c.isprintable() or c in "\n\t" for c in t) / max(1, len(t))
    if pr < 0.98:
        why.append(f"printable {pr:.2f}")
    w = re.findall(r"\S+", t)
    if len(w) >= 40:
        sh = Counter(tuple(w[i:i + 8]) for i in range(len(w) - 7)).most_common(1)[0][1]
        if sh > 4:
            why.append(f"8-word loop x{sh}")
        if len(set(w)) / len(w) < 0.25:
            why.append(f"low vocab {len(set(w)) / len(w):.2f}")
    return why


B, BL = load_bench(a.run), (load_bench(a.baseline) if a.baseline and os.path.isdir(a.baseline) else {})
kl = json.load(open(os.path.join(a.run, "kl.json"))) if os.path.exists(os.path.join(a.run, "kl.json")) else None
L, ok, incomplete = [f"# {os.path.basename(a.run.rstrip('/'))}", ""], True, False
for f in ("test-bins.sha", "kl-dump.sha"):
    p = os.path.join(a.run, f)
    if os.path.exists(p):
        L += [f"`{f}`:", "```", open(p).read().strip(), "```"]

# ---- speed table ----
def g(d, tag, k):
    return (d.get(tag) or {}).get(k)


rows = [("short0 B-tree", "short0", "decode_tps"), ("short1 intervals", "short1", "decode_tps"),
        ("short2 French Rev.", "short2", "decode_tps"), ("long25k prompt tok/s", "long25k", "prompt_tps"),
        ("long25k decode tok/s", "long25k", "decode_tps"), ("turn wall s", "turn", "wall_s"),
        ("turn new prompt tok", "turn", "prompt_n"), ("turn prompt tok/s", "turn", "prompt_tps"),
        ("turn decode tok/s", "turn", "decode_tps")]
L += ["", "## Speed (server timings; greedy, temperature 0)", "", "| metric | this run | baseline | ratio |", "|---|---|---|---|"]
for name, tag, k in rows:
    v, bv = g(B, tag, k), g(BL, tag, k)
    r = f"{v / bv:.3f}" if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and bv else ""
    L.append(f"| {name} | {v if v is not None else '-'} | {bv if bv is not None else '-'} | {r} |")
sh = [g(B, f"short{i}", "decode_tps") for i in range(3)]
shb = [g(BL, f"short{i}", "decode_tps") for i in range(3)]
mean = lambda x: sum(x) / len(x)
if all(sh):
    acc = [(g(B, f"short{i}", "draft_accepted") or 0) / max(1, g(B, f"short{i}", "draft_n") or 0) for i in range(3)]
    L.append(f"| short mean decode | {mean(sh):.2f} | {mean(shb):.2f} | {mean(sh) / mean(shb):.3f} |" if all(shb)
             else f"| short mean decode | {mean(sh):.2f} | - | |")
    L.append(f"\nMTP acceptance short0-2: " + ", ".join(f"{x * 100:.0f}%" for x in acc) +
             "; long25k prompt " + str(g(B, "long25k", "prompt_n")) + " tok")

# ---- verdicts ----
V = []
def check(name, cond, detail):
    global ok
    V.append(f"- {'PASS' if cond else 'FAIL'} {name}: {detail}")
    ok &= bool(cond)

if kl:
    r, t = kl["ref2b"], kl["twin2"]
    check("KL vs ref2b", r.get("finite", True) and r["kl_mean"] <= KL_REF2B_MEAN and r["top1"] >= KL_REF2B_TOP1
          and r["kl_p99"] <= KL_REF2B_P99,
          f"mean {r['kl_mean']:.3e} (≤{KL_REF2B_MEAN}), p99 {r['kl_p99']:.3f} (≤{KL_REF2B_P99}), "
          f"max {r['kl_max']:.3f}, top-1 {r['top1']:.2f}% (≥{KL_REF2B_TOP1})")
    check("KL vs twin.2", t["kl_mean"] <= KL_TWIN2_MEAN,
          f"mean {t['kl_mean']:.3e} (≤{KL_TWIN2_MEAN}), p99 {t['kl_p99']:.3f}, top-1 {t['top1']:.2f}%"
          + (" (bit-identical)" if t["kl_mean"] == 0 and t["kl_max"] == 0 else ""))
else:
    V.append("- MISSING KL"); incomplete = True
if B:
    for i in range(3):
        p = os.path.join(a.run, "bench", f"text-short{i}.txt")
        if os.path.exists(p):
            why = coherent(p)
            same = ""
            pb = os.path.join(a.baseline, "bench", f"text-short{i}.txt") if a.baseline else ""
            if pb and os.path.exists(pb):
                x, y = open(p).read(), open(pb).read()
                n = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                same = " identical to baseline" if x == y else f", first diff vs baseline at char {n}"
            check(f"greedy text short{i}", not why, ", ".join(why) or f"auto checks ok (n={g(B, f'short{i}', 'n')}){same}")
        else:
            V.append(f"- MISSING text short{i}"); incomplete = True
    if BL and all(sh) and all(shb):
        check("speed short decode", mean(sh) >= SPEED_FLOOR * mean(shb), f"{mean(sh):.2f} vs {mean(shb):.2f} tok/s")
        if g(B, "long25k", "decode_tps") and g(BL, "long25k", "decode_tps"):
            check("speed long25k decode", g(B, "long25k", "decode_tps") >= SPEED_FLOOR * g(BL, "long25k", "decode_tps"),
                  f"{g(B, 'long25k', 'decode_tps')} vs {g(BL, 'long25k', 'decode_tps')} tok/s")
        if g(B, "turn", "wall_s") and g(BL, "turn", "wall_s"):
            check("turn latency", g(B, "turn", "wall_s") <= TURN_CEIL * g(BL, "turn", "wall_s"),
                  f"{g(B, 'turn', 'wall_s')} vs {g(BL, 'turn', 'wall_s')} s")
    elif not BL:
        V.append("- (no baseline: speed not judged)")
else:
    V.append("- MISSING bench"); incomplete = True

verdict = "INCOMPLETE" if incomplete and ok else ("ACCEPT" if ok else "REJECT")
L += ["", f"## Verdict: **{verdict}**", ""] + V
open(os.path.join(a.run, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
sys.exit(0 if verdict == "ACCEPT" else 2 if verdict == "INCOMPLETE" else 1)
