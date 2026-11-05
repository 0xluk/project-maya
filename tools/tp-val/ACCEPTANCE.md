# Decode-only in-box TP (`MAYA_GLM_TP_DECODE=1`): acceptance criteria

Each candidate is checked with `run_ab.sh` (one A/B leg) and `report.py`, which applies these rules and exits 0 (ACCEPT),
1 (REJECT) or 2 (incomplete). TP changes the order of the sums, so the logits are not expected to be bit-identical to
twin.2. The bounds below come from the placement-noise study of the KL gate (Phase 5).

## 1. Correctness (hard gate): KL CLI, input s2, 128 positions, production settings on both boxes
| check | reference | pass if |
|---|---|---|
| KL vs ref2b | the single-process full-tier reference dump `ref2b/s2.r1` | KL mean ≤ **2.5e-2**, top-1 agreement ≥ **93%**, p99 ≤ **0.45**, and all logits finite |
| KL vs twin.2 | the production twin.2 dump (sha256 `70f47cf8…b0a5870`) | KL mean ≤ **1e-2** |

- twin.2 reproduces its own dump bit for bit when host memory is the same. When host RAM differs (for example, a build
  running on the front), the disk landing ring gets smaller and KL vs twin.2 is about 2e-4. That is still far below the bound.
- A real layout or reduction bug, such as a wrong shard, a missing all-reduce or a stale buffer, shows KL ≥ 0.1 and
  top-1 well below 90%.

## 2. Greedy texts (hard gate, plus a human read)
- Three fixed prompts are decoded greedily (temperature 0) to 400 tokens each.
- Automatic checks: at least 300 characters, no U+FFFD, ≥ 98% printable characters, no 8-word shingle repeated more
  than 4 times, and a word-type ratio ≥ 0.25.
- Read the texts (`bench/text-short*.txt`). They must be coherent and on topic.
- Production itself can change its greedy text from run to run at near-tie positions. So "identical to baseline" is
  only informational, and so is the first character that differs.

## 3. Speed (the point of TP, judged against the twin.2 baseline run `runs/baseline-twin2`)
| metric | pass if |
|---|---|
| mean decode tok/s over the 3 short prompts (400 tokens, MTP on) | ≥ 0.97 × baseline |
| decode tok/s after a ~25K-token prompt (300 tokens) | ≥ 0.97 × baseline |
| turn wall: ~2K new tokens on ~10K cached context, 200 tokens out | ≤ 1.10 × baseline |
Prompt reading (prefill) is not changed by TP. The long-prompt and turn prompt rates are shown only to catch side effects.
To enable TP, the decode gain should be clearly above the ±3% run-to-run noise.

## Baseline (twin.2 production binaries, no TP env; 2026-11-05)
| metric | value |
|---|---|
| short decode B-tree / intervals / French Rev. | 38.4 / 41.4 / 40.0 tok/s (mean 39.9); MTP acceptance 75 / 87 / 79% |
| ~25K context (25,031 tokens) | prompt 1,044 tok/s; decode 36.3 tok/s |
| turn: 2,018 new on 10,021 cached | 11.8 s wall (prompt 5.6 s at 357 tok/s; decode 34.6 tok/s) |
| KL vs ref2b | 2.25e-2 / p99 0.317 / top-1 96.1% |
| KL vs twin.2 dump | 1.97e-4 / top-1 100% (front landing ring 62 vs 67 experts: less host RAM than in the Phase 10 run) |

**Headroom warning:** the production twin.2 dump itself is **2.33e-2** vs ref2b (p99 0.361, top-1 96.1%), and this baseline
run is 2.25e-2. Under the 2.5e-2 bound there is only about 7-10% headroom. TP's own reordering noise (expected to be about
1e-4 to 1e-3 vs twin.2 when no expert route flips) does not threaten it. One extra near-tie route flip could, though. If a
TP build fails only the ref2b mean while KL vs twin.2 is ≤ 1e-2 and top-1/p99 pass, re-run once. A clean re-run counts.
