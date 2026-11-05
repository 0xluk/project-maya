# Two-box split for GLM-5.3-Flash

This branch lets one GLM-5.3-Flash model run across **two machines** over an ordinary network. Each machine runs one
Maya process on its own GPUs; the model is split by layers at a boundary layer `K`:

- **front**: the embedding and layers `[0, K)`, the OpenAI-compatible server, the tokenizer;
- **back**: layers `[K, n_layers)`, the NextN (MTP) block, the output head and the sampler.

Only the residual of each token crosses the network (hc 4 × 4096 values: 64 KB in fp32, ~24 KB as packed bf16), so
1 GbE is enough. Inside each machine the usual multi-GPU layer split still applies (`--layer-split K:b1,b2,..`).

Without `--role` nothing here runs and the engine is unchanged.

## What it was built for

Two older desktop boxes with 4× RTX 3060 12 GB each (PCIe 3.0, no P2P, 1 GbE between them) can't hold Maya-S24's experts
in one box, but together they hold ~98% of them in VRAM. Measured with Maya-S24, `--prefill 2048`, 1 conversation:

| | decode (short) | decode at ~30K context | prompt 2K | prompt 30K | agent turn (~2.4K new tokens) |
|---|---|---|---|---|---|
| 2 boxes × 4× RTX 3060, 20 + 32 GB RAM | 38–42 tok/s | 33–37 tok/s | 300–350 tok/s | ~1,100–1,170 tok/s | 6–7 s |
| for reference: 8× RTX 3060 in one box | 42–45 tok/s | – | 380–410 tok/s | ~1,050 tok/s | 5.0–5.3 s |

A coding-agent benchmark (5 tasks, 48 hidden tests) scored 48/48, the same as the single-box runs.

## Usage

Same Maya-S24 pack on both machines (each process only reads the tensors of its own layers). Same `--max-context`,
`--prefill` and `MAYA_GLM_PREFILL_MIN_CHUNKS` on both sides (they must match).

Back (start it first; it waits for the front):

```sh
MAYA_GLM_LINK_ALLOW=<front-ip> MAYA_GLM_LINK_KEY_FILE=/path/to/link.key \
MAYA_GLM_PREFILL_MIN_CHUNKS=4,512 \
  build/maya --serve --glm-pack <pack> --max-context 65536 --expert-cache auto --prefill 2048 --kv-streaming on \
  --layer-split 24:30,36,41 --role back --link <back-ip>:7710
```

Front (the server, as in a normal Maya config; add to `"args"` and `"env"`):

```json
"args": [ "...", "--layer-split", "24:7,13,19", "--role", "front", "--link", "<back-ip>:7710" ],
"env":  { "MAYA_GLM_LINK_KEY_FILE": "/path/to/link.key", "MAYA_GLM_PREFILL_MIN_CHUNKS": "4,512" }
```

`--layer-split K:b1,b2,..`: `K` is the boundary between the machines; `b1,b2,..` split each side's layers across its
local GPUs (the front uses the ones below `K`, the back the ones above). The back's last GPU also carries the head and
the NextN block, so give it fewer layers.

## Link settings

| Setting | Default | Meaning |
|---|---|---|
| `--role front\|back`, `--link host:port` | off | enable the split; the front connects, the back listens |
| `MAYA_GLM_LINK_ALLOW` | loopback only | comma-separated IPv4 addresses the back accepts a front from |
| `MAYA_GLM_LINK_KEY_FILE` | unset | shared key (≥ 32 bytes) on both sides: an HMAC-SHA256 challenge-response runs both ways on every connection; a missing or short file stops the engine before it loads |
| `STRATA_GLM_LINK_BIND` | the `--link` host | address the back listens on (an empty host is loopback, never every interface) |
| `STRATA_GLM_LINK_BF16` | 1 | the residual crosses as bf16 (round to nearest even) |
| `STRATA_GLM_LINK_PACK` | 1 with bf16 | lossless packing of the bf16 rows (~12 bits a value) |
| `STRATA_GLM_RELINK_WAIT` | 5 s | how long the front waits to reconnect after a link failure |

Requests are bracketed (begin / end / ack), so a cancelled or failed request leaves both sides in step; the back stays
up and accepts the front again. Speculative decoding (MTP) is pipelined across the link.

Also set an API key on the front's server (`STRATA_API_KEY`) if the port is reachable from your network.

## Several conversations at once (`--parallel`)

`--parallel N` works over the link with the split's batched rows. Give **both** sides the same `--parallel N`,
`MAYA_GLM_SPLIT_ROWS` and `MAYA_GLM_SEQ_CTX` (the back: on its command line / environment; the front: in `"args"` and
`"env"`):

```json
"args": [ "...", "--parallel", "4" ],
"env":  { "MAYA_GLM_SPLIT_ROWS": "2", "MAYA_GLM_EVEN_GROUPS": "1", "MAYA_GLM_SEQ_CTX": "32768" }
```

Both sides run the several-conversation scheduler in lock step. The front sends its own decisions to the back (take
a new request now or decode a step; for every token: go on, stop, cancel, pause, forced token), and the back follows
them. Each decode row crosses the link on its own (one hop per row, one token back). The front queues what it sends
behind rows that are not ready yet and keeps going; the back sends each token as soon as it is picked. A group that
gets its token back is issued again while the other group is still on the front, so both machines stay busy.

Measured on the same 2 × 4× RTX 3060 rigs (greedy 400-token answers, short prompts):

| | tok/s per request | tok/s in all |
|---|---|---|
| 1 request (any setting: it takes the speculative path) | 40–43 | 40–43 |
| 2 at once, `--parallel 2`, `SPLIT_ROWS=2` (one group of 2 rows) | 16–17 | 33 |
| 2 at once, `--parallel 2`, `SPLIT_ROWS=1` (two groups, pipelined) | 24 | 48 |
| 2 at once, `--parallel 4`, `SPLIT_ROWS=2`, `EVEN_GROUPS=1` | 25–26 | 50–52 |
| 4 at once, `--parallel 4`, `SPLIT_ROWS=2`, `EVEN_GROUPS=1` | 14.5 | 58 |

Two agent turns at once (~2K new tokens each on ~9–10K cached tokens, 128-token answers): 14.7 and 16.1 s, against
8.6–9.3 s for one turn alone. The prompts are still read one at a time. Cancelling one of two running requests ends
only that one; the other finishes and the link stays up.

Limits:
- Needs `MAYA_GLM_SPLIT_ROWS` ≥ 1 and at least 2 GPUs on each side. Without them the engine stops at startup.
- `MAYA_GLM_MULTI_MTP`, `MAYA_GLM_PREFILL_INTERLEAVE` and `MAYA_GLM_PREFILL_GAPS` are turned off over the link.
- A STOP during a prompt's read takes effect at the request's first token.
- A failure on either side ends every running request on both sides. The link drops and the front reconnects at
  its next request.
- Both sides must run this build when `--parallel` is above 1. With `--parallel 1` (the default) nothing changes.
- The gain comes from keeping both machines busy. A single request already decodes speculatively (the NextN
  drafts), so two conversations give ×1.2 and four ×1.45 in all over one request. That is below Maya's ×1.6–2.5 on one
  box.
- Greedy answers of concurrent requests match a request alone until the first near-tie between two tokens. Production
  runs of the same prompt split at the same places, because expert placement changes the order sums are done in.

## Correctness

Teacher-forced logits (`STRATA_GLM_KL_DUMP` with `maya generate --tokens-file`) against a single-process run on one
machine: KL mean ~1.2–1.6e-2, top-1 ~95%. That is inside the spread a single process shows when only the expert
placement changes (RAM tier size, CPU lane on/off, prompt chunk size: KL 1.1–2.5e-2, top-1 93.8–98.4%): near-ties in
the top-8 expert choice flip on rounding. A split whose expert tiers match a single process slot for slot stays within 2e-3 of it with the bf16 hop.

## Notes

- Host RAM matters more than the network: with 16 GB per box part of the experts came from SATA SSD on every prompt
  chunk and turns took ~31 s; with 20 + 32 GB nothing is read from disk.
- Bigger prompt chunks (`--prefill auto`) need more pinned host RAM on both sides.
- Tested on Linux with CUDA 13.2, sm_86, on Project Maya v1.1.50.
- RAM with `--parallel 4` at 32K for the extra conversations: the front box (20 GB) kept ≥ 3.9 GB available.
