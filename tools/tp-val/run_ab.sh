#!/bin/bash
# run_ab.sh LABEL [options]: one A/B leg of the two-box split, driven from a third machine over ssh.
#
#   stop production (front unit, then back unit; kill -9 any back left; wait until the GPUs are free)
#   -> serve the split with the given binaries + env, from COPIES of production's launch files
#   -> bench_client.py: warm-up, 3 greedy 400-token decodes, ~25K-context decode, ~2K-new-token turn on ~10K cached
#   -> stop it -> KL CLI dump (KL_INPUT, production settings) with the same binaries + env -> stop
#   -> klcmp vs the ref2b and twin.2 dumps -> report.py (acceptance)  -> [--restore] production units + health 200
#
# Options:
#   --front-bin PATH   binary on the front (default FRONT_BIN_PROD)     --back-bin PATH  (default BACK_BIN_PROD)
#   --env "K=V ..."    env for both sides (e.g. "MAYA_GLM_TP_DECODE=1")  --front-env / --back-env  one side only
#   --tests LIST       bench tests (default warm,short,long,turn)        --no-bench / --no-kl  skip a phase
#   --restore          start production again at the end and check health 200 + unchanged binaries/configs
#   --prepare-only     write the remote run copies (no GPU, production untouched) and exit, to inspect them
#   --baseline DIR     run dir to compare speed against in the report (default RUNS_DIR/baseline-twin2)
# Box layout comes from rigs.env next to this script (or RIGS_ENV=file); see rigs.env.example.
# Results: RUNS_DIR/LABEL/ (driver.log, bench/, kl.json, kl.txt, summary.md); remote copies under *_RUN_ROOT/LABEL.
# Every GPU phase holds GPU_LOCK (flock) on both boxes for the whole run.  A CLAUDE_STOP file in STOP_FLAG_DIR
# aborts at the next step (test processes are stopped, locks released, production restored if --restore).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RIGS=${RIGS_ENV:-$HERE/rigs.env}
[ -f "$RIGS" ] || { echo "no $RIGS (copy rigs.env.example)"; exit 2; }
. "$RIGS"
[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
LABEL=$1; shift
FRONT_BIN=$FRONT_BIN_PROD; BACK_BIN=$BACK_BIN_PROD; ENV_BOTH=""; FRONT_ENV=""; BACK_ENV=""
RESTORE=0; PREP_ONLY=0; DO_BENCH=1; DO_KL=1; TESTS=warm,short,long,turn
RUNS_DIR=${RUNS_DIR:-$HERE/runs}; BASELINE=$RUNS_DIR/baseline-twin2
while [ $# -gt 0 ]; do
  case $1 in
    --front-bin) FRONT_BIN=$2; shift ;;
    --back-bin) BACK_BIN=$2; shift ;;
    --env) ENV_BOTH=$2; shift ;;
    --front-env) FRONT_ENV=$2; shift ;;
    --back-env) BACK_ENV=$2; shift ;;
    --tests) TESTS=$2; shift ;;
    --no-bench) DO_BENCH=0 ;;
    --no-kl) DO_KL=0 ;;
    --restore) RESTORE=1 ;;
    --prepare-only) PREP_ONLY=1 ;;
    --baseline) BASELINE=$2; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
FENV=$(echo "$ENV_BOTH $FRONT_ENV" | xargs); BENV=$(echo "$ENV_BOTH $BACK_ENV" | xargs)
OUT=$RUNS_DIR/$LABEL; mkdir -p "$OUT/bench"
FR=$FRONT_RUN_ROOT/$LABEL; BR=$BACK_RUN_ROOT/$LABEL
SSHO="-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
fs() { ssh $SSHO "$FRONT_SSH" "$@"; }
bs() { ssh $SSHO "$BACK_SSH" "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
STATE=start; RESTORED=0; FAILED=0; PROD_STOPPED=0

check_stop() {
  if [ -n "${STOP_FLAG_DIR:-}" ] && [ -e "$STOP_FLAG_DIR/CLAUDE_STOP" ]; then
    log "STOP flag present: aborting"; FAILED=1; exit 3
  fi
}
die() { log "FAIL: $*"; FAILED=1; exit 1; }

# ---------------------------------------------------------------- GPU lock: one holder process per box
take_lock() {  # $1 = fs|bs
  $1 "rm -f $GPU_LOCK.tpval; setsid nohup flock -w 1200 $GPU_LOCK -c 'echo \$\$ > $GPU_LOCK.tpval; exec sleep 14400' \
      >/dev/null 2>&1 </dev/null &"
  local i
  for i in $(seq 1 240); do
    $1 "test -s $GPU_LOCK.tpval && kill -0 \$(cat $GPU_LOCK.tpval) 2>/dev/null" && return 0
    sleep 5
  done
  die "could not take $GPU_LOCK via $1 within 20 min"
}
drop_lock() { $1 "test -s $GPU_LOCK.tpval && kill \$(cat $GPU_LOCK.tpval) 2>/dev/null; rm -f $GPU_LOCK.tpval" || true; }

gpu_max_used() { $1 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1" 2>/dev/null; }
wait_gpu_free() {  # $1 = fs|bs
  local i u
  for i in $(seq 1 60); do
    u=$(gpu_max_used $1); [ -n "$u" ] && [ "$u" -lt 200 ] && return 0
    sleep 5
  done
  die "GPUs not free via $1 (max used ${u:-?} MiB)"
}

# ---------------------------------------------------------------- production fingerprint (to prove restore is exact)
fingerprint() {
  fs "sha256sum $FRONT_BIN_PROD $FRONT_CFG $FRONT_MAYA_SH $FRONT_KL_SH | awk '{print \$1, \$2}'; systemctl cat $FRONT_UNIT | sha256sum | sed 's/-\$/unit/'"
  bs "sha256sum $BACK_BIN_PROD $BACK_RUN_SH $BACK_KL_SH | awk '{print \$1, \$2}'; systemctl cat $BACK_UNIT | sha256sum | sed 's/-\$/unit/'"
}

stop_production() {
  log "stopping production: $FRONT_UNIT, then $BACK_UNIT"; PROD_STOPPED=1
  fs "sudo systemctl stop $FRONT_UNIT" || die "stop $FRONT_UNIT"
  bs "sudo systemctl stop $BACK_UNIT" || die "stop $BACK_UNIT"
  bs "pkill -9 -f -- '/maya .*--role [b]ack' || true"
  wait_gpu_free fs; wait_gpu_free bs
  log "production stopped, GPUs free"
}

# ---------------------------------------------------------------- test processes (each in its own session/pgid)
kill_group() {  # $1 = fs|bs, $2 = pid file, $3 = 1 for SIGTERM first
  $1 "p=\$(cat $2 2>/dev/null) || exit 0; [ -n \"\$p\" ] || exit 0
      if [ $3 = 1 ]; then kill -TERM -- -\$p 2>/dev/null; for i in \$(seq 1 40); do pgrep -g \$p >/dev/null || break; sleep 1; done; fi
      kill -9 -- -\$p 2>/dev/null; rm -f $2; true"
}
stop_test() {
  kill_group fs "$FR/front.pid" 1
  kill_group bs "$BR/back.pid" 0
  bs "pkill -9 -f -- '/maya .*--role [b]ack' || true"
  fs "pkill -9 -f -- '/maya .*--role [f]ront' || true"
  wait_gpu_free fs; wait_gpu_free bs
}
alive() { $1 "p=\$(cat $2 2>/dev/null) && pgrep -g \$p >/dev/null"; }

prepare_remote() {
  # back: a copy of the unit's env + its ExecStart script, with the binary swapped in
  bs "set -e; mkdir -p $BR; cd $BR
      systemctl cat $BACK_UNIT | sed -n 's/^Environment=/export /p' > unit.env
      cp $BACK_RUN_SH run-back.copy.sh; sed -i 's#build/maya #\"\$BIN\" #g' run-back.copy.sh
      grep -q '\"\$BIN\" --serve' run-back.copy.sh
      cp $BACK_KL_SH kl-back.copy.sh
      sed -i 's#STRATA_GLM_KL_DUMP=[^,]*,#STRATA_GLM_KL_DUMP=\$DUMP,#' kl-back.copy.sh; grep -q 'KL_DUMP=\$DUMP,' kl-back.copy.sh
      printf '%s\n' '#!/bin/bash' 'cd \$(dirname \$0); . ./unit.env' ${BENV:+'export $BENV'} 'export BIN=$BACK_BIN' \
        'exec bash ./run-back.copy.sh serve' > serve.sh
      printf '%s\n' '#!/bin/bash' 'cd \$(dirname \$0)' ${BENV:+'export $BENV'} 'export BIN=$BACK_BIN INPUT=$KL_INPUT DUMP=$BR/kl-$KL_INPUT.f32' \
        'exec bash ./kl-back.copy.sh $LABEL' > kl.sh
      test -x $BACK_BIN" || die "prepare back ($BR)"
  # front: a copy of the server config with exe/log/env changed, the unit's env (incl. EnvironmentFile), the KL script
  fs "set -e; mkdir -p $FR; cd $FR
      python3 - $FRONT_CFG front.json $FRONT_BIN $FR/front.engine.log '$FENV' <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c['exe'] = sys.argv[3]; c['log'] = sys.argv[4]
env = dict(c.get('env') or {})
for kv in sys.argv[5].split():
    k, v = kv.split('=', 1); env[k] = v
c['env'] = env; json.dump(c, open(sys.argv[2], 'w'), indent=1)
PY
      systemctl cat $FRONT_UNIT | sed -n 's/^EnvironmentFile=-\{0,1\}\(.*\)/set -a; . \1; set +a/p; s/^Environment=/export /p' > unit.env
      cp $FRONT_KL_SH kl-front.copy.sh
      printf '%s\n' '#!/bin/bash' 'cd \$(dirname \$0); . ./unit.env' ${FENV:+'export $FENV'} \
        'exec $FRONT_MAYA_SH --config $FR/front.json --plain' > serve.sh
      printf '%s\n' '#!/bin/bash' 'cd \$(dirname \$0)' ${FENV:+'export $FENV'} 'export BIN=$FRONT_BIN INPUT=$KL_INPUT' \
        'exec bash ./kl-front.copy.sh' > kl.sh
      test -x $FRONT_BIN" || die "prepare front ($FR)"
  scp -q $SSHO "$HERE/klcmp.py" "$FRONT_SSH:$FR/klcmp.py" || die "copy klcmp.py"
}

launch() {  # $1 = fs|bs, $2 = dir, $3 = script, $4 = log, $5 = pid file
  $1 "cd $2 || exit 1; setsid nohup bash $3 > $4 2>&1 < /dev/null & echo \$! > $5"
}

health() { curl -s -o /dev/null -m 5 -w '%{http_code}' "$API_URL/health" 2>/dev/null; }
wait_health() {  # $1 = timeout s, $2 = 1 to also watch the test pids
  local t0=$(date +%s) c
  while :; do
    c=$(health); [ "$c" = 200 ] && return 0
    if [ "$2" = 1 ]; then
      alive bs "$BR/back.pid" || die "back died during load (see $BR/serve.log)"
      alive fs "$FR/front.pid" || die "front died during load (see $FR/serve.log)"
    fi
    [ $(( $(date +%s) - t0 )) -gt $1 ] && die "no health 200 within $1 s (last $c)"
    check_stop; sleep 10
  done
}

restore_production() {
  log "restoring production: $BACK_UNIT, then $FRONT_UNIT"
  bs "sudo systemctl start $BACK_UNIT" || { log "start $BACK_UNIT failed"; return 1; }
  fs "sudo systemctl start $FRONT_UNIT" || { log "start $FRONT_UNIT failed"; return 1; }
  local t0=$(date +%s) c
  while :; do
    c=$(health); [ "$c" = 200 ] && break
    [ $(( $(date +%s) - t0 )) -gt 1200 ] && { log "RESTORE: no health 200 within 20 min"; return 1; }
    sleep 10
  done
  fingerprint > "$OUT/prod-fingerprint.after" 2>&1
  if cmp -s "$OUT/prod-fingerprint.before" "$OUT/prod-fingerprint.after"; then
    log "RESTORE OK: health 200, $(fs "systemctl is-active $FRONT_UNIT")/$(bs "systemctl is-active $BACK_UNIT"), binaries+configs+units unchanged"
  else
    log "RESTORE WARNING: production fingerprint changed:"; diff "$OUT/prod-fingerprint.before" "$OUT/prod-fingerprint.after" | tee -a "$OUT/driver.log"
  fi
  python3 "$HERE/bench_client.py" --url "$API_URL" --key-file "$API_KEY_FILE" --corpus "$CORPUS" --out "$OUT/restore" \
    --tests warm,short1 >> "$OUT/driver.log" 2>&1 || true
  RESTORED=1
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$STATE" = serve ] || [ "$STATE" = kl ] || [ "$STATE" = prepared ]; then
    log "cleanup after rc=$rc in state $STATE: stopping test processes"
    stop_test 2>/dev/null || true
  fi
  if [ "$STATE" != start ]; then drop_lock fs; drop_lock bs; fi
  if [ $RESTORE = 1 ] && [ $RESTORED = 0 ] && [ $PROD_STOPPED = 1 ]; then restore_production || log "RESTORE FAILED"; fi
  log "exit rc=$rc"
  exit $rc
}
trap cleanup EXIT
trap 'FAILED=1; exit 130' INT TERM

# ================================================================ main
log "=== $LABEL: front $FRONT_BIN [$FENV] | back $BACK_BIN [$BENV] | bench=$DO_BENCH kl=$DO_KL restore=$RESTORE"
check_stop
CORPUS=$RUNS_DIR/corpus-$CORPUS_COMMIT.txt
if [ ! -s "$CORPUS" ]; then
  fs "cd $CORPUS_REPO && for f in \$(git ls-tree --name-only $CORPUS_COMMIT src/core/ | grep '\\.cu\$' | sort); do git show $CORPUS_COMMIT:\$f; done" \
    > "$CORPUS.tmp" && mv "$CORPUS.tmp" "$CORPUS" || die "corpus"
fi
fingerprint > "$OUT/prod-fingerprint.before" 2>&1 || die "fingerprint"
fs "sha256sum $FRONT_BIN" > "$OUT/test-bins.sha" && bs "sha256sum $BACK_BIN" >> "$OUT/test-bins.sha" || die "test binaries missing"
# the twin.2 KL reference on the front (copied once from the back, sha-checked)
if ! fs "test -s $KL_TWIN2"; then
  fs "mkdir -p \$(dirname $KL_TWIN2)"; scp -q -3 $SSHO "$BACK_SSH:$KL_TWIN2_SRC" "$FRONT_SSH:$KL_TWIN2" || die "copy twin.2 dump"
fi
fs "sha256sum $KL_TWIN2" | grep -q "^$KL_TWIN2_SHA" || die "twin.2 reference dump sha mismatch"

if [ $PREP_ONLY = 1 ]; then prepare_remote; log "prepared $FR and $BR"; exit 0; fi
STATE=locking; log "taking $GPU_LOCK on both boxes"
take_lock fs; take_lock bs
check_stop
STATE=stopping; stop_production
STATE=prepared; prepare_remote
check_stop

if [ $DO_BENCH = 1 ]; then
  STATE=serve; log "serve: starting back, then front"
  launch bs "$BR" serve.sh "$BR/serve.log" "$BR/back.pid"
  sleep 5; launch fs "$FR" serve.sh "$FR/serve.log" "$FR/front.pid"
  t0=$(date +%s); wait_health 1200 1; log "health 200 after $(( $(date +%s) - t0 )) s"
  check_stop
  log "bench: $TESTS"
  python3 "$HERE/bench_client.py" --url "$API_URL" --key-file "$API_KEY_FILE" --corpus "$CORPUS" --out "$OUT/bench" \
    --tests "$TESTS" 2>&1 | tee -a "$OUT/driver.log" | grep -q '^done' || log "bench did not finish (see driver.log)"
  scp -q $SSHO "$FRONT_SSH:$FR/front.engine.log" "$OUT/front.engine.log" 2>/dev/null
  log "serve: stopping"; stop_test
fi
check_stop

if [ $DO_KL = 1 ]; then
  STATE=kl; log "KL CLI ($KL_INPUT): back, then front"
  bs "rm -f $BR/kl-$KL_INPUT.f32"
  # host RAM sets the disk landing ring size, which can move logits slightly (~2e-4 KL): keep it on record
  log "MemAvailable before KL: front $(fs "awk '/MemAvailable/{printf \"%.1f GB\", \$2/1048576}' /proc/meminfo"), back $(bs "awk '/MemAvailable/{printf \"%.1f GB\", \$2/1048576}' /proc/meminfo")"
  launch bs "$BR" kl.sh "$BR/kl.log" "$BR/back.pid"
  sleep 5; launch fs "$FR" kl.sh "$FR/kl.log" "$FR/front.pid"
  t0=$(date +%s)
  while alive bs "$BR/back.pid"; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && die "KL run over 30 min"
    alive fs "$FR/front.pid" || { sleep 30; alive bs "$BR/back.pid" && die "front KL exited, back still running (see $FR/kl.log)"; break; }
    check_stop; sleep 10
  done
  log "KL CLI done in $(( $(date +%s) - t0 )) s"
  stop_test
  bs "sha256sum $BR/kl-$KL_INPUT.f32" | tee "$OUT/kl-dump.sha" || die "no KL dump (see $BR/kl.log)"
  scp -q -3 $SSHO "$BACK_SSH:$BR/kl-$KL_INPUT.f32" "$FRONT_SSH:$FR/kl-$KL_INPUT.f32" || die "copy dump"
  N=$(fs "cat \$(dirname \$(readlink -f $KL_REF2B))/../$KL_INPUT.n")
  fs "cd $FR && $FRONT_PY -I klcmp.py --n $N --json kl-ref2b.json $KL_REF2B kl-$KL_INPUT.f32 && \
      $FRONT_PY -I klcmp.py --n $N --json kl-twin2.json $KL_TWIN2 kl-$KL_INPUT.f32 && rm -f kl-$KL_INPUT.f32" \
    | tee "$OUT/kl.txt" | tee -a "$OUT/driver.log"
  fs "cd $FR && python3 -c 'import json; print(json.dumps({\"ref2b\": json.load(open(\"kl-ref2b.json\"))[0], \"twin2\": json.load(open(\"kl-twin2.json\"))[0]}))'" \
    > "$OUT/kl.json" || die "KL compare"
  scp -q $SSHO "$BACK_SSH:$BR/kl.log" "$OUT/kl-back.log"; scp -q $SSHO "$FRONT_SSH:$FR/kl.log" "$OUT/kl-front.log"
fi

STATE=done
drop_lock fs; drop_lock bs
log "test processes stopped, GPU locks released"
python3 "$HERE/report.py" "$OUT" --baseline "$BASELINE" | tee -a "$OUT/driver.log"
[ $RESTORE = 1 ] && restore_production
log "=== $LABEL finished; summary: $OUT/summary.md"
