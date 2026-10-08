#!/usr/bin/env bash
# Stage the DLRMv4 dataset from shared storage onto node-local disk.
# Skips files that are already the right size. Writes ${DST}/.stage_complete on success.
set -uo pipefail

SRC=${DLRM_STAGE_SRC:-${DATADIR:?set DATADIR or DLRM_STAGE_SRC to the shared dataset}}
DST=${DLRM_LOCAL_DATADIR:-/data/mlperf_dlrmv4}
STREAMS=${DLRM_STAGE_STREAMS:-8}

if [[ ! -d "${SRC}/processed_5b" ]]; then
    echo "[stage] $(hostname -s): missing ${SRC}/processed_5b" >&2
    exit 1
fi

echo "[stage] $(hostname -s): ${SRC} -> ${DST} (${STREAMS} streams)"
mkdir -p "${DST}" || { echo "[stage] cannot create ${DST}" >&2; exit 1; }

python3 - "${SRC}" "${DST}" "${STREAMS}" <<'PY'
import os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

src, dst, streams = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs, unreadable = [], []
for root, dirs, files in os.walk(src):
    dirs[:] = sorted(d for d in dirs if not d.startswith("."))
    rel = os.path.relpath(root, src)
    out_dir = dst if rel == "." else os.path.join(dst, rel)
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(files):
        if name.startswith(".") or name.endswith(".part"):
            continue
        sp = os.path.join(root, name)
        if not os.path.isfile(sp):
            continue
        if not os.access(sp, os.R_OK):
            unreadable.append(sp)
            continue
        dp = os.path.join(out_dir, name)
        if os.path.exists(dp) and os.path.getsize(dp) == os.path.getsize(sp):
            continue
        jobs.append((sp, dp, os.path.getsize(sp)))

total = sum(j[2] for j in jobs)
print(
    f"[stage] {len(jobs)} files to copy, {total/1e9:.1f} GB; unreadable skipped: {len(unreadable)}",
    flush=True,
)

copied, failures, started, marked = [0], [], time.time(), [0]

def copy_one(job):
    sp, dp, size = job
    try:
        # Copy to .part then rename so an interrupted stage never leaves a
        # short file that the size check would later accept as complete.
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        shutil.copyfile(sp, dp + ".part")
        os.replace(dp + ".part", dp)
    except Exception as exc:
        failures.append((sp, repr(exc)))
        return
    copied[0] += size
    if copied[0] - marked[0] > (20 << 30):
        marked[0] = copied[0]
        rate = copied[0] / (time.time() - started)
        print(
            f"[stage]   {copied[0]/1e9:.1f}/{total/1e9:.1f} GB  {rate/1e6:.0f} MB/s  "
            f"eta {(total-copied[0])/max(rate,1)/60:.0f} min",
            flush=True,
        )

if jobs:
    with ThreadPoolExecutor(streams) as pool:
        list(pool.map(copy_one, jobs))

elapsed = time.time() - started
print(
    f"[stage] copied {copied[0]/1e9:.1f} GB in {elapsed/60:.1f} min "
    f"= {copied[0]/1e6/max(elapsed,1):.0f} MB/s; failures={len(failures)}",
    flush=True,
)
for path, err in failures[:10]:
    print(f"[stage]   FAILED {path}: {err}", flush=True)
if failures:
    sys.exit(1)

with open(os.path.join(dst, ".stage_complete"), "w") as fh:
    fh.write(f"source={src}\n")
    nfiles = 0
    for root, dirs, files in os.walk(dst):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        nfiles += sum(1 for name in files if not name.startswith(".") and name != ".stage_complete")
    fh.write(f"files={nfiles}\n")
PY
_rc=$?

if [[ ${_rc} -ne 0 ]]; then
    echo "[stage] $(hostname -s): FAILED - local copy left unmarked" >&2
    exit ${_rc}
fi

echo "[stage] $(hostname -s): complete"
cat "${DST}/.stage_complete"
