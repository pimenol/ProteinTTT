#!/bin/bash
# Copy to PROJECT storage everything in bfvd2/ that the 2026-09-01 archive
# (/mnt/proj1/open-35-8/pimenol1/bfvd2, migrate_bfvd2_proj1.sh) does not hold.
# Scratch purges files unaccessed for 60-90 days; proj1 has daily .snapshots.
#
# Packed (zstd chunks, same format and checks as the 09-01 archive):
#   msa_delta     msa/ entries absent from packed/msa/lists, or changed since
#   complement    whole tree
#   benchmark     whole tree (09-01 raw/benchmark predates esmfold2_*)
#   esmfold2_dev  whole tree
# Copied verbatim to raw/: the top-level files of bfvd2/.
#
# Not copied: bfvd2/bfvd2/ and old/ -- unchanged since 09-01 apart from the
# move of the former top-level files into old/.
#
# Runs on the login node (no SLURM), niced. Resumable like the 09-01 script.

set -uo pipefail

SRC=/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/bfvd2
ARCH=/mnt/proj1/open-35-8/pimenol1/bfvd2
DST=$ARCH/delta_20261022
CUTOFF='2026-08-31 00:00'   # start of the 09-01 migration job
JOBS=8           # concurrent chunks
THREADS=2        # zstd threads per chunk  (8 x 2 = 16 of 64 login cores)
PER_CHUNK=20000  # members per chunk

mkdir -p "$DST/packed" "$DST/raw"

finish_plan() {
    local lists="$1"
    for l in "$lists"/chunk_*; do wc -l < "$l" > "$l.count"; done
    touch "$lists/.done"
    echo "[plan/$(basename "$(dirname "$lists")")] $(cat "$lists"/chunk_*.count | awk '{s+=$1} END{print s}') entries"
}

plan_tree() {
    local name="$1" root="$2"
    local lists="$DST/packed/$name/lists"
    [ -f "$lists/.done" ] && { echo "[plan/$name] skip"; return 0; }
    mkdir -p "$lists"; rm -f "$lists"/chunk_*
    find "$root" -mindepth 1 -printf '%P\n' | sort \
        | split -l "$PER_CHUNK" -d -a 4 - "$lists/chunk_"
    finish_plan "$lists"
}

plan_msa_delta() {
    local lists="$DST/packed/msa_delta/lists"
    [ -f "$lists/.done" ] && { echo "[plan/msa_delta] skip"; return 0; }
    mkdir -p "$lists"; rm -f "$lists"/chunk_*
    # absent from the 09-01 lists, plus anything changed after the cutoff
    sort -u \
        <(comm -23 <(find "$SRC/msa" -mindepth 1 -printf '%P\n' | sort) \
                   <(cat "$ARCH"/packed/msa/lists/chunk_[0-9][0-9][0-9][0-9] | sort)) \
        <(find "$SRC/msa" -mindepth 1 -newerct "$CUTOFF" -printf '%P\n') \
        | split -l "$PER_CHUNK" -d -a 4 - "$lists/chunk_"
    finish_plan "$lists"
}

# Same as migrate_bfvd2_proj1.sh: promoted only after zstd -t and a member count.
do_chunk() {
    local list="$1" root="$2" out_dir="$3"
    local idx=${list##*chunk_}
    local out="$out_dir/chunk_${idx}.tar.zst"

    [ -s "$out" ] && [ -s "$out.sha256" ] && { echo "[$idx] skip"; return 0; }

    local t0=$SECONDS
    if ! tar -cf - --no-recursion -C "$root" -T "$list" \
         | zstd -9 --long=27 -T"$THREADS" -q -c > "$out.part"; then
        echo "[$idx] ERROR tar|zstd" >&2; rm -f "$out.part"; return 1
    fi
    if ! zstd -t "$out.part" 2>/dev/null; then
        echo "[$idx] ERROR zstd integrity" >&2; rm -f "$out.part"; return 1
    fi
    local got want
    got=$(zstd -dc "$out.part" 2>/dev/null | tar -tf - | wc -l)
    want=$(cat "$list.count")
    if [ "$got" -ne "$want" ]; then
        echo "[$idx] ERROR members $got != $want" >&2; rm -f "$out.part"; return 1
    fi

    mv "$out.part" "$out"
    sha256sum "$out" | awk '{print $1}' > "$out.sha256"
    echo "[$idx] OK $(( $(stat -c%s "$out") / 1048576 )) MB members=$got $((SECONDS-t0))s"
}
export -f do_chunk
export THREADS

pack_tree() {
    local name="$1" root="$2"
    local lists="$DST/packed/$name/lists" chunks="$DST/packed/$name/chunks"
    mkdir -p "$chunks"
    echo "=== packing $name -> $chunks"
    ls "$lists"/chunk_[0-9]* | grep -v '\.count$' \
        | xargs -P "$JOBS" -I{} bash -c 'do_chunk "$1" "$2" "$3"' _ {} "$root" "$chunks"
    echo "=== $name: $(ls "$chunks"/*.tar.zst 2>/dev/null | wc -l) / $(ls "$lists"/chunk_[0-9]* | grep -vc '\.count$') chunks"
}

echo "start: $(date '+%F %T')  host=$(hostname)"

plan_msa_delta
plan_tree complement   "$SRC/complement"
plan_tree benchmark    "$SRC/benchmark"
plan_tree esmfold2_dev "$SRC/esmfold2_dev"

pack_tree msa_delta    "$SRC/msa"
pack_tree complement   "$SRC/complement"
pack_tree benchmark    "$SRC/benchmark"
pack_tree esmfold2_dev "$SRC/esmfold2_dev"

echo "=== rsync top-level files"
find "$SRC" -mindepth 1 -maxdepth 1 -type f -printf '%P\n' \
    | rsync -a --info=stats2 --files-from=- "$SRC"/ "$DST/raw"/

cp "$0" "$DST/"
echo "=== final"
du -sh "$DST" 2>/dev/null
find "$DST" | wc -l
echo "done: $(date '+%F %T')"
