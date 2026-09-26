#!/bin/bash
# Archive each per-protein log dir into <protein>.tar (in place), then delete
# the original dir. Safe (verify-before-delete), resumable, inode-aware.
set -u

LOGS="/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/bfvd2/helix_analysis/bfvd2_new/logs"
PROGRESS="/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/archive_logs_progress.log"

# Skip dirs modified in the last 5 min, so an active run's current dir is left alone.
RECENT_MIN=20

# Be a good citizen on a shared node.
NICE="nice -n 19"
command -v ionice >/dev/null 2>&1 && NICE="ionice -c3 $NICE"

archived=0; skipped=0; failed=0
: > "$PROGRESS"
echo "$(date '+%F %T') START  logs=$LOGS" >> "$PROGRESS"

while IFS= read -r -d '' d; do
    name="$(basename "$d")"
    final="$LOGS/$name.tar"
    tmp="$LOGS/.$name.tar.tmp"

    # Resume: already archived.
    if [ -e "$final" ]; then
        skipped=$((skipped+1)); continue
    fi

    # Create archive to a temp name; only promote it after verification, so an
    # interrupted/partial tar can never be mistaken for a complete one.
    if $NICE tar -cf "$tmp" -C "$LOGS" "$name" 2>>"$PROGRESS" \
       && tar -tf "$tmp" >/dev/null 2>&1; then
        mv -f "$tmp" "$final"
        rm -rf -- "$d"            # delete original ONLY after verified archive
        archived=$((archived+1))
    else
        rm -f -- "$tmp" 2>/dev/null
        echo "$(date '+%F %T') FAIL  $name" >> "$PROGRESS"
        failed=$((failed+1))
    fi

    if (( (archived+skipped) % 2000 == 0 )); then
        echo "$(date '+%F %T') ...  archived=$archived skipped=$skipped failed=$failed" >> "$PROGRESS"
    fi
done < <(find "$LOGS" -mindepth 1 -maxdepth 1 -type d -mmin +$RECENT_MIN -print0)

echo "$(date '+%F %T') DONE  archived=$archived skipped=$skipped failed=$failed" >> "$PROGRESS"
