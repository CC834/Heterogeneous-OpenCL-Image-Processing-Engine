#!/bin/bash
# Dynamic optimization script for split_image_blur
# - For each batch size, it dynamically searches for the best GPU ratio
# - Uses a coarse-to-fine grid search around the best candidate
# - Can enforce an imbalance constraint

set -eu

PROGRAM="./split_image_blur"
RUNS_PER_CONFIG=3
TARGET_IMBALANCE=5.0

# Ratios are fraction of rows to GPU (0..1)
RATIO_MIN=0.05
RATIO_MAX=0.95

# Batch sizes to test
BATCH_SIZES=(50 100 250 500 650 1000)

RESULTS_FILE="optimization_results_dynamic.txt"

# Large number to represent "infinity" for comparisons
INF_VALUE=999999

# ---- deps ----
command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }
command -v awk >/dev/null 2>&1 || { echo "ERROR: awk not found"; exit 1; }
[ -x "$PROGRAM" ] || { echo "ERROR: $PROGRAM not found or not executable"; exit 1; }

echo "==========================================" | tee "$RESULTS_FILE"
echo "Dynamic Split-Image Optimization Script" | tee -a "$RESULTS_FILE"
echo "Batch sizes: ${BATCH_SIZES[*]}" | tee -a "$RESULTS_FILE"
echo "Runs per measurement: $RUNS_PER_CONFIG" | tee -a "$RESULTS_FILE"
echo "Imbalance constraint: <= ${TARGET_IMBALANCE}%" | tee -a "$RESULTS_FILE"
echo "Ratio clamp: [$RATIO_MIN, $RATIO_MAX]" | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

clamp_ratio() {
  echo "$1" | awk -v mn="$RATIO_MIN" -v mx="$RATIO_MAX" '
    {r=$1; if(r<mn) r=mn; if(r>mx) r=mx; printf("%.3f\n", r)}'
}

# Format a number with bc to ensure leading zero and fixed precision
format_ratio() {
  echo "$1" | awk '{printf "%.3f", $1}'
}

extract_total_time_ms() {
  echo "$1" | awk '
    /Total wall-clock time:/ {
      for(i=1;i<=NF;i++){
        if ($i ~ /^[0-9]+(\.[0-9]+)?$/ && $(i+1)=="ms") { print $i; exit; }
      }
    }'
}

extract_imbalance() {
  echo "$1" | awk -F': ' '
    /Workload imbalance:/ {
      gsub(/%/,"",$2);
      if ($2 ~ /^[0-9.]+$/) print $2;
    }' | tail -n 1
}

# Measures avg wall-time and avg imbalance for a given ratio+batch
# Prints: "<avg_time_ms> <avg_imbalance>"
# Returns INF_VALUE for both if measurement fails
measure_config() {
  local ratio="$1"
  local batch="$2"

  local sum_time="0"
  local sum_imb="0"
  local valid=0

  # warm-up (throw away)
  "$PROGRAM" "$ratio" "$batch" >/dev/null 2>&1 || true

  for run in $(seq 1 "$RUNS_PER_CONFIG"); do
    local out
    out=$("$PROGRAM" "$ratio" "$batch" 2>&1) || true

    local t
    local imb
    t=$(extract_total_time_ms "$out")
    imb=$(extract_imbalance "$out")

    if [ -n "${t:-}" ] && [ -n "${imb:-}" ]; then
      valid=$((valid+1))
      sum_time=$(echo "$sum_time + $t" | bc)
      sum_imb=$(echo "$sum_imb + $imb" | bc)
    fi
  done

  if [ "$valid" -eq 0 ]; then
    echo "$INF_VALUE $INF_VALUE"
    return
  fi

  local avg_t
  local avg_i
  avg_t=$(echo "scale=2; $sum_time / $valid" | bc)
  avg_i=$(echo "scale=2; $sum_imb / $valid" | bc)
  echo "$avg_t $avg_i"
}

# Chooses the best ratio among a list of ratios for this batch.
# Preference: lowest avg time subject to imbalance constraint.
choose_best_from_list() {
  local batch="$1"; shift
  local ratios=("$@")

  local best_t="$INF_VALUE"
  local best_i="$INF_VALUE"
  local best_r=""

  local best_t_relaxed="$INF_VALUE"
  local best_i_relaxed="$INF_VALUE"
  local best_r_relaxed=""

  for r in "${ratios[@]}"; do
    r=$(clamp_ratio "$r")

    read -r avg_t avg_i <<< "$(measure_config "$r" "$batch")"

    # log each probe
    if [ "$avg_t" = "$INF_VALUE" ]; then
      echo "    probe ratio=$r -> FAILED" | tee -a "$RESULTS_FILE"
      continue
    fi
    echo "    probe ratio=$r -> avg_time=${avg_t}ms, avg_imb=${avg_i}%" | tee -a "$RESULTS_FILE"

    # Track best without constraint (fallback)
    if [ -z "$best_r_relaxed" ] || [ "$(echo "$avg_t < $best_t_relaxed" | bc -l)" -eq 1 ]; then
      best_t_relaxed="$avg_t"
      best_i_relaxed="$avg_i"
      best_r_relaxed="$r"
    fi

    # Track best that meets imbalance constraint
    local ok
    ok=$(echo "$avg_i <= $TARGET_IMBALANCE" | bc -l)
    if [ "$ok" -eq 1 ]; then
      if [ -z "$best_r" ] || [ "$(echo "$avg_t < $best_t" | bc -l)" -eq 1 ]; then
        best_t="$avg_t"
        best_i="$avg_i"
        best_r="$r"
      fi
    fi
  done

  if [ -n "$best_r" ]; then
    echo "$best_r $best_t $best_i"
  elif [ -n "$best_r_relaxed" ]; then
    # none met constraint; return relaxed best
    echo "$best_r_relaxed $best_t_relaxed $best_i_relaxed"
  else
    # all failed
    echo "0 $INF_VALUE $INF_VALUE"
  fi
}

best_time="$INF_VALUE"
best_batch="0"
best_ratio="0"
best_imb="$INF_VALUE"

for batch in "${BATCH_SIZES[@]}"; do
  echo "============================================" | tee -a "$RESULTS_FILE"
  echo "BATCH SIZE: $batch" | tee -a "$RESULTS_FILE"
  echo "============================================" | tee -a "$RESULTS_FILE"

  # ---- Stage 1: coarse search across the whole range ----
  coarse_ratios=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95)

  echo "  Stage 1 (coarse search)..." | tee -a "$RESULTS_FILE"
  read -r r1 t1 i1 <<< "$(choose_best_from_list "$batch" "${coarse_ratios[@]}")"
  echo "  -> best coarse ratio=$r1 time=${t1}ms imb=${i1}%" | tee -a "$RESULTS_FILE"
  echo "" | tee -a "$RESULTS_FILE"

  # Skip further stages if coarse search failed entirely
  if [ "$t1" = "$INF_VALUE" ]; then
    echo "  Skipping fine/micro search - coarse search failed" | tee -a "$RESULTS_FILE"
    continue
  fi

  # ---- Stage 2: fine search around best coarse ratio ----
  # window +/- 0.08 step 0.02
  echo "  Stage 2 (fine search around $r1)..." | tee -a "$RESULTS_FILE"

  fine_ratios=()
  for delta in -0.08 -0.06 -0.04 -0.02 0 0.02 0.04 0.06 0.08; do
    fine_ratios+=( "$(format_ratio "$(echo "$r1 + $delta" | bc -l)")" )
  done

  read -r r2 t2 i2 <<< "$(choose_best_from_list "$batch" "${fine_ratios[@]}")"
  echo "  -> best fine ratio=$r2 time=${t2}ms imb=${i2}%" | tee -a "$RESULTS_FILE"
  echo "" | tee -a "$RESULTS_FILE"

  # Skip micro search if fine search failed
  if [ "$t2" = "$INF_VALUE" ]; then
    echo "  Skipping micro search - fine search failed" | tee -a "$RESULTS_FILE"
    # Use coarse result
    r3="$r1"; t3="$t1"; i3="$i1"
  else
    # ---- Stage 3: micro search around best fine ratio ----
    # window +/- 0.02 step 0.005
    echo "  Stage 3 (micro search around $r2)..." | tee -a "$RESULTS_FILE"

    micro_ratios=()
    for delta in -0.020 -0.015 -0.010 -0.005 0 0.005 0.010 0.015 0.020; do
      micro_ratios+=( "$(format_ratio "$(echo "$r2 + $delta" | bc -l)")" )
    done

    read -r r3 t3 i3 <<< "$(choose_best_from_list "$batch" "${micro_ratios[@]}")"
    echo "  -> best micro ratio=$r3 time=${t3}ms imb=${i3}%" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    # Fall back to fine result if micro failed
    if [ "$t3" = "$INF_VALUE" ]; then
      r3="$r2"; t3="$t2"; i3="$i2"
    fi
  fi

  echo "  FINAL for batch=$batch: ratio=$r3 avg_time=${t3}ms avg_imb=${i3}%" | tee -a "$RESULTS_FILE"
  echo "" | tee -a "$RESULTS_FILE"

  # Track global best by time
  if [ "$t3" != "$INF_VALUE" ] && [ "$(echo "$t3 < $best_time" | bc -l)" -eq 1 ]; then
    best_time="$t3"
    best_batch="$batch"
    best_ratio="$r3"
    best_imb="$i3"
  fi
done

echo "==========================================" | tee -a "$RESULTS_FILE"
echo "DYNAMIC OPTIMIZATION COMPLETE" | tee -a "$RESULTS_FILE"
echo "==========================================" | tee -a "$RESULTS_FILE"

if [ "$best_time" = "$INF_VALUE" ]; then
  echo "ERROR: No valid configuration found!" | tee -a "$RESULTS_FILE"
  exit 1
fi

echo "BEST CONFIGURATION:" | tee -a "$RESULTS_FILE"
echo "  Batch size: $best_batch" | tee -a "$RESULTS_FILE"
echo "  GPU ratio: $best_ratio" | tee -a "$RESULTS_FILE"
echo "  Avg total time: ${best_time}ms" | tee -a "$RESULTS_FILE"
echo "  Avg imbalance: ${best_imb}%" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Run with: $PROGRAM $best_ratio $best_batch" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
