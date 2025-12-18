#!/bin/bash

# Optimization script for split_image_blur
# Tests different batch sizes and GPU ratios to find best configuration

PROGRAM="./split_image_blur"
RUNS_PER_CONFIG=3  # Runs per configuration for averaging
TARGET_IMBALANCE=5.0

# Batch sizes to test
BATCH_SIZES=(50 100 250 500 1000)

# Output file for results
RESULTS_FILE="optimization_results.txt"
RAW_OUTPUT_DIR="raw_outputs"

# Check dependencies
command -v bc >/dev/null || { echo "bc not found"; exit 1; }

# Create raw output directory
mkdir -p "$RAW_OUTPUT_DIR"

echo "==========================================" | tee $RESULTS_FILE
echo "Split-Image Optimization Script" | tee -a $RESULTS_FILE
echo "Testing batch sizes: ${BATCH_SIZES[*]}" | tee -a $RESULTS_FILE
echo "Runs per config: ${RUNS_PER_CONFIG}" | tee -a $RESULTS_FILE
echo "Target imbalance: <= ${TARGET_IMBALANCE}%" | tee -a $RESULTS_FILE
echo "==========================================" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Store best result
best_time=999999
best_batch=0
best_ratio=0
best_imbalance=100

# For each batch size, find optimal ratio then measure performance
for batch in "${BATCH_SIZES[@]}"; do
    echo "============================================" | tee -a $RESULTS_FILE
    echo "TESTING BATCH SIZE: $batch" | tee -a $RESULTS_FILE
    echo "============================================" | tee -a $RESULTS_FILE

    # Step 1: Warm-up run (reduces first-run bias)
    echo "  Warm-up run..." | tee -a $RESULTS_FILE
    $PROGRAM 0.5 $batch > /dev/null 2>&1

    # Step 2: Find optimal ratio for this batch size (start at 0.5)
    current_ratio=0.5

    for iter in 1 2 3; do
        echo "  Calibration iteration $iter (ratio: $current_ratio)..." | tee -a $RESULTS_FILE

        # Run once to get recommended ratio
        output=$($PROGRAM $current_ratio $batch 2>&1)

        # Robust parsing: extract number after colon, remove %
        optimal=$(echo "$output" | awk -F': ' '/Recommended GPU ratio:/ {gsub(/%/,"",$2); print $2}')

        if [ -n "$optimal" ]; then
            current_ratio=$(echo "scale=3; $optimal / 100" | bc)
            # Clamp ratio to [0.05, 0.95]
            current_ratio=$(echo "$current_ratio" | awk '{if($1<0.05)print 0.05; else if($1>0.95)print 0.95; else print $1}')
        fi
    done

    echo "  Optimal ratio for batch $batch: $current_ratio" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    # Step 3: Run multiple times with optimal ratio and collect stats
    sum_time=0
    sum_imbalance=0
    valid_runs=0

    for run in $(seq 1 $RUNS_PER_CONFIG); do
        echo -n "  Run $run/$RUNS_PER_CONFIG... " | tee -a $RESULTS_FILE

        output=$($PROGRAM $current_ratio $batch 2>&1)

        # Save raw output
        echo "$output" > "${RAW_OUTPUT_DIR}/batch${batch}_run${run}.txt"

        # Robust parsing
        total_time=$(echo "$output" | awk -F': ' '/Total wall-clock time:/ {print $2}' | awk '{print $1}')
        imbalance=$(echo "$output" | awk -F': ' '/Workload imbalance:/ {gsub(/%/,"",$2); print $2}')
        time_per_image=$(echo "$output" | awk -F': ' '/Combined time per image:/ {print $2}' | awk '{print $1}')

        if [ -n "$total_time" ] && [ -n "$imbalance" ]; then
            echo "time/img=${time_per_image}ms, imbalance=${imbalance}%, total=${total_time}ms" | tee -a $RESULTS_FILE
            sum_time=$(echo "$sum_time + $total_time" | bc)
            sum_imbalance=$(echo "$sum_imbalance + $imbalance" | bc)
            valid_runs=$((valid_runs + 1))
        else
            echo "FAILED to parse" | tee -a $RESULTS_FILE
        fi
    done

    # Calculate averages (only if we have valid runs)
    if [ "$valid_runs" -gt 0 ]; then
        avg_time=$(echo "scale=2; $sum_time / $valid_runs" | bc)
        avg_imbalance=$(echo "scale=1; $sum_imbalance / $valid_runs" | bc)
        avg_time_per_image=$(echo "scale=3; $avg_time / 5000" | bc)

        echo "" | tee -a $RESULTS_FILE
        echo "  BATCH $batch RESULTS:" | tee -a $RESULTS_FILE
        echo "    Optimal ratio: $current_ratio" | tee -a $RESULTS_FILE
        echo "    Avg total time: ${avg_time}ms" | tee -a $RESULTS_FILE
        echo "    Avg time/image: ${avg_time_per_image}ms" | tee -a $RESULTS_FILE
        echo "    Avg imbalance: ${avg_imbalance}%" | tee -a $RESULTS_FILE
        echo "    Valid runs: $valid_runs/$RUNS_PER_CONFIG" | tee -a $RESULTS_FILE
        echo "" | tee -a $RESULTS_FILE

        # Check if this is the best so far (must meet imbalance target)
        imbalance_ok=$(echo "$avg_imbalance <= $TARGET_IMBALANCE" | bc -l)
        is_faster=$(echo "$avg_time < $best_time" | bc -l)

        if [ "$imbalance_ok" -eq 1 ] && [ "$is_faster" -eq 1 ]; then
            best_time=$avg_time
            best_batch=$batch
            best_ratio=$current_ratio
            best_imbalance=$avg_imbalance
            echo "  *** NEW BEST (meets imbalance target) ***" | tee -a $RESULTS_FILE
        elif [ "$is_faster" -eq 1 ]; then
            echo "  (Faster but imbalance > ${TARGET_IMBALANCE}%)" | tee -a $RESULTS_FILE
        fi
    else
        echo "  BATCH $batch: No valid runs!" | tee -a $RESULTS_FILE
    fi
done

# Print summary
echo "" | tee -a $RESULTS_FILE
echo "==========================================" | tee -a $RESULTS_FILE
echo "OPTIMIZATION COMPLETE" | tee -a $RESULTS_FILE
echo "==========================================" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

if [ "$best_batch" -gt 0 ]; then
    echo "BEST CONFIGURATION (imbalance <= ${TARGET_IMBALANCE}%):" | tee -a $RESULTS_FILE
    echo "  Batch size: $best_batch" | tee -a $RESULTS_FILE
    echo "  GPU ratio: $best_ratio" | tee -a $RESULTS_FILE
    echo "  Total time: ${best_time}ms" | tee -a $RESULTS_FILE
    echo "  Imbalance: ${best_imbalance}%" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    echo "Run with: $PROGRAM $best_ratio $best_batch" | tee -a $RESULTS_FILE
else
    echo "No configuration met the imbalance target of ${TARGET_IMBALANCE}%." | tee -a $RESULTS_FILE
    echo "Try adjusting TARGET_IMBALANCE or running more calibration iterations." | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "Results saved to: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "Raw outputs saved to: $RAW_OUTPUT_DIR/" | tee -a $RESULTS_FILE
