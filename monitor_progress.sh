#!/bin/bash

# Script to monitor AlphaEdit experiment progress

RESULTS_DIR="/home/stud/golab/AlphaEdit/results/AlphaEdit"
EXPECTED_TOTAL=2000

echo "AlphaEdit Experiment Progress Monitor"
echo "====================================="

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Find all run directories
runs=$(ls -1 "$RESULTS_DIR" | grep "^run_" | sort -V)

if [ -z "$runs" ]; then
    echo "No run directories found."
    exit 0
fi

echo "Found run directories:"
for run in $runs; do
    echo "  - $run"
done

echo ""
echo "Progress for each run:"
echo "======================"

for run in $runs; do
    run_dir="$RESULTS_DIR/$run"
    
    # Count completed cases
    completed=$(find "$run_dir" -name "*_edits-case_*.json" | wc -l)
    percentage=$(echo "scale=1; $completed * 100 / $EXPECTED_TOTAL" | bc -l 2>/dev/null || echo "N/A")
    
    echo "Run: $run"
    echo "  Completed: $completed / $EXPECTED_TOTAL ($percentage%)"
    
    # Check latest file timestamp
    latest_file=$(find "$run_dir" -name "*_edits-case_*.json" -exec stat -c "%Y %n" {} \; 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$latest_file" ]; then
        latest_time=$(stat -c "%y" "$latest_file" 2>/dev/null)
        echo "  Latest file: $(basename "$latest_file")"
        echo "  Last updated: $latest_time"
    fi
    
    # Check for GLUE evaluation files
    glue_files=$(find "$run_dir/glue_eval" -name "*.json" 2>/dev/null | wc -l)
    if [ "$glue_files" -gt 0 ]; then
        echo "  GLUE evaluations: $glue_files files"
    fi
    
    echo ""
done

# Show recent activity
echo "Recent Activity:"
echo "================"
find "$RESULTS_DIR" -name "*_edits-case_*.json" -newermt "1 hour ago" 2>/dev/null | wc -l | xargs echo "Files created in last hour:"
find "$RESULTS_DIR" -name "*_edits-case_*.json" -newermt "10 minutes ago" 2>/dev/null | wc -l | xargs echo "Files created in last 10 minutes:"

# Check if any jobs are running
echo ""
echo "Active Jobs:"
echo "============"
squeue -u $(whoami) --format="%.10i %.20j %.8T %.10M %.6D %N" 2>/dev/null || echo "Unable to check job queue (squeue not available)"

echo ""
echo "Disk Usage:"
echo "==========="
du -sh "$RESULTS_DIR" 2>/dev/null || echo "Unable to check disk usage"

# Estimate completion time if there's progress
latest_run=$(echo "$runs" | tail -1)
if [ -n "$latest_run" ]; then
    run_dir="$RESULTS_DIR/$latest_run"
    completed=$(find "$run_dir" -name "*_edits-case_*.json" | wc -l)
    
    if [ "$completed" -gt 0 ]; then
        # Find first and last file times
        first_file_time=$(find "$run_dir" -name "*_edits-case_*.json" -exec stat -c "%Y" {} \; 2>/dev/null | sort -n | head -1)
        last_file_time=$(find "$run_dir" -name "*_edits-case_*.json" -exec stat -c "%Y" {} \; 2>/dev/null | sort -n | tail -1)
        
        if [ -n "$first_file_time" ] && [ -n "$last_file_time" ] && [ "$completed" -gt 1 ]; then
            elapsed=$((last_file_time - first_file_time))
            if [ "$elapsed" -gt 0 ]; then
                rate=$(echo "scale=6; $completed / $elapsed" | bc -l 2>/dev/null)
                remaining=$((EXPECTED_TOTAL - completed))
                if [ -n "$rate" ] && [ "$rate" != "0" ]; then
                    eta_seconds=$(echo "scale=0; $remaining / $rate" | bc -l 2>/dev/null)
                    if [ -n "$eta_seconds" ] && [ "$eta_seconds" -gt 0 ]; then
                        eta_hours=$((eta_seconds / 3600))
                        eta_minutes=$(( (eta_seconds % 3600) / 60 ))
                        
                        echo ""
                        echo "Progress Estimation:"
                        echo "==================="
                        echo "Processing rate: $(printf '%.4f' "$rate") cases/second"
                        echo "Estimated time to completion: ${eta_hours}h ${eta_minutes}m"
                        echo "Estimated completion time: $(date -d "+${eta_seconds} seconds" 2>/dev/null || echo 'N/A')"
                    fi
                fi
            fi
        fi
    fi
fi
