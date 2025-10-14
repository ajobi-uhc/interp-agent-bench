#!/bin/bash
# Run curriculum stages for model interpretability training
# Usage:
#   ./run_curriculum.sh              # Run stages sequentially
#   ./run_curriculum.sh --parallel   # Run all stages in parallel

set -e  # Exit on error

CURRICULUM_DIR="configs/curriculum_v1"
LOG_DIR="logs/curriculum_$(date +%Y%m%d_%H%M%S)"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=================================="
echo "🎓 Running Curriculum v1"
echo "=================================="
echo "Log directory: $LOG_DIR"
echo ""

# Define stages
STAGES=(
    "stage_1:Guided Discovery"
    "stage_2:Verification"
    "stage_3:Multiple Choice"
    "stage_4:Pure Discovery"
)

# Function to run a single stage
run_stage() {
    local stage_file=$1
    local stage_name=$2
    local log_file="$LOG_DIR/${stage_file%.yaml}.log"

    echo "[$stage_name] Starting..."
    echo "[$stage_name] Config: $CURRICULUM_DIR/$stage_file"
    echo "[$stage_name] Log: $log_file"

    if python run_agent.py "$CURRICULUM_DIR/$stage_file" > "$log_file" 2>&1; then
        echo "[$stage_name] ✅ Completed"
    else
        echo "[$stage_name] ❌ Failed (exit code: $?)"
        echo "[$stage_name] Check log: $log_file"
        return 1
    fi
}

# Check if parallel mode requested
if [ "$1" = "--parallel" ]; then
    echo "🚀 Running stages in PARALLEL mode"
    echo "⚠️  Warning: This may hit Modal GPU quotas"
    echo ""

    # Start all stages in background
    pids=()
    for stage_info in "${STAGES[@]}"; do
        IFS=':' read -r stage_file stage_name <<< "$stage_info"
        run_stage "$stage_file.yaml" "$stage_name" &
        pids+=($!)
    done

    # Wait for all to complete
    echo "Waiting for all stages to complete..."
    failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            failed=1
        fi
    done

    if [ $failed -eq 0 ]; then
        echo ""
        echo "=================================="
        echo "✅ All stages completed"
        echo "=================================="
    else
        echo ""
        echo "=================================="
        echo "⚠️  Some stages failed"
        echo "=================================="
        exit 1
    fi
else
    echo "🚀 Running stages SEQUENTIALLY"
    echo ""

    # Run stages one by one
    failed=0
    for stage_info in "${STAGES[@]}"; do
        IFS=':' read -r stage_file stage_name <<< "$stage_info"
        echo ""
        if ! run_stage "$stage_file.yaml" "$stage_name"; then
            failed=1
            echo ""
            echo "⚠️  Stopping curriculum due to failure"
            break
        fi
    done

    echo ""
    echo "=================================="
    if [ $failed -eq 0 ]; then
        echo "✅ Curriculum completed"
    else
        echo "❌ Curriculum failed"
    fi
    echo "=================================="

    [ $failed -eq 0 ] || exit 1
fi

echo ""
echo "📊 Results summary:"
echo "Logs: $LOG_DIR"
echo "Notebooks: notebooks/"
