#!/bin/bash

# Script to run all tests 100 times to verify stability
# Usage: ./run_tests_100x.sh

set -e

# Configuration
TOTAL_RUNS=100
LOG_DIR="test_runs_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$LOG_DIR/test_summary.txt"

# Set up CuVS library path
export LD_LIBRARY_PATH=/home/searchscale/code/cuvs-api/cuvs/cpp/build:$LD_LIBRARY_PATH

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize counters
PASSED_RUNS=0
FAILED_RUNS=0
TOTAL_TESTS=0
TOTAL_FAILURES=0
TOTAL_ERRORS=0
TOTAL_SKIPPED=0

echo "Starting $TOTAL_RUNS test runs..."
echo "Results will be logged to: $LOG_DIR"
echo "======================================"

# Create summary header
cat > "$SUMMARY_FILE" << EOF
Test Run Summary - $(date)
====================================
Total Runs: $TOTAL_RUNS

Run Details:
EOF

for i in $(seq 1 $TOTAL_RUNS); do
    echo "Running test iteration $i/$TOTAL_RUNS..."
    
    RUN_LOG="$LOG_DIR/run_${i}.log"
    
    # Run tests and capture output
    if mvn test > "$RUN_LOG" 2>&1; then
        PASSED_RUNS=$((PASSED_RUNS + 1))
        STATUS="PASSED"
        
        # Extract test statistics from this run
        TESTS=$(grep "Tests run:" "$RUN_LOG" | tail -1 | sed 's/.*Tests run: \([0-9]*\).*/\1/')
        FAILURES=$(grep "Tests run:" "$RUN_LOG" | tail -1 | sed 's/.*Failures: \([0-9]*\).*/\1/')
        ERRORS=$(grep "Tests run:" "$RUN_LOG" | tail -1 | sed 's/.*Errors: \([0-9]*\).*/\1/')
        SKIPPED=$(grep "Tests run:" "$RUN_LOG" | tail -1 | sed 's/.*Skipped: \([0-9]*\).*/\1/')
        
        # Update totals
        TOTAL_TESTS=$((TOTAL_TESTS + TESTS))
        TOTAL_FAILURES=$((TOTAL_FAILURES + FAILURES))
        TOTAL_ERRORS=$((TOTAL_ERRORS + ERRORS))
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))
        
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        STATUS="FAILED"
        
        # Show failure details
        echo "  FAILURE in run $i!"
        echo "  Check $RUN_LOG for details"
        
        # Extract error summary
        echo "  Error summary:"
        grep -A 5 -B 5 "FAILURE\|ERROR" "$RUN_LOG" | head -20 | sed 's/^/    /'
    fi
    
    # Add to summary
    echo "Run $i: $STATUS" >> "$SUMMARY_FILE"
    
    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/$TOTAL_RUNS runs ($(( (i * 100) / TOTAL_RUNS ))%)"
        echo "  Current stats: $PASSED_RUNS passed, $FAILED_RUNS failed"
    fi
done

# Calculate final statistics
SUCCESS_RATE=$(( (PASSED_RUNS * 100) / TOTAL_RUNS ))
AVG_TESTS_PER_RUN=$(( TOTAL_TESTS / TOTAL_RUNS ))
AVG_FAILURES_PER_RUN=$(( TOTAL_FAILURES / TOTAL_RUNS ))
AVG_ERRORS_PER_RUN=$(( TOTAL_ERRORS / TOTAL_RUNS ))
AVG_SKIPPED_PER_RUN=$(( TOTAL_SKIPPED / TOTAL_RUNS ))

# Write final summary
cat >> "$SUMMARY_FILE" << EOF

====================================
FINAL RESULTS:
====================================
Total Runs:           $TOTAL_RUNS
Passed Runs:          $PASSED_RUNS
Failed Runs:          $FAILED_RUNS
Success Rate:         $SUCCESS_RATE%

Test Statistics:
- Total Tests Run:    $TOTAL_TESTS
- Total Failures:     $TOTAL_FAILURES  
- Total Errors:       $TOTAL_ERRORS
- Total Skipped:      $TOTAL_SKIPPED

Average per run:
- Tests:              $AVG_TESTS_PER_RUN
- Failures:           $AVG_FAILURES_PER_RUN
- Errors:             $AVG_ERRORS_PER_RUN
- Skipped:            $AVG_SKIPPED_PER_RUN

Log Directory: $LOG_DIR
EOF

# Display final results
echo ""
echo "======================================"
echo "TEST RUNS COMPLETED"
echo "======================================"
echo "Total Runs:     $TOTAL_RUNS"
echo "Passed:         $PASSED_RUNS"
echo "Failed:         $FAILED_RUNS"
echo "Success Rate:   $SUCCESS_RATE%"
echo ""
echo "Average Tests per Run: $AVG_TESTS_PER_RUN"
echo "Average Failures:      $AVG_FAILURES_PER_RUN"
echo "Average Errors:        $AVG_ERRORS_PER_RUN"
echo "Average Skipped:       $AVG_SKIPPED_PER_RUN"
echo ""
echo "Detailed results saved to: $SUMMARY_FILE"
echo "Individual run logs in:    $LOG_DIR/"

if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: Some test runs failed!"
    echo "Check individual run logs for failure details."
    exit 1
else
    echo ""
    echo "✅ All test runs passed successfully!"
    exit 0
fi