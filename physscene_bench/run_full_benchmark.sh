#!/bin/bash
# PhysScene-Bench Full Evaluation Script
# This script runs the complete benchmark evaluation and generates reports.

set -e

echo "🚀 Starting PhysScene-Bench Full Evaluation"
echo "============================================="

# Check environment
echo "📋 Checking environment..."

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  WARNING: OPENROUTER_API_KEY not set."
    echo "   To use real LLMs, get a free API key from https://openrouter.ai/"
    echo "   Then run: export OPENROUTER_API_KEY='your_key_here'"
    echo "   For now, using mock mode for demonstration."
    echo ""
fi

# Set MuJoCo to headless mode
export MUJOCO_GL=osmesa

# Navigate to scripts directory
cd "$(dirname "$0")/scripts"

echo "🧪 Step 1: Running test evaluation (1 prompt, 1 model)"
echo "-------------------------------------------------------"
python3 bench_runner.py --test
echo "✅ Test completed successfully!"
echo ""

echo "🔬 Step 2: Running full benchmark (10 prompts × 6 models)"
echo "--------------------------------------------------------"
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "🔑 API key found - using real LLM evaluations"
    echo "⏱️  This will take approximately 20-30 minutes..."
    python3 bench_runner.py
else
    echo "🤖 No API key - using mock evaluations for demo"
    echo "⏱️  This will complete in ~2 minutes..."
    # Run with all models but using mock responses
    python3 bench_runner.py
fi
echo "✅ Benchmark evaluation completed!"
echo ""

echo "📊 Step 3: Generating comprehensive report"
echo "-----------------------------------------"
python3 generate_report.py
echo "✅ Report generation completed!"
echo ""

echo "📁 Step 4: Results summary"
echo "-------------------------"
echo "Results saved in: ../results/"
echo ""
echo "Key files generated:"
echo "  📄 evaluation_report.md - Main results report"
echo "  🎯 model_performance_radar.png - Model comparison chart"
echo "  🔥 score_heatmap.png - Score distribution heatmap"
echo "  🏠 top_layouts.png - Best layout visualizations"
echo "  📊 benchmark_summary.json - Raw data summary"
echo ""

echo "🎉 PhysScene-Bench evaluation complete!"
echo ""
echo "Next steps:"
echo "  1. View the report: cat ../results/evaluation_report.md"
echo "  2. Check visualizations in ../results/*.png"
echo "  3. Analyze detailed results in ../results/{model_name}/"
echo ""

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "💡 To run with real LLM evaluations:"
    echo "   1. Get free API key: https://openrouter.ai/"
    echo "   2. Export OPENROUTER_API_KEY='your_key'"
    echo "   3. Re-run this script"
    echo ""
fi

echo "🔗 Ready to push results to GitHub:"
echo "   cd .. && git add -A && git commit -m 'feat: PhysScene-Bench results'"