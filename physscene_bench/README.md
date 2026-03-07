# PhysScene-Bench: 3D Indoor Scene Layout Evaluation System

PhysScene-Bench is a comprehensive benchmark for evaluating Large Language Models' ability to generate realistic 3D indoor scene layouts. It uses a three-layer evaluation system to assess semantic fidelity, physical plausibility, and functional affordance.

## System Overview

### Architecture
```
PhysScene-Bench/
├── prompts/
│   └── layout_prompts.json      # 10 diverse room layout prompts
├── scripts/
│   ├── bench_runner.py          # Main evaluation runner
│   ├── evaluator.py            # Three-layer evaluation engine
│   ├── generate_report.py      # Report and visualization generator
│   └── test_evaluator.py       # Testing utilities
└── results/
    ├── {model_name}/           # Model-specific results
    │   └── {prompt_id}/
    │       ├── generated_layout.json
    │       └── evaluation_results.json
    ├── benchmark_summary.json  # Consolidated results
    └── evaluation_report.md    # Final report with visualizations
```

## Evaluation Framework

### Layer 1: Semantic Fidelity
- **Category Match Score**: Precision, recall, and F1 for expected furniture categories
- **Count Accuracy**: How well object counts match expectations

### Layer 2: Physical Plausibility  
- **Collision Score**: Percentage of object pairs without spatial overlap
- **Boundary Score**: Percentage of objects within room boundaries
- **Stability Score**: Physics simulation-based stability assessment using MuJoCo

### Layer 3: Functional Affordance
- **Facing Score**: Evaluation of directional relationships (sofa facing TV, etc.)
- **Proximity Score**: Assessment of semantically related object distances
- **Walkability Score**: Ratio of navigable space to total room area
- **Wall Alignment Score**: Large furniture placement against walls

## Installation & Setup

### Prerequisites
```bash
pip install mujoco numpy matplotlib seaborn pandas requests scipy
```

### Environment Variables
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export MUJOCO_GL="osmesa"  # For headless physics simulation
```

## Usage

### Basic Evaluation
```bash
cd scripts/

# Test with 1 prompt and 1 model
python3 bench_runner.py --test

# Full evaluation with all prompts and models
python3 bench_runner.py

# Generate comprehensive report
python3 generate_report.py
```

### Custom Configuration
```bash
# Use custom prompts file
python3 bench_runner.py --prompts /path/to/your/prompts.json

# Save results to custom directory  
python3 bench_runner.py --results /path/to/results/

# Generate report from custom results
python3 generate_report.py --results /path/to/results/
```

## Supported Models

The system evaluates these free OpenRouter models:
- `meta-llama/llama-3.3-70b-instruct:free`
- `qwen/qwen3-next-80b-a3b-instruct:free`
- `google/gemma-3-27b-it:free`
- `mistralai/mistral-small-3.1-24b-instruct:free`
- `nousresearch/hermes-3-llama-3.1-405b:free`
- `z-ai/glm-4.5-air:free`

## Output Format

### Generated Scene Layout
```json
{
  "objects": [
    {
      "category": "sofa",
      "position": [2.0, 1.0],
      "rotation": 0,
      "dimensions": {"width": 2.0, "depth": 0.9, "height": 0.85}
    }
  ]
}
```

### Evaluation Results
```json
{
  "overall_score": 0.723,
  "layer_scores": {
    "semantic_fidelity": 0.815,
    "physical_plausibility": 0.667,
    "functional_affordance": 0.706
  },
  "detailed_scores": {
    "semantic": {...},
    "physical": {...},
    "functional": {...}
  }
}
```

## Testing

### Unit Testing
```bash
# Test evaluator with mock data
python3 test_evaluator.py
```

### Integration Testing
```bash
# Test full pipeline without API calls
python3 bench_runner.py --test
```

## Key Features

### Robust JSON Parsing
- Handles markdown code blocks from LLM responses
- Graceful error recovery for malformed JSON
- Rate limiting and exponential backoff

### Physics Simulation
- MuJoCo-based stability assessment
- Automatic XML scene generation
- Headless rendering for server deployment

### Comprehensive Reporting
- Model comparison tables
- Radar charts for multi-dimensional performance
- Score heatmaps across prompts and models
- Layout visualizations for top-scoring examples

### Mock Mode
- Works without API keys for testing
- Rule-based layout generation
- Full evaluation pipeline validation

## Prompt Examples

The system includes 10 diverse room types:
- Living rooms with TV-sofa arrangements
- Bedrooms with bed-nightstand relationships  
- Kitchen dining areas with table-chair groups
- Home offices with desk-chair setups
- Classrooms with student-teacher layouts
- Conference rooms with meeting configurations
- Studio apartments with multi-zone planning
- Reading rooms with armchair-bookshelf pairs
- Kids' rooms with play-study areas
- Lounges with multiple conversation groups

## Performance Metrics

### Scoring Scale
All scores range from 0.0 to 1.0, where:
- **0.9-1.0**: Excellent spatial understanding
- **0.7-0.9**: Good layout quality with minor issues
- **0.5-0.7**: Adequate but with notable problems
- **0.3-0.5**: Poor spatial reasoning
- **0.0-0.3**: Major failures in layout generation

### Benchmark Statistics
- **Total Evaluations**: 60 runs (10 prompts × 6 models)
- **Average Runtime**: ~2 minutes per evaluation
- **Success Rate Target**: >90% for practical deployment
- **Minimum Acceptable Score**: 0.6 overall

## Troubleshooting

### Common Issues
1. **MuJoCo GL Error**: Ensure `MUJOCO_GL=osmesa` is set
2. **API Rate Limits**: Built-in 2-second delays between requests
3. **JSON Parse Errors**: Automatic fallback to mock generation
4. **Memory Issues**: Use `--test` flag for large-scale debugging

### Debug Mode
```bash
# Enable verbose logging
python3 bench_runner.py --test --verbose

# Test individual components
python3 test_evaluator.py
```

## Contributing

### Adding New Prompts
1. Add entries to `prompts/layout_prompts.json`
2. Include `expected_objects` and `functional_checks`
3. Test with `--test` flag first

### Adding New Models
1. Update `MODELS` list in `bench_runner.py`
2. Ensure model supports JSON output format
3. Test API compatibility

### Extending Evaluation Metrics
1. Add new scoring functions to `evaluator.py`
2. Update `evaluate_layout()` to include new metrics
3. Modify report generation for new visualizations

## Citation

If you use PhysScene-Bench in your research, please cite:

```bibtex
@software{physscene_bench_2024,
  title={PhysScene-Bench: A Comprehensive Benchmark for 3D Indoor Scene Layout Generation},
  author={OpenClaw Development Team},
  year={2024},
  url={https://github.com/www-Ye/SceneCraft-LLM}
}
```