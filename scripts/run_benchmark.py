#!/usr/bin/env python3
"""
Run SceneCraft benchmark (PhysScene).

This script provides a bridge to the existing benchmark code.
"""
import sys
from pathlib import Path

# Add paths for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmark"))

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the benchmark."""
    logger.info("SceneCraft Benchmark (PhysScene)")
    logger.info("For full benchmark functionality, run scripts in the benchmark/ directory")
    
    # Check if benchmark directory exists
    benchmark_dir = repo_root / "benchmark"
    if not benchmark_dir.exists():
        logger.error("Benchmark directory not found")
        return 1
    
    # List available benchmark scripts
    scripts = list(benchmark_dir.glob("**/*.py"))
    if scripts:
        logger.info("Available benchmark scripts:")
        for script in sorted(scripts):
            rel_path = script.relative_to(repo_root)
            logger.info(f"  python {rel_path}")
    else:
        logger.warning("No benchmark scripts found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())