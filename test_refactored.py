#!/usr/bin/env python3
"""
Quick test of the refactored SceneCraft package.
"""
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test main package import
        import scenecraft
        logger.info(f"✅ scenecraft package imported (version: {scenecraft.__version__})")
        
        # Test asset management
        from scenecraft.assets import AssetManager, TabletopAssetDownloader
        logger.info("✅ Asset management modules imported")
        
        # Test scene generation
        from scenecraft.generation import TabletopGenerator
        logger.info("✅ Scene generation modules imported")
        
        # Test simulation
        from scenecraft.simulation import PhysicsEngine, Renderer
        logger.info("✅ Simulation modules imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_asset_manager():
    """Test AssetManager functionality."""
    try:
        from scenecraft.assets import AssetManager
        
        manager = AssetManager()
        logger.info(f"✅ AssetManager initialized")
        
        # Test catalog loading (might be empty if no assets downloaded)
        categories = manager.get_available_categories()
        logger.info(f"Available categories: {len(categories)} ({categories[:5]}...)")
        
        # Test material properties
        props = manager.get_material_properties("mug")
        logger.info(f"✅ Material properties loaded: {props['rgba']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AssetManager test failed: {e}")
        return False

def test_scene_config():
    """Test scene configuration and XML building.""" 
    try:
        from scenecraft.generation.tabletop import breakfast_scene
        
        scene = breakfast_scene()
        logger.info(f"✅ Scene config loaded: {scene['name']} with {len(scene['objects'])} objects")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scene config test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Testing refactored SceneCraft package...")
    
    tests = [
        ("Import test", test_imports),
        ("AssetManager test", test_asset_manager), 
        ("Scene config test", test_scene_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        logger.info(f"\n--- {name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test failed: {name}")
    
    logger.info(f"\n🏁 Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the code.")
        return 1

if __name__ == "__main__":
    sys.exit(main())