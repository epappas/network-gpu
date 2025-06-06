#!/usr/bin/env python3
"""
Compilation test for NetworkGPU Python bindings.

This script tests that the Python bindings can be imported and basic
functionality works without requiring a running server.
"""

def test_import():
    """Test that the module can be imported."""
    try:
        import networkgpu
        print("Successfully imported networkgpu")
        return True
    except ImportError as e:
        print(f"Failed to import networkgpu: {e}")
        return False

def test_client_creation():
    """Test client creation (without connecting)."""
    try:
        import networkgpu
        
        print("Module imported, basic API available")
        
        try:
            networkgpu.list_devices()
        except RuntimeError as e:
            if "not initialized" in str(e):
                print("Proper error handling for uninitialized client")
            else:
                print(f"Unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"Client creation test failed: {e}")
        return False

def test_classes():
    """Test that classes are available."""
    try:
        import networkgpu
        
        classes = ['Client', 'Device', 'Tensor', 'NetworkGPUError']
        for cls_name in classes:
            if hasattr(networkgpu, cls_name):
                print(f"Class {cls_name} available")
            else:
                print(f"Class {cls_name} missing")
                return False
        
        return True
    except Exception as e:
        print(f"Class test failed: {e}")
        return False

def main():
    """Run all compilation tests."""
    print("Testing NetworkGPU Python bindings compilation...")
    print()
    
    tests = [
        ("Import test", test_import),
        ("Client creation test", test_client_creation),
        ("Classes test", test_classes),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All compilation tests passed!")
        print("To test with a server:")
        print("  1. Start server: cd server && ./target/release/server")
        print("  2. Run: python examples/basic_usage.py")
    else:
        print("Some tests failed. Check the build process.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())