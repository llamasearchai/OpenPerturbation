#!/usr/bin/env python3
"""Test script to verify the fixes work correctly."""

import sys
from pathlib import Path

# Add the demo_endpoints module to path
sys.path.insert(0, str(Path(__file__).parent))

from demo_endpoints import safe_get_dict_value, safe_len

def test_safe_functions():
    """Test the safe helper functions."""
    print("Testing safe helper functions...")
    
    # Test safe_get_dict_value
    test_dict = {"key": "value", "list": [1, 2, 3]}
    assert safe_get_dict_value(test_dict, "key") == "value"
    assert safe_get_dict_value(test_dict, "missing", "default") == "default"
    assert safe_get_dict_value("not_a_dict", "key", "default") == "default"
    assert safe_get_dict_value(None, "key", "default") == "default"
    assert safe_get_dict_value(123, "key", "default") == "default"
    
    # Test safe_len
    assert safe_len([1, 2, 3]) == 3
    assert safe_len("hello") == 5
    assert safe_len({1, 2, 3}) == 3
    assert safe_len(123) == 0  # int has no __len__
    assert safe_len(None) == 0
    
    print("✓ All safe function tests passed!")

if __name__ == "__main__":
    test_safe_functions()
    print("All tests completed successfully!")