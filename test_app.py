#!/usr/bin/env python3
"""
Test script to run the PLSTM-TAL application with mock dependencies
"""

# Import mocks first
import mocks

# Now try to import the application
try:
    print("Testing basic imports...")
    from config.settings import STOCK_INDICES, PLSTM_CONFIG
    print("✓ Config imported successfully")
    
    print("Testing application import...")
    import app
    print("✓ Application imported successfully")
    
    print("\nApplication is ready to run!")
    print(f"Stock indices configured: {list(STOCK_INDICES.keys())}")
    print(f"PLSTM config loaded: {PLSTM_CONFIG['sequence_length']} sequence length")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()