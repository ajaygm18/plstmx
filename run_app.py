#!/usr/bin/env python3
"""
Streamlit runner for PLSTM-TAL application with mock dependencies
"""

# Import mocks first
import mocks

# Set up basic streamlit session state
import streamlit as st

# Mock streamlit's session state
if not hasattr(st, 'session_state'):
    class SessionState:
        def __init__(self):
            self._state = {}
        
        def __getattr__(self, key):
            return self._state.get(key, None)
        
        def __setattr__(self, key, value):
            if key.startswith('_'):
                super().__setattr__(key, value)
            else:
                self._state[key] = value
        
        def __contains__(self, key):
            return key in self._state
        
        def get(self, key, default=None):
            return self._state.get(key, default)
    
    st.session_state = SessionState()

# Now import and run the application
try:
    print("Starting PLSTM-TAL Stock Market Prediction App...")
    print("=" * 60)
    
    import app
    
    # Simulate running the main function
    app.main()
    
    print("=" * 60)
    print("Application completed successfully!")
    print("\nNote: This is running with mock dependencies.")
    print("To run with real dependencies, install: streamlit pandas numpy tensorflow scikit-learn plotly yfinance")
    print("Then run: streamlit run app.py")
    
except Exception as e:
    print(f"Error running application: {e}")
    import traceback
    traceback.print_exc()