import pytest
import os

def main():
    """Run the test suite"""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest with verbose output
    pytest.main(['-v', os.path.join(current_dir, 'test_app.py')])

if __name__ == '__main__':
    main()
