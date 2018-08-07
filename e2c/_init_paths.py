import os, sys

# Set path to root directory so that local imports work
add_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if add_path not in sys.path:
    sys.path.insert(0, add_path)