# tests/run_all_tests.py

import unittest

# Discover and run all test_*.py files in the tests/ folder
loader = unittest.TestLoader()
suite = loader.discover(start_dir=".", pattern="test_*.py")

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
