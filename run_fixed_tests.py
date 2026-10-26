#!/usr/bin/env python3
"""Run the specific tests we fixed to verify they pass."""

import sys
import unittest
from tests.test_tracking_and_events import TestEventNarrator, TestPerformanceMonitor

def run_specific_tests():
    """Run the specific tests we fixed."""
    # Create a test suite with our fixed tests
    test_names = [
        'test_global_cooldown_enforcement',
        'test_track_reappearance_after_grace_period_is_new_object',
        'test_should_announce_respects_cooldowns',
        'test_urgency_escalation_event'
    ]

    suite = unittest.TestSuite()

    # Add EventNarrator tests
    for test_name in test_names:
        suite.addTest(TestEventNarrator(test_name))

    # Add PerformanceMonitor test
    suite.addTest(TestPerformanceMonitor('test_get_summary_returns_correct_dictionary'))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)

if __name__ == '__main__':
    success, tests_run, failures, errors = run_specific_tests()

    print(f"\n{'='*50}")
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)