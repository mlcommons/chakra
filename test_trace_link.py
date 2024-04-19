import os
import unittest

from param_bench.train.compute.python.tools.trace_link import (
    approximate_match,
    exact_match,
    trace_analysis,
)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTraceLink(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(CURR_DIR, "data")

    def test_exact_match(self):
        et_file = os.path.join(self.config_path, "linear_et.json.gz")
        kineto_file = os.path.join(self.config_path, "linear_kineto.json.gz")
        # Annotation to slice multiple iterations in kineto trace
        annotation = "Optimizer.step#SGD.step"

        et_nodes, kineto_et_events = trace_analysis(et_file, kineto_file, annotation)
        et_enhanced = exact_match(kineto_et_events, et_nodes)
        self.assertTrue(et_enhanced)

    def test_approximate_match(self):
        et_file = os.path.join(self.config_path, "resnet_et.json.gz")
        kineto_file = os.path.join(self.config_path, "resnet_kineto.json.gz")
        # Annotation to slice multiple iterations in kineto trace
        annotation = "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__"

        et_nodes, kineto_et_events = trace_analysis(et_file, kineto_file, annotation)
        et_enhanced = approximate_match(kineto_et_events, et_nodes)
        self.assertTrue(et_enhanced)


if __name__ == "__main__":
    unittest.main()
