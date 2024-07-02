#include <gtest/gtest.h>
#include <iostream>
#include "et_feeder.h"

class ETFeederTest : public ::testing::Test {
protected:
  ETFeederTest() {}
  virtual ~ETFeederTest() {}

  void SetUp(const std::string& filename) {
    trace = new Chakra::ETFeeder(filename);
  }

  virtual void TearDown() {
    delete trace;
  }

  Chakra::ETFeeder* trace;
};

TEST_F(ETFeederTest, IterateAndPrintNodes) {
  SetUp("tests/data/chakra.0.et");

  while (trace->hasNodesToIssue()) {
    std::shared_ptr<Chakra::ETFeederNode> node = trace->getNextIssuableNode();
    if (node != nullptr) {
      std::cout << "Node ID: " << node->id() << ", Node Name: " << node->name() << ", is_cpu_op: " << node->is_cpu_op() << std::endl;

      // Resolve dependencies
      trace->freeChildrenNodes(node->id());

      // Remove the node
      trace->removeNode(node->id());
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

