#include <gtest/gtest.h>
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

TEST_F(ETFeederTest, ConstructorNodeIDTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->getNextIssuableNode();
  uint64_t firstNodeID = node->id();
  ASSERT_EQ(firstNodeID, 216);

  node = trace->getNextIssuableNode();
  uint64_t secondNodeID = node->id();
  ASSERT_EQ(secondNodeID, 432);
}

TEST_F(ETFeederTest, ConstructorNodeValuesTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->getNextIssuableNode();
  ChakraProtoMsg::NodeType firstNodeType = node->type();
  ASSERT_EQ(firstNodeType, ChakraProtoMsg::COMP_NODE);
  ASSERT_TRUE(node->is_cpu_op());

  std::string attr = "rf_id";
  ChakraProtoMsg::AttributeProto rf_id = node->get_other_attr(attr);
  ASSERT_EQ(rf_id.int64_val(), 2);

  node = trace->getNextIssuableNode();
  uint64_t secondNodeType = node->type();
  ASSERT_EQ(secondNodeType, ChakraProtoMsg::COMM_COLL_NODE);
  ASSERT_TRUE(node->is_cpu_op());

  rf_id = node->get_other_attr(attr);
  ASSERT_EQ(rf_id.int64_val(), 110);
}

TEST_F(ETFeederTest, ConstructorETFeederTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->getNextIssuableNode();
  std::vector<std::shared_ptr<Chakra::ETFeederNode>> children =
      node->getChildren();
  ASSERT_EQ(children[0]->id(), 217);
  ASSERT_EQ(children[1]->id(), 430);
  ASSERT_EQ(children[2]->id(), 435);
}

TEST_F(ETFeederTest, RemoveTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->lookupNode(216);
  ASSERT_EQ(node->id(), 216);
  trace->removeNode(216);
  freopen("/dev/null", "w", stderr);
  try {
    node = trace->lookupNode(216);
    ASSERT_TRUE(false) << "node should be removed \n";
  } catch (const std::exception& e) {
    // this is the desired behaviour
  }
  freopen("/dev/tty", "w", stderr);
}

TEST_F(ETFeederTest, RemoveAndGetNextTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->lookupNode(216);
  ASSERT_EQ(node->id(), 216);
  trace->removeNode(216);
  node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 216);
}

TEST_F(ETFeederTest, FreeChildrenTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->lookupNode(216);
  ASSERT_EQ(node->id(), 216);
  trace->freeChildrenNodes(216);
  node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 216);
  node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 217);
}

TEST_F(ETFeederTest, HasNodesToIssueTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 216);
  ASSERT_TRUE(trace->hasNodesToIssue());
  trace->removeNode(5);
  ASSERT_TRUE(trace->hasNodesToIssue());
}

TEST_F(ETFeederTest, PushBackIssuableNodeTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node;
  trace->pushBackIssuableNode(217);
  node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 216);
  node = trace->getNextIssuableNode();
  ASSERT_EQ(node->id(), 217);
}

TEST_F(ETFeederTest, AddNodeTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node;
  node = trace->lookupNode(216);
  trace->removeNode(216);
  trace->addNode(node);
  std::shared_ptr<Chakra::ETFeederNode> node2;
  node2 = trace->lookupNode(216);
  ASSERT_EQ(node2->id(), 216);
}

TEST_F(ETFeederTest, NodeGetChildrenTest) {
  SetUp("tests/data/chakra.0.et");
  std::shared_ptr<Chakra::ETFeederNode> node;
  node = trace->lookupNode(216);
  std::vector<std::shared_ptr<Chakra::ETFeederNode>> children =
      node->getChildren();
  ASSERT_EQ(children[0]->id(), 217);
  ASSERT_EQ(children[2]->id(), 435);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
