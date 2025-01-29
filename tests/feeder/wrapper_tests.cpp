#include <gtest/gtest.h>
#include "WrapperNode.h"

class WrapperNodeTest : public ::testing::Test {
 protected:
  WrapperNodeTest() {}
  virtual ~WrapperNodeTest() {}

  void SetUp(const std::string& filename) {
    node.createWrapper(filename);
  }

  virtual void TearDown() {
    node.releaseMemory();
  }

  WrapperNode node;
};

TEST_F(WrapperNodeTest, ConstructorNodeIDTest) {
  // tests/data/small_chakra.0.json is a pruned dataset for quick tests
  // tests/data/chakra.0.json is the full dataset, which is also available
  SetUp("tests/data/small_chakra.0.json");
  node.getNextIssuableNode();
  uint64_t firstNodeID = node.getNodeID();
  ASSERT_EQ(firstNodeID, 216);

  node.getNextIssuableNode();
  uint64_t secondNodeID = node.getNodeID();
  ASSERT_EQ(secondNodeID, 432);
}

TEST_F(WrapperNodeTest, ConstructorNodeValuesTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.getNextIssuableNode();
  uint64_t firstNodeType = node.getNodeType();
  ASSERT_EQ(firstNodeType, ChakraProtoMsg::COMP_NODE);
  ASSERT_TRUE(node.isCPUOp());

  node.getNextIssuableNode();
  uint64_t secondNodeType = node.getNodeType();
  ASSERT_EQ(secondNodeType, ChakraProtoMsg::COMM_COLL_NODE);
  ASSERT_TRUE(node.isCPUOp());
}

TEST_F(WrapperNodeTest, ConstructorWrapperNodeTest) {
  std::string filename = "tests/data/small_chakra.0.json";
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  SetUp(filename);
  node.getNextIssuableNode();
  if (ext == "et") {
    std::vector<std::shared_ptr<Chakra::ETFeederNode>> children;
    node.getChildren(children);
    ASSERT_EQ(children[0]->id(), 217);
    ASSERT_EQ(children[1]->id(), 430);
    ASSERT_EQ(children[2]->id(), 435);
  } else if (ext == "json") {
    std::vector<JSONNode> children;
    node.getChildren(children);
    ASSERT_EQ(children[0].id(), 217);
    ASSERT_EQ(children[1].id(), 430);
    ASSERT_EQ(children[2].id(), 435);
  }
}

TEST_F(WrapperNodeTest, RemoveTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.lookupNode(216);
  ASSERT_EQ(node.getNodeID(), 216);
  node.removeNode(216);
  freopen("/dev/null", "w", stderr);
  try {
    node.lookupNode(216);
    ASSERT_TRUE(false) << "node should be removed \n";
  } catch (const std::exception& e) {
    // this is the desired behaviour
  }
  freopen("/dev/tty", "w", stderr);
}

TEST_F(WrapperNodeTest, RemoveAndGetNextTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.lookupNode(216);
  ASSERT_EQ(node.getNodeID(), 216);
  node.removeNode(216);
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 216);
}

TEST_F(WrapperNodeTest, FreeChildrenTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.lookupNode(216);
  ASSERT_EQ(node.getNodeID(), 216);
  node.freeChildrenNodes(216);
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 216);
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 217);
}

TEST_F(WrapperNodeTest, HasNodesToIssueTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 216);
  ASSERT_TRUE(node.hasNodesToIssue());
  node.removeNode(5);
  ASSERT_TRUE(node.hasNodesToIssue());
}

TEST_F(WrapperNodeTest, PushBackIssuableNodeTest) {
  SetUp("tests/data/small_chakra.0.json");
  node.pushBackIssuableNode(217);
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 216);
  node.getNextIssuableNode();
  ASSERT_EQ(node.getNodeID(), 217);
}

TEST_F(WrapperNodeTest, AddNodeTest) {
  std::string filename = "tests/data/small_chakra.0.json";
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  SetUp(filename);
  if (ext == "et") {
    std::shared_ptr<Chakra::ETFeederNode> pnode1;
    node.lookupNode(216);
    pnode1 = node.getProtobufNode();
    node.removeNode(216);
    node.addNode(pnode1);
    std::shared_ptr<Chakra::ETFeederNode> pnode2;
    node.lookupNode(216);
    pnode2 = node.getProtobufNode();
    ASSERT_EQ(pnode2->id(), 216);
  } else if (ext == "json") {
    JSON jnode1;
    node.lookupNode(216);
    jnode1 = node.getJSONNode();
    node.removeNode(216);
    node.addNode(jnode1);
    JSONNode jnode2;
    node.lookupNode(216);
    jnode2 = node.getJSONNode();
    ASSERT_EQ(jnode2.id(), 216);
  }
}

TEST_F(WrapperNodeTest, NodeGetChildrenTest) {
  std::string filename = "tests/data/small_chakra.0.json";
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  SetUp(filename);
  node.lookupNode(216);
  if (ext == "et") {
    std::vector<std::shared_ptr<Chakra::ETFeederNode>> children;
    node.getChildren(children);
    ASSERT_EQ(children[0]->id(), 217);
    ASSERT_EQ(children[2]->id(), 435);
  } else if (ext == "json") {
    std::vector<JSONNode> children;
    node.getChildren(children);
    ASSERT_EQ(children[0].id(), 217);
    ASSERT_EQ(children[2].id(), 435);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
