#pragma once

#include "et_feeder.h"
#include "et_feeder_node.h"
#include "json_node.h"

using json = nlohmann::json;

enum format { Protobuf, JSON };

// WrapperNode class wraps protobuf and JSON
class WrapperNode {
 private:
  enum format format_type_;
  Chakra::ETFeeder* et_feeder_;
  std::shared_ptr<Chakra::ETFeederNode> node_{nullptr};
  std::ifstream jsonfile_;
  json data_;
  JSONNode json_node_;
  int64_t node_idx_ = -1;
  std::queue<std::shared_ptr<Chakra::ETFeederNode>> push_back_queue_proto;
  std::queue<JSONNode> push_back_queue_json;
  std::unordered_map<uint64_t, JSONNode> dep_graph_json{};
  std::unordered_set<uint64_t> dep_free_node_id_set_json{};
  std::priority_queue<
      JSONNode, // type of stored elements
      std::vector<JSONNode>, // underlying container to store elements
      CompareJSONNodesGT> // compare type providing a strick weak ordering
      dep_free_node_queue_json{};
  std::unordered_set<JSONNode, std::hash<JSONNode>>
      dep_unresolved_node_set_json{};
  int window_size_json;
  bool json_et_complete_;

 public:
  WrapperNode();
  WrapperNode(const WrapperNode& t);
  WrapperNode(std::string filename);
  ~WrapperNode();
  void releaseMemory();
  void createWrapper(std::string filename);
  std::shared_ptr<Chakra::ETFeederNode> getProtobufNode();
  JSONNode getJSONNode();
  void addNode(JSONNode node);
  void addNode(std::shared_ptr<Chakra::ETFeederNode> node);
  void removeNode(uint64_t node_id);
  void readNextWindow();
  JSONNode readNode(uint64_t node_id);
  void resolveDep();
  void pushBackIssuableNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);
  bool isValidNode();
  void push_to_queue();
  bool is_queue_empty();
  void queue_front();
  void pop_from_queue();
  void getNextIssuableNode();
  uint64_t getNodeID();
  std::string getNodeName();
  int getNodeType();
  bool isCPUOp();
  uint64_t getRuntime();
  uint64_t getNumOps();
  uint64_t getTensorSize();
  int64_t getCommType();
  uint32_t getCommPriority();
  uint64_t getCommSize();
  uint32_t getCommSrc();
  uint32_t getCommDst();
  uint32_t getCommTag();
  bool hasNodesToIssue();
  void lookupNode(uint64_t node_id);
  void getChildren(
      std::vector<std::shared_ptr<Chakra::ETFeederNode>>& childrenNodes);
  void getChildren(std::vector<JSONNode>& childrenNodes);
  int64_t findNodeIndexJSON(uint64_t node_id);
};