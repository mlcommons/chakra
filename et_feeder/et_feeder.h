#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "third_party/utils/protoio.hh"
#include "et_feeder/et_feeder_node.h"

namespace Chakra {

class ETFeeder {
 public:
  ETFeeder(std::string filename);
  ~ETFeeder();

  void addNode(std::shared_ptr<ETFeederNode> node);
  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  std::shared_ptr<ETFeederNode> getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<ETFeederNode> lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);

 private:
  std::shared_ptr<ETFeederNode> readNode();
  void readNextWindow();
  void resolveDep();

  ProtoInputStream trace_;
  const uint32_t window_size_;
  bool et_complete_;

  std::unordered_map<uint64_t, std::shared_ptr<ETFeederNode>> dep_graph_{};
  std::unordered_set<uint64_t> dep_free_node_id_set_{};
  std::queue<std::shared_ptr<ETFeederNode>> dep_free_node_queue_{};
  std::unordered_set<std::shared_ptr<ETFeederNode>> dep_unresolved_node_set_{};
};

} // namespace Chakra
