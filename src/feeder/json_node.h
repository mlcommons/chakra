#pragma once

#include <json/json.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <set>
#include <string>

using json = nlohmann::json;

enum NodeType : int {
  INVALID_NODE = 0,
  METADATA_NODE = 1,
  MEM_LOAD_NODE = 2,
  MEM_STORE_NODE = 3,
  COMP_NODE = 4,
  COMM_SEND_NODE = 5,
  COMM_RECV_NODE = 6,
  COMM_COLL_NODE = 7
};

class JSONNode {
 private:
  uint64_t node_id;
  std::string node_name;
  int node_type;
  bool is_cpu_op;
  uint64_t runtime;
  uint64_t num_ops;
  uint64_t tensor_size;
  int64_t comm_type;
  uint32_t comm_priority;
  uint64_t comm_size;
  uint32_t comm_src;
  uint32_t comm_dst;
  uint32_t comm_tag;

 public:
  std::vector<uint64_t> data_deps{};
  std::vector<uint64_t> dep_unresolved_parent_ids_json{};
  std::vector<JSONNode> children_vec_json{};

  // Compare function for set
  struct CompareJSONNodesLT {
    bool operator()(const JSONNode& a, const JSONNode& b) const {
      return a.node_id < b.node_id;
    }
  };
  std::set<JSONNode, CompareJSONNodesLT> children_set_json{};

  JSONNode();
  JSONNode(const JSONNode& t);
  JSONNode(json data, uint64_t id);
  uint64_t id() const;
  std::string name() const;
  int type() const;
  bool isCPUOp() const;
  uint64_t getRuntime() const;
  uint64_t getNumOps() const;
  uint64_t getTensorSize() const;
  int64_t getCommType() const;
  uint32_t getCommPriority() const;
  uint64_t getCommSize() const;
  uint32_t getCommSrc() const;
  uint32_t getCommDst() const;
  uint32_t getCommTag() const;
  void addDepUnresolvedParentID(uint64_t node_id);
  std::vector<uint64_t> getDepUnresolvedParentIDs();
  void setDepUnresolvedParentIDs(
      std::vector<uint64_t> const& dep_unresolved_parent_ids);
  void addChild(JSONNode node);
  std::vector<JSONNode> getChildren();

  // Define the == operator for comparison
  bool operator==(const JSONNode& other) const {
    return node_id == other.node_id && node_name == other.node_name &&
        node_type == other.node_type && is_cpu_op == other.is_cpu_op &&
        runtime == other.runtime && num_ops == other.num_ops &&
        tensor_size == other.tensor_size && comm_type == other.comm_type &&
        comm_priority == other.comm_priority && comm_size == other.comm_size &&
        comm_src == other.comm_src && comm_dst == other.comm_dst &&
        comm_tag == other.comm_tag && data_deps == other.data_deps &&
        dep_unresolved_parent_ids_json ==
        other.dep_unresolved_parent_ids_json &&
        children_vec_json == other.children_vec_json &&
        children_set_json == other.children_set_json;
  }

  // Overload the assignment operator
  JSONNode& operator=(const JSONNode& other) {
    if (this != &other) {
      // Copy all member variables
      node_id = other.node_id;
      node_name = other.node_name;
      node_type = other.node_type;
      is_cpu_op = other.is_cpu_op;
      runtime = other.runtime;
      num_ops = other.num_ops;
      tensor_size = other.tensor_size;
      comm_type = other.comm_type;
      comm_priority = other.comm_priority;
      comm_size = other.comm_size;
      comm_src = other.comm_src;
      comm_dst = other.comm_dst;
      comm_tag = other.comm_tag;
      data_deps = other.data_deps;
      dep_unresolved_parent_ids_json = other.dep_unresolved_parent_ids_json;
      children_vec_json = other.children_vec_json;
      children_set_json = other.children_set_json;
    }
    return *this;
  }
};

// Define a custom hash function for unordered set
namespace std {
template <>
struct hash<JSONNode> {
  std::size_t operator()(const JSONNode& node) const {
    std::size_t h1 = std::hash<int64_t>()(node.id());
    std::size_t h2 = std::hash<std::string>()(node.name());
    std::size_t h3 = std::hash<int>()(node.type());
    std::size_t h4 = std::hash<bool>()(node.isCPUOp());
    std::size_t h5 = std::hash<int64_t>()(node.getRuntime());

    // A prime number for bit manipulation
    const std::size_t prime = 31;

    // Combine the hash of the current member with the hashes of the previous
    // members
    std::size_t hash = h1;
    hash = hash * prime + h2;
    hash = hash * prime + h3;
    hash = hash * prime + h4;
    hash = hash * prime + h5;

    return hash;
  }
};
} // namespace std

// Compare function for JSON node for priority queue
struct CompareJSONNodesGT
    : public std::binary_function<JSONNode, JSONNode, bool> {
  bool operator()(const JSONNode lhs, const JSONNode rhs) const {
    return lhs.id() > rhs.id();
  }
};