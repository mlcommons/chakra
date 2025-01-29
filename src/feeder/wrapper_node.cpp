#include "wrapper_node.h"

// WrapperNode default constructor
WrapperNode::WrapperNode() {}

// WrapperNode copy constructor
WrapperNode::WrapperNode(const WrapperNode& t) {
  // Copy the attributes from the original instance to the new instance
  format_type_ = t.format_type_;
  et_feeder_ = t.et_feeder_;
  node_ = t.node_;
  data_ = t.data_;
  node_idx_ = t.node_idx_;
  push_back_queue_proto = t.push_back_queue_proto;
  push_back_queue_json = t.push_back_queue_json;
  dep_graph_json = t.dep_graph_json;
  dep_free_node_id_set_json = t.dep_free_node_id_set_json;
  dep_free_node_queue_json = t.dep_free_node_queue_json;
  dep_unresolved_node_set_json = t.dep_unresolved_node_set_json;
  window_size_json = t.window_size_json;
}

// WrapperNode create
// format_type_ is assigned based on the extension of the file
void WrapperNode::createWrapper(std::string filename) {
  std::string ext = filename.substr(filename.find_last_of(".") + 1);
  if (ext == "et") {
    std::cout << "Using Protobuf format" << std::endl;
    format_type_ = Protobuf;
    et_feeder_ = new Chakra::ETFeeder(filename);
  } else if (ext == "json") {
    std::cout << "Using JSON format" << std::endl;
    format_type_ = JSON;
    json_et_complete_ = false;
    jsonfile_.open(filename);
    data_ = json::parse(jsonfile_); // Parse JSON file
    window_size_json = data_["workload_graph"].size(); // Number of nodes
    // For legacy purposes. The entire JSON file is read at once
    readNextWindow();
  } else {
    std::cerr << "Error: File format not supported." << std::endl;
    exit(-1);
  }
}

// WrapperNode constructor
WrapperNode::WrapperNode(std::string filename) {
  createWrapper(filename);
}

// Release memory
void WrapperNode::releaseMemory() {
  switch (format_type_) {
    case Protobuf: {
      delete et_feeder_;
      break;
    }
    case JSON: {
      jsonfile_.close();
      break;
    }
    default: {
      std::cerr << "Error in releaseMemory()" << std::endl;
      exit(-1);
    }
  }
}

WrapperNode::~WrapperNode() {}

// Return protobuf node
std::shared_ptr<Chakra::ETFeederNode> WrapperNode::getProtobufNode() {
  return node_;
}

// Return JSON node
JSONNode WrapperNode::getJSONNode() {
  return json_node_;
}

// Find the index in JSON dictionary
int64_t WrapperNode::findNodeIndexJSON(uint64_t node_id) {
  int64_t i;
  for (i = 0; i < window_size_json; i++) {
    if (data_["workload_graph"][i]["Id"] == node_id) {
      break;
    }
  }
  return i;
}

// Overloaded function - addNode
// Add JSON node to dependency graph
void WrapperNode::addNode(JSONNode node) {
  dep_graph_json[node.id()] = node;
}

// Add Protobuf node to dependency graph
void WrapperNode::addNode(std::shared_ptr<Chakra::ETFeederNode> node) {
  et_feeder_->addNode(node);
}

// Remove node from dependency graph
void WrapperNode::removeNode(uint64_t node_id) {
  switch (format_type_) {
    case Protobuf: {
      et_feeder_->removeNode(node_id);
      break;
    }
    case JSON: {
      dep_graph_json.erase(node_id);
      if (!json_et_complete_ &&
          (dep_free_node_queue_json.size() < window_size_json)) {
        readNextWindow();
      }
      break;
    }
    default: {
      std::cerr << "Error in removeNode()" << std::endl;
      exit(-1);
    }
  }
}

// Read nodes in graph
// node_idx is the continuous index of the JSON nodes and is different from
// node_id
JSONNode WrapperNode::readNode(uint64_t node_idx) {
  JSONNode node(data_, node_idx);
  bool dep_unresolved = false;
  for (int i = 0; i < node.data_deps.size(); ++i) {
    auto parent_node = dep_graph_json.find(node.data_deps[i]);
    if (parent_node != dep_graph_json.end()) {
      parent_node->second.addChild(
          node); // Add node as a child to the parent node
    } else {
      dep_unresolved = true;
      node.addDepUnresolvedParentID(node.data_deps[i]);
    }
  }

  if (dep_unresolved) {
    dep_unresolved_node_set_json.emplace(node);
  }

  return node;
}

// Read nodes in a window
// For JSON, the entire graph is read in a single window
void WrapperNode::readNextWindow() {
  uint64_t num_read = 0;
  do {
    if (num_read >= window_size_json) {
      json_et_complete_ = true;
      break;
    }
    JSONNode new_node = readNode(num_read);
    addNode(new_node);
    ++num_read;
    resolveDep();
  } while ((num_read < 256 * window_size_json) ||
           (dep_unresolved_node_set_json.size() !=
            0)); // arbitrarily large 256 * window_size_json

  for (auto node_id_node : dep_graph_json) {
    uint64_t node_id = node_id_node.first;
    JSONNode node(node_id_node.second);
    // Unordered set does not allow duplicates. So, count returns 1 if key
    // exists, 0 otherwise
    if ((dep_free_node_id_set_json.count(node_id) == 0) &&
        (node.data_deps.size() == 0)) {
      dep_free_node_id_set_json.emplace(node_id);
      dep_free_node_queue_json.emplace(node);
    }
  }
}

// Resolve dependencies
void WrapperNode::resolveDep() {
  switch (format_type_) {
    case Protobuf: {
      et_feeder_->resolveDep();
      break;
    }
    case JSON: {
      // Loop over unresolved nodes
      for (auto it = dep_unresolved_node_set_json.begin();
           it != dep_unresolved_node_set_json.end();) {
        JSONNode node = *it;
        std::vector<uint64_t> dep_unresolved_parent_ids_json =
            node.getDepUnresolvedParentIDs();
        // Loop over unresolved parent IDs
        for (auto inner_it = dep_unresolved_parent_ids_json.begin();
             inner_it != dep_unresolved_parent_ids_json.end();) {
          auto parent_node = dep_graph_json.find(*inner_it);
          if (parent_node != dep_graph_json.end()) {
            // Add current node as a child to the parent
            parent_node->second.addChild(node);
            inner_it = dep_unresolved_parent_ids_json.erase(inner_it);
          } else {
            ++inner_it;
          }
        }
        if (dep_unresolved_parent_ids_json.size() == 0) {
          it = dep_unresolved_node_set_json.erase(it);
        } else {
          node.setDepUnresolvedParentIDs(dep_unresolved_parent_ids_json);
          ++it;
        }
      }
      break;
    }
    default: {
      std::cerr << "Error in resolveDep()" << std::endl;
      exit(-1);
    }
  }
}

// Push dependency free nodes
void WrapperNode::pushBackIssuableNode(uint64_t node_id) {
  switch (format_type_) {
    case Protobuf: {
      et_feeder_->pushBackIssuableNode(node_id);
      break;
    }
    case JSON: {
      JSONNode node = dep_graph_json[node_id];
      dep_free_node_id_set_json.emplace(node_id);
      dep_free_node_queue_json.emplace(node);
      break;
    }
    default: {
      std::cerr << "Error in pushBackIssuableNode()" << std::endl;
      exit(-1);
    }
  }
}

// Free children
void WrapperNode::freeChildrenNodes(uint64_t node_id) {
  switch (format_type_) {
    case Protobuf: {
      et_feeder_->freeChildrenNodes(node_id);
      break;
    }
    case JSON: {
      JSONNode node = dep_graph_json[node_id];
      for (auto child : node.getChildren()) {
        for (auto it = child.data_deps.begin(); it != child.data_deps.end();
             ++it) {
          if (*it == node_id) {
            child.data_deps.erase(it);
            break;
          }
        }
        if (child.data_deps.size() == 0) {
          dep_free_node_id_set_json.emplace(child.id());
          dep_free_node_queue_json.emplace(child);
        }
      }
      break;
    }
    default: {
      std::cerr << "Error in freeChildrenNodes()" << std::endl;
      exit(-1);
    }
  }
}

// Check if the node is valid
bool WrapperNode::isValidNode() {
  switch (format_type_) {
    case Protobuf: {
      if (node_ == nullptr)
        return false;
      else
        return true;
    }
    case JSON: {
      if (node_idx_ < 0)
        return false;
      else
        return true;
    }
    default: {
      std::cerr << "Error in isValid()" << std::endl;
      exit(-1);
    }
  }
}

// Push node to queue
void WrapperNode::push_to_queue() {
  switch (format_type_) {
    case Protobuf: {
      push_back_queue_proto.push(node_);
      break;
    }
    case JSON: {
      push_back_queue_json.push(json_node_);
      break;
    }
    default: {
      std::cerr << "Error in push_to_queue()" << std::endl;
      exit(-1);
    }
  }
}

// Check if queue is empty
bool WrapperNode::is_queue_empty() {
  switch (format_type_) {
    case Protobuf: {
      return push_back_queue_proto.empty();
    }
    case JSON: {
      return push_back_queue_json.empty();
    }
    default: {
      std::cerr << "Error in is_queue_empty()" << std::endl;
      exit(-1);
    }
  }
}

// Get element in the queue front
void WrapperNode::queue_front() {
  switch (format_type_) {
    case Protobuf: {
      node_ = push_back_queue_proto.front();
      break;
    }
    case JSON: {
      json_node_ = push_back_queue_json.front();
      break;
    }
    default: {
      std::cerr << "Error in queue_front()" << std::endl;
      exit(-1);
    }
  }
}

// Pop node from queue
void WrapperNode::pop_from_queue() {
  switch (format_type_) {
    case Protobuf: {
      push_back_queue_proto.pop();
      break;
    }
    case JSON: {
      push_back_queue_json.pop();
      break;
    }
    default: {
      std::cerr << "Error in pop_from_queue()" << std::endl;
      exit(-1);
    }
  }
}

// Get next issuable node from dependency free queue
void WrapperNode::getNextIssuableNode() {
  switch (format_type_) {
    case Protobuf: {
      node_ = et_feeder_->getNextIssuableNode();
      break;
    }
    case JSON: {
      if (dep_free_node_queue_json.size() != 0) {
        json_node_ = dep_free_node_queue_json.top();
        node_idx_ = findNodeIndexJSON(json_node_.id());
        dep_free_node_id_set_json.erase(json_node_.id());
        dep_free_node_queue_json.pop();
      } else
        node_idx_ = -1;
      break;
    }
    default: {
      std::cerr << "Error in getNextIssuableNode()" << std::endl;
      exit(-1);
    }
  }
}

// Get node ID
uint64_t WrapperNode::getNodeID() {
  switch (format_type_) {
    case Protobuf: {
      return node_->id();
    }
    case JSON: {
      return json_node_.id();
    }
    default: {
      std::cerr << "Error in getNodeID()" << std::endl;
      exit(-1);
    }
  }
}

// Get node name
std::string WrapperNode::getNodeName() {
  switch (format_type_) {
    case Protobuf: {
      return node_->name();
    }
    case JSON: {
      return json_node_.name();
    }
    default: {
      std::cerr << "Error in getNodeName()" << std::endl;
      exit(-1);
    }
  }
}

// Get node type
int WrapperNode::getNodeType() {
  switch (format_type_) {
    case Protobuf: {
      return node_->type();
    }
    case JSON: {
      return json_node_.type();
    }
    default: {
      std::cerr << "Error in getNodeType()" << std::endl;
      exit(-1);
    }
  }
}

// Check if CPU operation
bool WrapperNode::isCPUOp() {
  switch (format_type_) {
    case Protobuf: {
      return node_->is_cpu_op();
    }
    case JSON: {
      return json_node_.isCPUOp();
    }
    default: {
      std::cerr << "Error in isCPUOp()" << std::endl;
      exit(-1);
    }
  }
}

// Get runtime
uint64_t WrapperNode::getRuntime() {
  switch (format_type_) {
    case Protobuf: {
      return node_->runtime();
    }
    case JSON: {
      return json_node_.getRuntime();
    }
    default: {
      std::cerr << "Error in getRuntime()" << std::endl;
      exit(-1);
    }
  }
}

// Get num ops
uint64_t WrapperNode::getNumOps() {
  switch (format_type_) {
    case Protobuf: {
      return node_->num_ops();
    }
    case JSON: {
      return json_node_.getNumOps();
    }
    default: {
      std::cerr << "Error in getNumOps()" << std::endl;
      exit(-1);
    }
  }
}

// Get tensor size
uint64_t WrapperNode::getTensorSize() {
  switch (format_type_) {
    case Protobuf: {
      return node_->tensor_size();
    }
    case JSON: {
      return json_node_.getTensorSize();
    }
    default: {
      std::cerr << "Error in getTensorSize()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm type
int64_t WrapperNode::getCommType() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_type();
    }
    case JSON: {
      return json_node_.getCommType();
    }
    default: {
      std::cerr << "Error in getCommType()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm priority
uint32_t WrapperNode::getCommPriority() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_priority();
    }
    case JSON: {
      return json_node_.getCommPriority();
    }
    default: {
      std::cerr << "Error in getCommPriority()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm size
uint64_t WrapperNode::getCommSize() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_size();
    }
    case JSON: {
      return json_node_.getCommSize();
    }
    default: {
      std::cerr << "Error in getCommSize()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm src
uint32_t WrapperNode::getCommSrc() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_src();
    }
    case JSON: {
      return json_node_.getCommSrc();
    }
    default: {
      std::cerr << "Error in getCommSrc()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm dst
uint32_t WrapperNode::getCommDst() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_dst();
    }
    case JSON: {
      return json_node_.getCommDst();
    }
    default: {
      std::cerr << "Error in getCommDst()" << std::endl;
      exit(-1);
    }
  }
}

// Get comm tag
uint32_t WrapperNode::getCommTag() {
  switch (format_type_) {
    case Protobuf: {
      return node_->comm_tag();
    }
    case JSON: {
      return json_node_.getCommTag();
    }
    default: {
      std::cerr << "Error in getCommTag()" << std::endl;
      exit(-1);
    }
  }
}

// Check if has more nodes to issue
bool WrapperNode::hasNodesToIssue() {
  switch (format_type_) {
    case Protobuf: {
      return et_feeder_->hasNodesToIssue();
    }
    case JSON: {
      return !(dep_graph_json.empty() && dep_free_node_queue_json.empty());
    }
    default: {
      std::cerr << "Error in hasNodesToIssue()" << std::endl;
      exit(-1);
    }
  }
}

// Lookup Node
void WrapperNode::lookupNode(uint64_t node_id) {
  switch (format_type_) {
    case Protobuf: {
      node_ = et_feeder_->lookupNode(node_id);
      break;
    }
    case JSON: {
      try {
        json_node_ = dep_graph_json.at(node_id);
      } catch (const std::out_of_range& e) {
        std::cerr << "looking for node_id=" << node_id
                  << " in dep graph, however, not loaded yet" << std::endl;
        throw(e);
      }
      break;
    }
    default: {
      std::cerr << "Error in lookupNode()" << std::endl;
      exit(-1);
    }
  }
}

// Overloaded function returns children protobuf nodes
void WrapperNode::getChildren(
    std::vector<std::shared_ptr<Chakra::ETFeederNode>>& childrenNodes) {
  childrenNodes = node_->getChildren();
}

// Overloaded function returns children JSON nodes
void WrapperNode::getChildren(std::vector<JSONNode>& childrenNodes) {
  childrenNodes = json_node_.getChildren();
}