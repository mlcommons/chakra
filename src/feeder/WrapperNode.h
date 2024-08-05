#pragma once

#include "et_feeder.h"
#include "et_feeder_node.h"
#include "JSONNode.h"

using json = nlohmann::json;

enum format {
	Protobuf,
	JSON
};

// WrapperNode class wraps protobuf and JSON
class WrapperNode {
	private:
		enum format format_type_;
		Chakra::ETFeeder* et_feeder_;
		std::shared_ptr<Chakra::ETFeederNode> node_ {nullptr};
		std::ifstream jsonfile_;
		json data_;
		JSONNode json_node_;
		int64_t node_idx_ = -1;
		int32_t involved_dim_size_ = 1;
		std::vector<bool> involved_dim_;
		std::queue<std::shared_ptr<Chakra::ETFeederNode>> push_back_queue_proto;
		std::queue<JSONNode> push_back_queue_json;
		std::unordered_map<int64_t, JSONNode> dep_graph_json{};
  		std::unordered_set<int64_t> dep_free_node_id_set_json{};
		std::priority_queue<
			JSONNode, //type of stored elements
			std::vector<JSONNode>, // underlying container to store elements
			CompareJSONNodesGT> // compare type providing a strick weak ordering
			dep_free_node_queue_json{};
  		std::unordered_set<JSONNode, std::hash<JSONNode>> dep_unresolved_node_set_json{};
		int window_size_json;
	
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
		void removeNode(int64_t node_id);
		void readNextWindow();
		JSONNode readNode(int64_t node_id);
		void resolveDep();
		void pushBackIssuableNode(int64_t node_id);
		void freeChildrenNodes(int64_t node_id);
		void addDepUnresolvedParentID(int64_t node_id);
		std::vector<int64_t> getDepUnresolvedParentIDs();
		void setDepUnresolvedParentIDs(std::vector<int64_t> const& dep_unresolved_parent_ids);
		bool isValidNode();
		void push_to_queue();
		bool is_queue_empty();
		void queue_front();
		void pop_from_queue();
		void getNextIssuableNode();
		int64_t getNodeID();
		std::string getNodeName();
		int getNodeType();
		bool isCPUOp();
		int64_t getRuntime();
		int64_t getNumOps();
		int64_t getTensorSize(); 
		int64_t getCommType();
		int32_t getCommPriority(); 
		int64_t getCommSize();
		int32_t getCommSrc();
		int32_t getCommDst();
		int32_t getCommTag();
		int32_t getInvolvedDimSize();
		bool getInvolvedDim(int i);
		bool hasNodesToIssue();
		void lookupNode(int64_t node_id);
		void getChildren(std::vector<std::shared_ptr<Chakra::ETFeederNode>>& childrenNodes);
		void getChildren(std::vector<JSONNode>& childrenNodes);
		int64_t findNodeIndexJSON(int64_t node_id);
};