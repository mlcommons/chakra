#include <iostream>
#include <string>

#include "third_party/utils/cxxopts.hpp"
#include "third_party/utils/protoio.hh"
#include "et_generator/dependency_graph.h"

using namespace std;
using namespace Chakra;

namespace {
constexpr uint64_t k_flops = 2*512*512;

string getFilename(string exp_name, int npu_id) {
  return exp_name + "." + to_string(npu_id) + ".eg";
}

void oneCompNode(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    node->set_name("COMP_NODE");
    node->set_simulated_run_time(5);

    dg->flushEG();
    delete dg;
  }
}

void twoCompNodesIndependent(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    node->set_name("COMP_NODE");
    node->set_simulated_run_time(5);

    node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    node->set_name("COMP_NODE");
    node->set_simulated_run_time(5);

    dg->flushEG();
    delete dg;
  }
}

void twoCompNodesDependent(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node1, *node2;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    node1->set_name("COMP_NODE");
    node1->set_simulated_run_time(5);

    node2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    node2->set_name("COMP_NODE");
    node2->set_simulated_run_time(5);

    dg->assignDep(node1, node2);

    dg->flushEG();
    delete dg;
  }
}

void oneCommNodeAllReduce(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    node->set_name("COMM_COLL_NODE");
    for (int i = 0; i < num_dims; ++i) {
      node->add_involved_dim(true);
    }
    node->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    node->set_comm_size(65536);

    dg->flushEG();
    delete dg;
  }
}

void oneCommNodeAllToAll(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    node->set_name("COMM_COLL_NODE");
    for (int i = 0; i < num_dims; ++i) {
      node->add_involved_dim(true);
    }
    node->set_comm_type(ChakraProtoMsg::ALL_TO_ALL);
    node->set_comm_size(65536);

    dg->flushEG();
    delete dg;
  }
}

void oneCommNodeAllGather(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    node->set_name("COMM_COLL_NODE");
    for (int i = 0; i < num_dims; ++i) {
      node->add_involved_dim(true);
    }
    node->set_comm_type(ChakraProtoMsg::ALL_GATHER);
    node->set_comm_size(65536);

    dg->flushEG();
    delete dg;
  }
}

void oneCommNodeReduceScatter(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    node = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    node->set_name("COMM_COLL_NODE");
    for (int i = 0; i < num_dims; ++i) {
      node->add_involved_dim(true);
    }
    node->set_comm_type(ChakraProtoMsg::REDUCE_SCATTER);
    node->set_comm_size(65536);

    dg->flushEG();
    delete dg;
  }
}

void commNodesSingleSendSingleRecv(int num_npus, int num_dims,
                                        uint32_t comm_size) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(
        getFilename(string(__func__) + "." + to_string(comm_size),
          npu_id));

    node = dg->addNode(ChakraProtoMsg::COMM_SEND_NODE);
    node->set_name("COMM_SEND_NODE");
    node->set_comm_src(npu_id);
    if (npu_id != num_npus-1)
      node->set_comm_dst(npu_id + 1);
    else
      node->set_comm_dst(0);
    node->set_comm_size(comm_size);
    node->set_comm_tag(0);

    node = dg->addNode(ChakraProtoMsg::COMM_RECV_NODE);
    node->set_name("COMM_RECV_NODE");
    if (npu_id != 0)
      node->set_comm_src(npu_id - 1);
    else
      node->set_comm_src(num_npus - 1);
    node->set_comm_dst(npu_id);
    node->set_comm_size(comm_size);
    node->set_comm_tag(0);

    dg->flushEG();
    delete dg;
  }
}

void invalidNodeCase1(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    invalid_node = dg->addNode(ChakraProtoMsg::INVALID_NODE);
    invalid_node->set_name("INVALID_NODE");

    dg->assignDep(comp_node, invalid_node);

    dg->flushEG();
    delete dg;
  }
}

void invalidNodeCase2(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    invalid_node = dg->addNode(ChakraProtoMsg::INVALID_NODE);
    invalid_node->set_name("INVALID_NODE");

    dg->assignDep(comp_node, invalid_node);

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void invalidNodeCase3(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    invalid_node = dg->addNode(ChakraProtoMsg::INVALID_NODE);
    invalid_node->set_name("INVALID_NODE");

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void invalidNodeCase4(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *comp_node, *invalid_node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    invalid_node = dg->addNode(ChakraProtoMsg::INVALID_NODE);
    invalid_node->set_name("INVALID_NODE");

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    dg->assignDep(invalid_node, comp_node);

    comp_node = dg->addNode(ChakraProtoMsg::COMP_NODE);
    comp_node->set_name("COMP_NODE");
    comp_node->set_simulated_run_time(5);

    dg->assignDep(invalid_node, comp_node);

    dg->flushEG();
    delete dg;
  }
}

void threeLayerDataParallel(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    Node *fwd_0 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_0->set_name("COMP_NODE_FWD_0");
    fwd_0->set_simulated_run_time(5);

    Node *fwd_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_1->set_name("COMP_NODE_FWD_1");
    fwd_1->set_simulated_run_time(5);

    Node *fwd_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_2->set_name("COMP_NODE_FWD_2");
    fwd_2->set_simulated_run_time(5);

    dg->assignDep(fwd_0, fwd_1);
    dg->assignDep(fwd_1, fwd_2);

    Node *bwd_wg_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_2->set_name("COMP_NODE_BWD_WG_2");
    bwd_wg_2->set_simulated_run_time(5);

    Node *bwd_ig_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_ig_2->set_name("COMP_NODE_BWD_IG_2");
    bwd_ig_2->set_simulated_run_time(5);

    Node *comm_2 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_2->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_2");
    for (int i = 0; i < num_dims; ++i) {
      comm_2->add_involved_dim(true);
    }
    comm_2->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_2->set_comm_size(65536);

    Node *bwd_wg_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_1->set_name("COMP_NODE_BWD_WG_1");
    bwd_wg_1->set_simulated_run_time(5);

    Node *bwd_ig_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_ig_1->set_name("COMP_NODE_BWD_IG_1");
    bwd_ig_1->set_simulated_run_time(5);

    Node *comm_1 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_1->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_1");
    for (int i = 0; i < num_dims; ++i) {
      comm_1->add_involved_dim(true);
    }
    comm_1->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_1->set_comm_size(65536);

    Node *bwd_wg_0 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_0->set_name("COMP_NODE_BWD_WG_0");
    bwd_wg_0->set_simulated_run_time(5);

    Node *comm_0 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_0->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_0");
    for (int i = 0; i < num_dims; ++i) {
      comm_0->add_involved_dim(true);
    }
    comm_0->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_0->set_comm_size(65536);

    dg->assignDep(fwd_2, bwd_wg_2);
    dg->assignDep(bwd_wg_2, bwd_ig_2);
    dg->assignDep(bwd_ig_2, bwd_wg_1);
    dg->assignDep(bwd_wg_1, bwd_ig_1);
    dg->assignDep(bwd_ig_1, bwd_wg_0);

    dg->assignDep(bwd_wg_2, comm_2);
    dg->assignDep(bwd_wg_1, comm_1);
    dg->assignDep(bwd_wg_0, comm_0);

    dg->flushEG();
    delete dg;
  }
}

void threeLayerDataParallelSequentiallyDependent(int num_npus, int num_dims) {
  DependencyGraph *dg;
  Node *node;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    int comm_src, comm_dst;
    if (npu_id != 0)
      comm_src = npu_id - 1;
    else
      comm_src = num_npus - 1;
    if (npu_id != num_npus-1)
      comm_dst = npu_id + 1;
    else
      comm_dst = 0;

    Node *fwd_0 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_0->set_name("COMP_NODE_FWD_0");
    fwd_0->set_simulated_run_time(5);

    Node *fwd_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_1->set_name("COMP_NODE_FWD_1");
    fwd_1->set_simulated_run_time(5);

    Node *fwd_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    fwd_2->set_name("COMP_NODE_FWD_2");
    fwd_2->set_simulated_run_time(5);

    dg->assignDep(fwd_0, fwd_1);
    dg->assignDep(fwd_1, fwd_2);

    Node *bwd_wg_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_2->set_name("COMP_NODE_BWD_WG_2");
    bwd_wg_2->set_simulated_run_time(5);

    Node *comm_2 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_2->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_2");
    for (int i = 0; i < num_dims; ++i) {
      comm_2->add_involved_dim(true);
    }
    comm_2->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_2->set_comm_size(65536);

    Node *bwd_ig_2 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_ig_2->set_name("COMP_NODE_BWD_IG_2");
    bwd_ig_2->set_simulated_run_time(5);

    Node *bwd_wg_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_1->set_name("COMP_NODE_BWD_WG_1");
    bwd_wg_1->set_simulated_run_time(5);

    Node *comm_1 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_1->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_1");
    for (int i = 0; i < num_dims; ++i) {
      comm_1->add_involved_dim(true);
    }
    comm_1->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_1->set_comm_size(65536);

    Node *bwd_ig_1 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_ig_1->set_name("COMP_NODE_BWD_IG_1");
    bwd_ig_1->set_simulated_run_time(5);

    Node *bwd_wg_0 = dg->addNode(ChakraProtoMsg::COMP_NODE);
    bwd_wg_0->set_name("COMP_NODE_BWD_WG_0");
    bwd_wg_0->set_simulated_run_time(5);

    Node *comm_0 = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
    comm_0->set_name("COMM_COLL_NODE_BWD_ALL_REDUCE_0");
    for (int i = 0; i < num_dims; ++i) {
      comm_0->add_involved_dim(true);
    }
    comm_0->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
    comm_0->set_comm_size(65536);

    dg->assignDep(fwd_2, bwd_wg_2);
    dg->assignDep(bwd_wg_2, comm_2);
    dg->assignDep(comm_2, bwd_ig_2);
    dg->assignDep(bwd_ig_2, bwd_wg_1);
    dg->assignDep(bwd_wg_1, comm_1);
    dg->assignDep(comm_1, bwd_ig_1);
    dg->assignDep(bwd_ig_1, bwd_wg_0);
    dg->assignDep(bwd_wg_0, comm_0);

    dg->flushEG();
    delete dg;
  }
}

void parallelismComparisonDataParallel(
    int num_npus, int num_dims,
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t k_flops) {
  DependencyGraph *dg;

  int num_layers = num_npus;
  Node *fwd_comp[num_layers],
  *bwd_wg_comp[num_layers], *bwd_ig_comp[num_layers],
  *bwd_wg_comm[num_layers];

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time = (2 * (m / num_npus) * k * n) / k_flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t bwd_wg_comm_size = n * k * data_type_size;

    for (int i = 0; i < num_layers; ++i) {
      fwd_comp[i] = dg->addNode(ChakraProtoMsg::COMP_NODE);
      fwd_comp[i]->set_name("COMP_NODE_FWD_" + to_string(i));
      fwd_comp[i]->set_simulated_run_time(fwd_simulated_run_time);
    }

    for (int i = 0; i < num_layers; ++i) {
      bwd_wg_comp[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMP_NODE);
      bwd_wg_comp[num_layers - i - 1]->set_name("COMP_NODE_BWD_WG_" + to_string(num_layers - i - 1));
      bwd_wg_comp[num_layers - i - 1]->set_simulated_run_time(bwd_simulated_run_time);

      bwd_wg_comm[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
      bwd_wg_comm[num_layers - i - 1]->set_name("COMM_COLL_NODE_BWD_WG_ALL_REDUCE_" + to_string(num_layers - i - 1));
      for (int j = 0; j < num_dims; ++j) {
        bwd_wg_comm[num_layers - i - 1]->add_involved_dim(true);
      }
      bwd_wg_comm[num_layers - i - 1]->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
      bwd_wg_comm[num_layers - i - 1]->set_comm_size(bwd_wg_comm_size);

      if (i != (num_layers - 1)) {
        bwd_ig_comp[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_ig_comp[num_layers - i - 1]->set_name("COMP_NODE_BWD_IG_" + to_string(num_layers - i - 1));
        bwd_ig_comp[num_layers - i - 1]->set_simulated_run_time(bwd_simulated_run_time);
      }
    }

    for (int i = 0; i < num_layers - 1; ++i) {
      dg->assignDep(fwd_comp[i], fwd_comp[i+1]);
    }

    dg->assignDep(fwd_comp[num_layers-1], bwd_wg_comp[num_layers-1]);

    for (int i = 0; i < num_layers; ++i) {
      if (i != 0) {
        dg->assignDep(bwd_wg_comp[i], bwd_ig_comp[i]);
      }
      dg->assignDep(bwd_wg_comp[i], bwd_wg_comm[i]);
    }

    for (int i = 1; i < num_layers; ++i) {
      dg->assignDep(bwd_ig_comp[i], bwd_wg_comp[i-1]);
    }

    dg->flushEG();
    delete dg;
  }
}

void parallelismComparisonModelParallel(
    int num_npus, int num_dims,
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t k_flops) {
  DependencyGraph *dg;

  int num_layers = num_npus;
  Node *fwd_comp[num_layers], *fwd_comm[num_layers],
  *bwd_wg_comp[num_layers], *bwd_ig_comp[num_layers], *bwd_ig_comm[num_layers],
  *bwd_wg_comp_prev, *bwd_ig_comm_prev;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time = (2 * m * k * (n / num_npus)) / k_flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t fwd_comm_size = m * (n / num_npus) * data_type_size;
    uint32_t bwd_ig_comm_size = m * k * data_type_size;

    for (int i = 0; i < num_layers; ++i) {
      fwd_comp[i] = dg->addNode(ChakraProtoMsg::COMP_NODE);
      fwd_comp[i]->set_name("COMP_NODE_FWD_" + to_string(i));
      fwd_comp[i]->set_simulated_run_time(fwd_simulated_run_time);

      fwd_comm[i] = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
      fwd_comm[i]->set_name("COMM_COLL_NODE_FWD_ALL_GATHER_" + to_string(i));
      for (int j = 0; j < num_dims; ++j) {
        fwd_comm[i]->add_involved_dim(true);
      }
      fwd_comm[i]->set_comm_type(ChakraProtoMsg::ALL_GATHER);
      fwd_comm[i]->set_comm_size(fwd_comm_size);

      dg->assignDep(fwd_comp[i], fwd_comm[i]);
      if (i != 0) {
        dg->assignDep(fwd_comm[i-1], fwd_comp[i]);
      }
    }

    for (int i = 0; i < num_layers; ++i) {
      bwd_ig_comp[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMP_NODE);
      bwd_ig_comp[num_layers - i - 1]->set_name("COMP_NODE_BWD_IG_" + to_string(num_layers - i - 1));
      bwd_ig_comp[num_layers - i - 1]->set_simulated_run_time(bwd_simulated_run_time);
      if (i == 0) {
        dg->assignDep(fwd_comm[num_layers-1], bwd_ig_comp[num_layers - i - 1]);
      }

      bwd_wg_comp[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMP_NODE);
      bwd_wg_comp[num_layers - i - 1]->set_name("COMP_NODE_BWD_WG_" + to_string(num_layers - i - 1));
      bwd_wg_comp[num_layers - i - 1]->set_simulated_run_time(bwd_simulated_run_time);

      bwd_ig_comm[num_layers - i - 1] = dg->addNode(ChakraProtoMsg::COMM_COLL_NODE);
      bwd_ig_comm[num_layers - i - 1]->set_name("COMM_COLL_NODE_BWD_IG_" + to_string(num_layers - i - 1));
      for (int j = 0; j < num_dims; ++j) {
        bwd_ig_comm[num_layers - i - 1]->add_involved_dim(true);
      }
      bwd_ig_comm[num_layers - i - 1]->set_comm_type(ChakraProtoMsg::ALL_REDUCE);
      bwd_ig_comm[num_layers - i - 1]->set_comm_size(bwd_ig_comm_size);

      dg->assignDep(bwd_ig_comp[num_layers - i - 1], bwd_wg_comp[num_layers - i - 1]);
      dg->assignDep(bwd_ig_comp[num_layers - i - 1], bwd_ig_comm[num_layers - i - 1]);

      if (i != 0) {
        dg->assignDep(bwd_wg_comp_prev, bwd_ig_comp[num_layers - i - 1]);
        dg->assignDep(bwd_ig_comm_prev, bwd_ig_comp[num_layers - i - 1]);
      }

      bwd_wg_comp_prev = bwd_wg_comp[num_layers - i - 1];
      bwd_ig_comm_prev = bwd_ig_comm[num_layers - i - 1];
    }

    dg->flushEG();
    delete dg;
  }
}

uint32_t parallelismComparisonPipelineParallel_get_tag(
    uint32_t src_npu_id, uint32_t dst_npu_id, uint32_t minibatch_id, uint32_t is_fwd)
{
  uint32_t tag = (
      ((src_npu_id & 0x3ff) << 22)
      | ((dst_npu_id & 0x3ff) << 12)
      | ((minibatch_id & 0x3ff) << 2)
      | (is_fwd & 0x3));
  return tag;
}

// GPipe
void parallelismComparisonPipelineParallel(
    int num_npus, int num_dims,
    uint64_t m, uint64_t k, uint64_t n, uint32_t data_type_size,
    uint64_t k_flops,
    uint32_t num_microbatches) {
  DependencyGraph *dg;
  Node *fwd_comp, *fwd_comp_prev, *fwd_send, *fwd_recv, *fwd_recv_prev,
            *bwd_wg_comp, *bwd_wg_comp_prev, *bwd_ig_comp, *bwd_ig_comp_prev,
            *bwd_ig_send, *bwd_ig_recv, *bwd_ig_recv_prev;

  for (int npu_id = 0; npu_id < num_npus; ++npu_id) {
    dg = new DependencyGraph(getFilename(string(__func__), npu_id));

    uint64_t fwd_simulated_run_time =
      (2 * (m / num_microbatches) * k * n) / k_flops;
    uint64_t bwd_simulated_run_time = fwd_simulated_run_time;
    uint32_t fwd_comm_size =
      (m / num_microbatches) * n * data_type_size;
    uint32_t bwd_ig_comm_size =
      (m / num_microbatches) * k * data_type_size;

    if (npu_id == 0) {
      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        fwd_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        fwd_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_comp->set_simulated_run_time(fwd_simulated_run_time);
        if (mb != 0) {
          dg->assignDep(fwd_comp_prev, fwd_comp);
        }
        fwd_comp_prev = fwd_comp;

        fwd_send = dg->addNode(ChakraProtoMsg::COMM_SEND_NODE);
        fwd_send->set_name("COMM_SEND_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_send->set_comm_src(0);
        fwd_send->set_comm_dst(1);
        fwd_send->set_comm_size(fwd_comm_size);
        fwd_send->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(0, 1, mb, 1));
        dg->assignDep(fwd_comp, fwd_send);
      }

      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        bwd_ig_recv = dg->addNode(ChakraProtoMsg::COMM_RECV_NODE);
        bwd_ig_recv->set_name("COMM_RECV_NODE_MB_" + to_string(mb) + "_BWD");
        bwd_ig_recv->set_comm_src(1);
        bwd_ig_recv->set_comm_dst(0);
        bwd_ig_recv->set_comm_size(bwd_ig_comm_size);
        bwd_ig_recv->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(1, 0, mb, 0));
        if (mb == 0) {
          dg->assignDep(fwd_send, bwd_ig_recv);
        } else {
          dg->assignDep(bwd_ig_recv_prev, bwd_ig_recv);
        }
        bwd_ig_recv_prev = bwd_ig_recv;

        bwd_wg_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_wg_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_BWD_WG");
        bwd_wg_comp->set_simulated_run_time(bwd_simulated_run_time);
        dg->assignDep(bwd_ig_recv, bwd_wg_comp);
      }
    } else if (npu_id == (num_npus - 1)) {
      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        fwd_recv = dg->addNode(ChakraProtoMsg::COMM_RECV_NODE);
        fwd_recv->set_name("COMM_RECV_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_recv->set_comm_src(num_npus - 2);
        fwd_recv->set_comm_dst(num_npus - 1);
        fwd_recv->set_comm_size(fwd_comm_size);
        fwd_recv->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            num_npus - 2, num_npus - 1, mb, 1));
        if (mb != 0) {
          dg->assignDep(fwd_recv_prev, fwd_recv);
        }
        fwd_recv_prev = fwd_recv;

        fwd_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        fwd_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_FWD_COMP");
        fwd_comp->set_simulated_run_time(fwd_simulated_run_time);
        fwd_comp_prev = fwd_comp;
        dg->assignDep(fwd_recv, fwd_comp);
      }

      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        bwd_ig_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_ig_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_BWD_IG");
        bwd_ig_comp->set_simulated_run_time(bwd_simulated_run_time);
        if (mb == 0) {
          dg->assignDep(fwd_comp, bwd_ig_comp);
        } else {
          dg->assignDep(bwd_wg_comp_prev, bwd_ig_comp);
        }

        bwd_wg_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_wg_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_BWD_WG");
        bwd_wg_comp->set_simulated_run_time(bwd_simulated_run_time);
        dg->assignDep(bwd_ig_comp, bwd_wg_comp);
        bwd_wg_comp_prev = bwd_wg_comp;

        bwd_ig_send = dg->addNode(ChakraProtoMsg::COMM_SEND_NODE);
        bwd_ig_send->set_name("COMM_SEND_NODE_MB_" + to_string(mb) + "_BWD_IG");
        bwd_ig_send->set_comm_src(num_npus - 1);
        bwd_ig_send->set_comm_dst(num_npus - 2);
        bwd_ig_send->set_comm_size(bwd_ig_comm_size);
        bwd_ig_send->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            num_npus - 1, num_npus - 2, mb, 0));
        dg->assignDep(bwd_ig_comp, bwd_ig_send);
      }
    } else {
      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        fwd_recv = dg->addNode(ChakraProtoMsg::COMM_RECV_NODE);
        fwd_recv->set_name("COMM_RECV_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_recv->set_comm_src(npu_id - 1);
        fwd_recv->set_comm_dst(npu_id);
        fwd_recv->set_comm_size(fwd_comm_size);
        fwd_recv->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            npu_id - 1, npu_id, mb, 1));
        if (mb != 0) {
          dg->assignDep(fwd_recv_prev, fwd_recv);
        }
        fwd_recv_prev = fwd_recv;

        fwd_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        fwd_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_comp->set_simulated_run_time(fwd_simulated_run_time);
        fwd_comp_prev = fwd_comp;
        dg->assignDep(fwd_recv, fwd_comp);

        fwd_send = dg->addNode(ChakraProtoMsg::COMM_SEND_NODE);
        fwd_send->set_name("COMM_SEND_NODE_MB_" + to_string(mb) + "_FWD");
        fwd_send->set_comm_src(npu_id);
        fwd_send->set_comm_dst(npu_id + 1);
        fwd_send->set_comm_size(fwd_comm_size);
        fwd_send->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            npu_id, npu_id + 1, mb, 1));
        dg->assignDep(fwd_comp, fwd_send);
      }

      for (uint32_t mb = 0; mb < num_microbatches; ++mb) {
        bwd_ig_recv = dg->addNode(ChakraProtoMsg::COMM_RECV_NODE);
        bwd_ig_recv->set_name("COMM_RECV_NODE_MB_" + to_string(mb) + "_BWD_IG");
        bwd_ig_recv->set_comm_src(npu_id + 1);
        bwd_ig_recv->set_comm_dst(npu_id);
        bwd_ig_recv->set_comm_size(bwd_ig_comm_size);
        bwd_ig_recv->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            npu_id + 1, npu_id, mb, 0));
        if (mb == 0) {
          dg->assignDep(fwd_send, bwd_ig_recv);
        } else {
          dg->assignDep(bwd_ig_recv_prev, bwd_ig_recv);
        }
        bwd_ig_recv_prev = bwd_ig_recv;

        bwd_ig_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_ig_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_BWD_IG");
        bwd_ig_comp->set_simulated_run_time(bwd_simulated_run_time);
        dg->assignDep(bwd_ig_recv, bwd_ig_comp);

        bwd_wg_comp = dg->addNode(ChakraProtoMsg::COMP_NODE);
        bwd_wg_comp->set_name("COMP_NODE_MB_" + to_string(mb) + "_BWD_WG");
        bwd_wg_comp->set_simulated_run_time(bwd_simulated_run_time);
        dg->assignDep(bwd_ig_comp, bwd_wg_comp);
        bwd_wg_comp_prev = bwd_wg_comp;

        bwd_ig_send = dg->addNode(ChakraProtoMsg::COMM_SEND_NODE);
        bwd_ig_send->set_name("COMM_SEND_NODE_MB_" + to_string(mb) + "_BWD_IG");
        bwd_ig_send->set_comm_src(npu_id);
        bwd_ig_send->set_comm_dst(npu_id - 1);
        bwd_ig_send->set_comm_size(bwd_ig_comm_size);
        bwd_ig_send->set_comm_tag(parallelismComparisonPipelineParallel_get_tag(
            npu_id, npu_id - 1, mb, 0));
        dg->assignDep(bwd_ig_comp, bwd_ig_send);
      }
    }

    dg->flushEG();
    delete dg;
  }
}
}

int main(int argc, char **argv)
{
  cxxopts::Options options("graphgen", "generates example execution graphs");

  options.add_options()
    ("num_npus", "Number of NPUs",
     cxxopts::value<int>()->default_value("64"))
    ("num_dims", "Number of dimensions in the network topology",
     cxxopts::value<int>()->default_value("2"))
    ;
  auto result = options.parse(argc, argv);
  auto num_npus = result["num_npus"].as<int>();
  auto num_dims = result["num_dims"].as<int>();

  oneCompNode(num_npus, num_dims);
  twoCompNodesIndependent(num_npus, num_dims);
  twoCompNodesDependent(num_npus, num_dims);

  oneCommNodeAllReduce(num_npus, num_dims);
  oneCommNodeAllToAll(num_npus, num_dims);
  oneCommNodeAllGather(num_npus, num_dims);
  oneCommNodeReduceScatter(num_npus, num_dims);

  for (uint32_t i = 6; i < 17; ++i) {
    commNodesSingleSendSingleRecv(num_npus, num_dims, 1 << i);
  }

  invalidNodeCase1(num_npus, num_dims);
  invalidNodeCase2(num_npus, num_dims);
  invalidNodeCase3(num_npus, num_dims);
  invalidNodeCase4(num_npus, num_dims);

  threeLayerDataParallel(num_npus, num_dims);
  threeLayerDataParallelSequentiallyDependent(num_npus, num_dims);

  // parallelism comparison
  parallelismComparisonDataParallel(num_npus, num_dims,
                                        512, 512, 512, 2, k_flops);
  parallelismComparisonModelParallel(num_npus, num_dims,
                                        512, 512, 512, 2, k_flops);
  parallelismComparisonPipelineParallel(num_npus, num_dims,
                                        512, 512, 512, 2, k_flops, 64);

  return 0;
}
