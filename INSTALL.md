# Chakra
## Installation
We use `pip` to install/uninstall the `chakra` package:
```shell
# Install package from source
$ pip install .

# Install latest from github
$ pip install https://github.com/mlcommons/chakra/archive/refs/heads/main.zip

# Install specific revision from github
$ pip install https://github.com/mlcommons/chakra/archive/ae7c671db702eb1384015bb2618dc753eed787f2.zip

# Uninstall package
$ pip uninstall chakra
```

## Execution Trace Converter (et_converter)
This tool converts execution traces into the Chakra format.
This converter supports three types of formats: ASTRA-sim text files, FlexFlow, and PyTorch.

You can use the following commands for each input type.

### ASTRA-sim Text Files
```shell
$ python -m chakra.et_converter.et_converter\
    --input_type Text\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --num_dims <num_dims>\
    --num_passes <num_passes>
```

### FlexFlow Execution Graphs
```shell
$ python -m chakra.et_converter.et_converter\
    --input_type FlexFlow\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --npu_frequency <npu_frequency>\
    --num_dims <num_dims>
```

### PyTorch Execution Graphs
```shell
$ python -m chakra.et_converter.et_converter\
    --input_type PyTorch\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_dims <num_dims>
```

## Execution Trace Generator (et_generator)
This is an execution trace generator that generates synthetic execution traces.
A user can define a new function in the generator to generate new synthetic execution traces.
You can follow the commands below to run the generator.
```shell
$ python -m chakra.et_generator.et_generator\
    --num_npus <num_npus>\
    --num_dims <num_dims>
```

## Execution Trace Visualizer (et_visualizer)
This tool visualizes a given execution trace (ET) by converting the ET to a graph in various supported formats: PDF, Graphviz (dot), or GraphML.
The output format is determined by the file extension (postfix) of the output filename.
For PDF and Graphviz formats, use ".pdf" and ".dot" extensions respectively.
For GraphML, suitable for visualizing large-scale graphs, use the ".graphml" extension. 
The PDF and Graphviz formats are generated using the Graphviz library, while the GraphML format is generated using the NetworkX library. 
Graphviz files can be visualized with a Graphviz viewer such as https://dreampuf.github.io/GraphvizOnline/.
For visualizing GraphML files, you can use Gephi (https://gephi.org/).

Run the tool with the following command:
```shell
$ python -m chakra.et_visualizer.et_visualizer\
    --input_filename <input_filename>\
    --output_filename <output_filename>
```

The input_filename is the path to the execution trace you want to visualize, and the output_filename is the name of the output file you want to create.
Remember to specify the correct extension in the output_filename to select the desired output format.

## Timeline Visualizer (timeline_visualizer)
This tool visualizes the execution timeline of a given execution trace (ET).

You can run this timeline visualizer with the following command.
```shell
$ python -m chakra.timeline_visualizer.timeline_visualizer\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --npu_frequency <npu_frequency>
```

The input file is an execution trace file in csv, and the output file is a json file.
The input file format is shown below.
```csv
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
issue,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
callback,<dummy_str>=npu_id,<dummy_str>=curr_cycle,<dummy_str>=node_id,<dummy_str>=node_name
...
```
As this tool requires an execution trace of an ET, a simulator has to print out execution traces.
The output json file is chrome-tracing-compatible.
When you open the file with `chrome://tracing`, you will see an execution timeline like the one below.
![](doc/timeline_visualizer.png)

## Execution Trace Feeder (et_feeder)
This is a trace feeder that feeds dependency-free nodes to a simulator.
Therefore, a simulator has to import this feeder as a library.
Currently, ASTRA-sim is the only simulator that supports the trace feeder.
You can run execution traces on ASTRA-sim with the following commands.
```
$ git clone --recurse-submodules git@github.com:astra-sim/astra-sim.git
$ cd astra-sim
$ git checkout Chakra
$ git submodule update --init --recursive
$ cd extern/graph_frontend/chakra/
$ git checkout main
$ cd -
$ ./build/astra_analytical/build.sh -c

$ cd extern/graph_frontend/chakra/
$ python -m chakra.et_generator.et_generator\
    --num_npus <num_npus>\
    --num_dims <num_dims>

$ cd -
$ ./run.sh
```

## Execution Trace Jsonizer (et_jsonizer)
This tool prints the nodes within execution traces for better comprehension.
The printed information includes the node's id, name, type, and any associated metadata, which are all outputted in a user-friendly text format.
```
$ python -m chakra.et_jsonizer.et_jsonizer\
    --input_filename <input_filename>\
    --output_filename <output_filename>
```
