# Chakra
## Installation
We use `setuptools` to install/uninstall the `chakra` package:
```shell
# Install package
$ python setup.py install

# Uninstall package
$ python -m pip uninstall chakra
```

## Execution Graph Converter (eg_converter)
This tool converts execution graphs into the Chakra format.
This converter supports three types of formats: ASTRA-sim text files, FlexFlow, and PyTorch.

You can use the following commands for each input type.

### ASTRA-sim Text Files
```shell
$ python -m eg_converter.eg_converter\
    --input_type Text\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --num_npus <num_npus>\
    --num_dims <num_dims>\
    --num_passes <num_passes>
```

### FlexFlow Execution Graphs
```shell
$ python -m eg_converter.eg_converter\
    --input_type FlexFlow\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --npu_frequency <npu_frequency>\
    --num_dims <num_dims>
```

### PyTorch Execution Graphs
```shell
$ python -m eg_converter.eg_converter\
    --input_type PyTorch\
    --input_filename <input_filename>\
    --output_filename <output_filename>\
    --default_simulated_run_time <default_simulated_run_time>\
    --num_dims <num_dims>
```
