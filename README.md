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
