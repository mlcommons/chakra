#!/bin/bash

set -e -x

input_type=PyTorch
model_type=dp2_small_2408

for i in {0..1}
do
    # echo $kineto_file
    kineto_file=`ls result/$i`
    cp result/$i/${kineto_file} kineto_${i}.pt.trace.json
    chakra_trace_link --rank ${i} --chakra-host-trace pytorch_et_${i}.json --chakra-device-trace kineto_${i}.pt.trace.json --output-file pytorch_et_${i}_plus.json
    chakra_converter $input_type --input pytorch_et_${i}_plus.json --output ${model_type}.${i}.et
    chakra_jsonizer --input_filename ${model_type}.${i}.et --output_filename ${model_type}.${i}.json
done
chakra_pg_extractor --input_filename ./${model_type} --output_filename ${model_type}.json
rm -rf pytorch_et_*
rm -rf result
rm -rf kineto_*
mkdir et_${model_type}
mv ${model_type}.* et_${model_type}/