echo 'Running dataset download script'

mkdir -p traces
cd traces

pip3 install gdown charset_normalizer chardet
gdown --id 1lz6VCqQ-n5lSyshH0XKSqdynKOVRqGZs -O nemo-chakra-mixtral-8x7B-traces.zip
tar -xzvf nemo-chakra-mixtral-8x7B-traces.zip