# MLSys 2026 MLCommons Chakra Artifact Evaluation

## Install/Set up Chakra

### Create python virtual environment for Chakra
```bash
# Create a virtual environment in the path/to/chakra/ 
$ python3 -m venv chakra_env

# Activate the virtual environment
$ source chakra_env/bin/activate
```

## Install Chakra and Convert NeMo Traces to Chakra .et

### Install Chakra
```bash
source chakra_env/bin/activate
pip install .
```

### Pin protobuf version
> **Critical:** The protobuf version used to **generate** the `.et` traces must match the
> version compiled into the ASTRA-sim Docker image. The Dockerfile builds **protobuf 6.33.0**.
> Pin your Chakra environment to the same version before converting traces.
```bash
pip install protobuf==6.33.0
```

### Install PARAM (required by `chakra_trace_link`)
`chakra_trace_link` depends on `et_replay` from the [PARAM](https://github.com/facebookresearch/param) project.
```bash
git clone https://github.com/facebookresearch/param.git
cd param/et_replay
git checkout 7b19f586dd8b267333114992833a0d7e0d601630
pip install .
cd ../..
```

### Download traces
```bash
cd mlsys26
bash download_nemo_chakra_traces.sh
```

### Convert traces (trace link + converter in one step)
```bash
bash convert_traces.sh
```

Outputs are written to:
- `mlsys26/traces/linked/`  — linked JSON (host + device merged per rank)
- `mlsys26/traces/et/`      — protobuf `.et` files ready for ASTRA-sim

## Using ASTRA-sim for Chakra-Based Simulation of Diverse Networked Systems

ASTRA-sim leverages Chakra’s ET feeder to replace its original custom workload format. This integration has enabled a range of co-design studies on emerging platforms, particularly for exploring and optimizing networking infrastructures.

### ASTRA-sim Installation
```bash
# # Inside the .../mlsys26 directory, clone astra-sim and pin to the validated commit for this artifact
git clone git@github.com:astra-sim/astra-sim.git


cd ./astra-sim
git checkout changhai/chakra_main_paper
git submodule update --init --recursive
cd ..
```

```bash
# Run from the mlsys26/ directory

# Align the protobuf versions through the following patch
bash astra-sim-patch.sh ./astra-sim/Dockerfile

# Remove any old container and image first, if any (full clean rebuild)
docker rm -f astra-sim-mlsys26 2>/dev/null || true
docker rmi -f astra-sim:mlsys26 2>/dev/null || true

# Build Docker image
docker build -t astra-sim:mlsys26 -f ./astra-sim/Dockerfile ./astra-sim

# Run container with bind mounts:
#   /app/astra-sim            <- astra-sim source + build output
#   /app/astra-sim/mlsys26/plots  <- run scripts and configs
#   /traces                   <- .et trace files
docker run -it --name astra-sim-mlsys26 --shm-size=8g \
    -v "$(pwd)/astra-sim:/app/astra-sim" \
    -v "$(pwd)/plots:/app/astra-sim/mlsys26/plots" \
    -v "$(pwd)/traces/et:/traces" \
    astra-sim:mlsys26 bash
```

### Build ASTRA-sim inside the container
```bash
# Inside the container:
cd /app/astra-sim
./build/astra_analytical/build.sh
```


### Final Step (with Astra-Sim and Chakra all in place) - Run the simulation
```bash
# Inside the container (after building):
bash /app/astra-sim/mlsys26/plots/m8x7/mixtral_8x7b.sh
```

### Draw the plots (Fig. 6,7,8,12)
```bash
# Assume going back to the path/to/chakra/mlsys26 and with chakra_env activated
# Go to plots directory
cd plots

# install matplotlib for plotting
$ pip install matplotlib 

# Figure 6
python chakra_kineto_reconstruct.py

# Figure 7
python plot_coll_ib.py

# Figure 8
bash run_plot_memory.sh

# Figure 12
cd ./m8x7/
python plot_astra-sim_bw_analysis.py

# Cleanup the results logs in the directory generated (Optional)
cd /app/astra-sim/mlsys26/plots/m8x7/
find . -maxdepth 1 -type d ! -name . -exec rm -rf {} +
```

