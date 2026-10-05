#!/bin/bash

# exit script on failure
set -ev

CUDA_RUNFILE=cuda_13.3.1_610.43.02_linux.run
wget https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/${CUDA_RUNFILE}
sudo apt install linux-headers-$(uname -r)
chmod +x ${CUDA_RUNFILE}
sudo ./${CUDA_RUNFILE} --silent --toolkit
# permamently add NVCC to PATH:
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
echo "export PATH=\$PATH:\$CUDA_HOME/bin" >> ~/.bashrc
