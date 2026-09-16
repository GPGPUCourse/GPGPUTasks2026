#!/bin/bash

# exit script on failure
set -euo pipefail
set -v

# ask sudo password
echo "========================================"
echo "This script requires sudo privileges."
echo "Please enter your sudo password below."
echo "Note: nothing will appear while typing the password."
echo "========================================"

# Ask for sudo password upfront and cache credentials.
sudo -v

echo
echo "sudo authentication successful."
echo


# 1) Add Intel oneAPI APT repo + key  (Ubuntu 24.04 Noble)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update

# 2) Install Intel OpenCL runtime for CPU
sudo apt install -yq intel-oneapi-runtime-opencl   # OpenCL runtime (includes ICD loader setup)

# 3) Verify devices
sudo apt install -yq clinfo
clinfo -l  # should show: Platform "Intel(R) CPU Runtime for OpenCL(TM) Applications" and your CPU

# 4) install cmake
sudo apt install cmake -y