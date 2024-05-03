# Installation of T-PRIME with Docker
## Install Docker with GPU support
### Prerequisites (Ubuntu)
Install NVIDIA Drivers and CUDA Toolkit following these [instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu).
### Install Docker with CUDA support
First [install Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt). Then [configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) Docker in order to get access to GPU.
## Build T-PRIME docker image
Navigate to the root folder of T-PRIME repository and run the following commands to build the Docker image for T-PRIME:
```
cd docker/
sudo docker build -t <user>/t-prime-rx .
```
## Setup static IP interface for USRPs (only for USRP with Ethernet connection)
In order to communicate via Ethernet with the USRP, we need to run the following command:
```
bash docker/setup_x310s_default.sh --device <NIC_ID>:192.168.XX.1
```
for a full usage of USRP setup script, see help message:
```
$ bash docker/setup_x310s_default.sh --help
Usage: bash setup_x310s_default.sh --device "interface1:ipaddr1[:uhd_fpga_image1],interface2:ipaddr2[:uhd_fpga_image2],..." [OPTIONS]

Each device must be initialized by providing the following info:
  - interface: name of ethernet interface where the SDR is connected;
  - ip address: IP address to be assigned to the Eth port. Note: The code assumes the Eth port IP ending in xxx.xxx.xxx.1 and the USRP IP ending in xxx.xxx.xxx.2;
  - uhd_fpga_image: type of UHD image to be installed to FPGA (default is HG). This value is only required with --image_dl option enabled.
 Example: "enp7s0f0:192.168.40.1,enp7s0f1:192.168.50.1,enp7s0f2:192.168.60.1"

OPTIONS includes:
   -i | --image_dl - download the FPGA images compatible with current UHD driver. Use in case of image version mismatch error.
```
## Run T-PRIME docker image
From the root of the repository, run the following command to start the T-PRIME Docker container
```
sudo docker run -ti --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host --privileged -v `pwd`:/root/t-prime <user>/t-prime-rx
```
Once the docker container has started, run the following command to resolve the missing libraries (needed for `nvcr.io/nvidia/pytorch:24.04-py3` image):
```
cd ~/t-prime
source docker/setup_env.sh
```
If USRP is connected via USB (i.e., B200, B210), run the following command to enable USB connection:
```
sh docker/setup_usb.sh
```
Test visibility of USRP device with following command:
```
$ uhd_find_devices 
[INFO] [UHD] linux; GNU C++ version 11.4.0; Boost_107400; DPDK_21.11; UHD_4.5.0.HEAD-0-g471af98f
--------------------------------------------------
-- UHD Device 0
--------------------------------------------------
Device Address:
    serial: 30AF824
    name: 
    product: B205mini
    type: b200
```
If `uhd_find_devices` outputs an error, restarting the docker and repeating the commands above seem to solve the issue.
To run the receiver script, use the following command within the docker:
```
python Tprime_USRP_run.py -fq 2.427e9 -t 180 --model_path TPrime_transformer/model_cp/model_lg_otaglobal_inf_RMSn_bckg_ft.pt --model_size lg --RMSNorm --rx_type b200
```
