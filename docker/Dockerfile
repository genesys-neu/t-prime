FROM nvcr.io/nvidia/pytorch:24.04-py3
LABEL MAINTAINER="Mauro Belgiovine <belgiovine.m@northeastern.edu>"
# update package list
RUN apt update
# install Linux utilities, including build toolchain
RUN apt install -y build-essential ccache clang clang-format-14 cmake cmake-curses-gui cpufrequtils curl ethtool git inetutils-tools nano wget
# install UHD dependencies
RUN apt install -y doxygen dpdk libboost-all-dev libdpdk-dev libgps-dev libudev-dev libusb-1.0-0-dev python3-dev python3-docutils python3-mako python3-numpy python3-pip python3-requests python3-ruamel.yaml
# create a directory to clone the UHD repository into
RUN cd $HOME && mkdir workarea && cd workarea
# clone the UHD repository and checkout the latest gged release
RUN git clone https://github.com/EttusResearch/uhd.git uhd && cd uhd && git checkout v4.5.0.0 && cd host && mkdir build && cd build && cmake .. && make -j `nproc` && make test && make install
# setup the library references
RUN ldconfig
# install net-tools (needed to look at the ethernet card configuration)
RUN apt install -y net-tools

#install SoapySDR
RUN apt install -y cmake g++ libpython3-dev python3-numpy swig
RUN cd $HOME/workarea && git clone https://github.com/pothosware/SoapySDR.git && cd SoapySDR && git pull origin master && mkdir build && cd build && cmake .. && make -j `nproc` && make install -j`nproc`
RUN ldconfig

#install SoapyUHD
RUN cd $HOME/workarea && git clone https://github.com/pothosware/SoapyUHD.git && cd SoapyUHD && mkdir build && cd build && cmake .. && make -j `nproc` && make install -j`nproc`
# missing dependency for T-PRIME
RUN pip install pycuda
RUN apt install -y usbutils

RUN uhd_images_downloader
