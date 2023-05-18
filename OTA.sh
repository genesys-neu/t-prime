#!/bin/bash


for protocol in 802.11b 802.11g 802.11n 802.11ax
do
  file_name="${protocol}_IQ_frame_2"
  echo "Starting to transmit ${protocol}"
  python3 rf_replay_data_transmitter_usrp_uhd.py --args="type=x300,addr=192.168.40.14,master_clock_rate=184.32e6" --freq=2.427e9 --rate=20e6 --gain=30 --path="waveforms/DSTL/1_1/" --file=$file_name --waveform_format="matlab" &
  pid=$!
  sleep 25
  kill -INT $pid &
  wait
  clear
  
done
