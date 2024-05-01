cp /workspace/uhd/host/utils/uhd-usrp.rules /etc/udev/rules.d/
/lib/systemd/systemd-udevd --daemon
udevadm control --reload-rules
udevadm trigger
lsusb

