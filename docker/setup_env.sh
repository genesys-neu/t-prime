for l in `ls -d /opt/hpcx/*/lib/`; do export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$l"; done
for b in `ls -d /opt/hpcx/*/bin/`; do export PATH="$PATH:$b"; done


