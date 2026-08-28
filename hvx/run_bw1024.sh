#!/bin/bash
source /home/kyle/hexagon/env.sh >/dev/null 2>&1
cd /home/kyle/Documents/GitHub/sgm-bench/hvx
for arm in 0 1 2; do
  echo -n "arm$arm "
  hexagon-sim -mv73na_1 --timing --simulated_returnval ./b_1024 -- $arm 2>&1 | grep -oE "Pcycles=[0-9]+|naive=OK rolling=OK"
done
