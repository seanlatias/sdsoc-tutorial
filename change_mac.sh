#!/usr/bin/env python

import sys, os, re

if len(sys.argv) <= 1:
  sys.exit("ERROR: Usage: change_mac.sh MAC_ADDRESS")
  
f0 = open("BOOT.BIN", "r");
f1 = open("BOOT.BIN.new", "w");

for line in f0:
  f1.write(re.sub(r"..:..:..:..:..:..", sys.argv[1], line))

f0.close()
f1.close()

os.system("mv BOOT.BIN BOOT.BIN.old; mv BOOT.BIN.new BOOT.BIN")
