#!/usr/bin/env python

from os.path import expanduser
home = expanduser("~")

host = False
with open(home + "/.ssh/config", "ab+") as f:
	for line in f:
		if line == "Host 10.0.0.*\n":
			host = True
	if not host:
		f.write("Host 10.0.0.*\n")
		f.write("\tStrictHostKeyChecking no\n")
		f.write("\tUserKnownHostsFile=/dev/null\n")
		f.write("\tLogLevel QUIET")

