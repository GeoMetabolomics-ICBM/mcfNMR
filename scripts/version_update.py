#! /usr/bin/python3

import sys
from mcfnmr.utils.system import get_mcfnmr_home


def replace_version(file, version_str, newv):
	# Replaces line
	#   <version_str> = <oldv>
	# by
	#   <version_str> = <newv>
	# in file.
	with open(file, "r") as f:
		ll = f.readlines()
	
	for i in range(len(ll)):
		ix = ll[i].find(version_str + " = ")
		if ix != -1:
			oldl = ll[i]
			beg = oldl.split("=")[0].strip()
			print(f"Found\n  '{oldl[:-1]}'\nin '{file}'")
			newl = " = ".join([beg, '"'+newv+'"'])
			print(f"\nReplacing with\n  '{newl}'\n")
			ll[i] = newl + "\n"
	
	with open(file, "w") as f:
		f.writelines(ll)

	print(f"Done updating '{file}'")

	

if __name__=="__main__":
	home = get_mcfnmr_home()
	if len(sys.argv) != 2:
		print("version_update() takes exactly argument: the new version.")
		sys.exit(1)
	newv = sys.argv[1]
	print(f"New version: {newv}")

	# Update version in pyproject.toml
	fn = home / "pyproject.toml"
	replace_version(fn, "version", newv)

	# Update version in mcfnmr.__init__.py
	fn = home / "mcfnmr" / "__init__.py"
	replace_version(fn, "__version__", newv)







