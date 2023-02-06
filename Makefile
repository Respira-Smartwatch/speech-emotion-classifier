# Author: Joseph Bellahcen <joeclb@icloud.com>

setup:
	sudo apt update
	sudo apt install -y python3-scipy llvm llvm-dev
	pip3 install -r requirements.txt

.PHONY: setup
