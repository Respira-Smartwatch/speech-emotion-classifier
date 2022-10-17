PYTHON = $(HOME)/.pyenv/versions/respira/bin/python3
RAVDESS_SRC = zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

##########################################################################
#
#                   SETUP:  INSTALL PYTHON REQUIREMENTS
#                           DOWNLOAD RAVDESS DATASET
#
##########################################################################
setup:
	@$(PYTHON) -m pip install -r requirements.txt >/dev/null
	wget $(RAVDESS_SRC) -O dataset.zip
	unzip -q dataset.zip -d dataset/

clean:
	rm -rf dataset
