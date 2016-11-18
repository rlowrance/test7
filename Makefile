inputs += ../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/chk.csv
targets += ../data/working/bds/chk.txt
targets += ../data/working/describe/chk.csv

.PHONY: all
all: ${targets}

../data/working/bds/chk.txt: bds.py ${inputs}
	python pdb.py

../data/working/describe/chk.csv: describe.py ${inputs}
	python describe.py
