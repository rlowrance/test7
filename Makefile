inputs += ../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/chk.csv
targets += ../data/working/bds/chk-counts.txt
targets += ../data/working/bds/fcx-counts.txt
targets += ../data/working/bds/jpm-counts.txt
targets += ../data/working/bds/ms-counts.txt
targets += ../data/working/bds/msft-counts.txt
targets += ../data/working/bds/noc-counts.txt
targets += ../data/working/bds/orcl-counts.txt
targets += ../data/working/bds/pkd-counts.txt
targets += ../data/working/bds/rig-counts.txt
targets += ../data/working/describe/chk.csv

.PHONY: all
all: ${targets}

# bds

../data/working/bds/chk-counts.txt: bds.py ${inputs}
	python bds.py --ticker chk

../data/working/bds/fcx-counts.txt: bds.py ${inputs}
	python bds.py --ticker fcx

../data/working/bds/jpm-counts.txt: bds.py ${inputs}
	python bds.py --ticker jpm

../data/working/bds/ms-counts.txt: bds.py ${inputs}
	python bds.py --ticker ms

../data/working/bds/msft-counts.txt: bds.py ${inputs}
	python bds.py --ticker msft

../data/working/bds/noc-counts.txt: bds.py ${inputs}
	python bds.py --ticker noc

../data/working/bds/orcl-counts.txt: bds.py ${inputs}
	python bds.py --ticker orcl

../data/working/bds/pkd-counts.txt: bds.py ${inputs}
	python bds.py --ticker pkd

../data/working/bds/rig-counts.txt: bds.py ${inputs}
	python bds.py --ticker rig

# describe
../data/working/describe/chk.csv: describe.py ${inputs}
	python describe.py
