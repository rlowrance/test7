inputs += ../data/input/7chord_team_folder/NYU/7chord_ticker_universe_nyu_poc/chk.csv
targets += ../data/working/bds/chk/0counts.txt
targets += ../data/working/bds/fcx/0counts.txt
targets += ../data/working/bds/jpm/0counts.txt
targets += ../data/working/bds/ms/0counts.txt
targets += ../data/working/bds/msft/0counts.txt
targets += ../data/working/bds/noc/0counts.txt
targets += ../data/working/bds/orcl/0counts.txt
targets += ../data/working/bds/pkd/0counts.txt
targets += ../data/working/bds/rig/0counts.txt
targets += ../data/working/describe/chk.csv

.PHONY: all
all: ${targets}

# bds

../data/working/bds/chk-counts.txt: bds.py ${inputs}
	python bds.py  chk

../data/working/bds/fcx-counts.txt: bds.py ${inputs}
	python bds.py  fcx

../data/working/bds/jpm-counts.txt: bds.py ${inputs}
	python bds.py  jpm

../data/working/bds/ms-counts.txt: bds.py ${inputs}
	python bds.py  ms

../data/working/bds/msft-counts.txt: bds.py ${inputs}
	python bds.py  msft

../data/working/bds/noc-counts.txt: bds.py ${inputs}
	python bds.py  noc

../data/working/bds/orcl-counts.txt: bds.py ${inputs}
	python bds.py  orcl

../data/working/bds/pkd-counts.txt: bds.py ${inputs}
	python bds.py  pkd

../data/working/bds/rig-counts.txt: bds.py ${inputs}
	python bds.py  rig

# describe
../data/working/describe/chk.csv: describe.py ${inputs}
	python describe.py
