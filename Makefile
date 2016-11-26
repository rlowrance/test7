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

targets += ../data/maturity_split/chk/0log.txt
targets += ../data/maturity_split/fcx/0log.txt
targets += ../data/maturity_split/jpm/0log.txt
targets += ../data/maturity_split/msft/0log.txt
targets += ../data/maturity_split/noc/0log.txt
targets += ../data/maturity_split/orcl/0log.txt
targets += ../data/maturity_split/pkd/0log.txt
targets += ../data/maturity_split/rig/0log.txt

targets += ../data/working/describe/chk.csv

.PHONY: all
all: ${targets}

# bds

../data/working/bds/chk/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  chk

../data/working/bds/fcx/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  fcx

../data/working/bds/jpm/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  jpm

../data/working/bds/ms/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  ms

../data/working/bds/msft/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  msft

../data/working/bds/noc/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  noc

../data/working/bds/orcl/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  orcl

../data/working/bds/pkd/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  pkd

../data/working/bds/rig/0counts.txt: bds.py seven.py ${inputs}
	python bds.py  rig

# describe
../data/working/describe/chk.csv: describe.py ${inputs}
	python describe.py

# maturity_split
../data/working/maturity_split/chk/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py chk

../data/working/maturity_split/fcx/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py fcx

../data/working/maturity_split/jpm/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py jpm

../data/working/maturity_split/msft/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py msft

../data/working/maturity_split/noc/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py noc

../data/working/maturity_split/orcl/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py orcl

../data/working/maturity_split/pkd/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py pkd

../data/working/maturity_split/rig/0log.txt: maturity_split.py ${inputs}
	python maturity_split.py rig
