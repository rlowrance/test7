# re-run test of one cusip
python etl.py          config_test_files_one.json
python events_cusip.py config_test_files_one.json
