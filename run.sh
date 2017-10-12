# re-run test of one cusip
python etl.py etl_configuration.json debug.True
python events_cusip.py events_cusip_configuration.json debug.True
