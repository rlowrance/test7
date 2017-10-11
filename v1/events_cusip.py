'''process messages in queue events.{cusip}'''
# documentation is in events_cusip.org
import argparse
import pdb
import pika
import sys

import Configuration  # reading and setting
import messages  # all the message formats


def main(argv):
    args = parse_arguments(argv)
    config = Configuration.Configuration.from_path(args.config)
    config["events_cusip"]["cusip"] = args.cusip

if __name__== '__main__':
    if False:
        pdb
    main(sys.argv)
