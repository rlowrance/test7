# invocations:
#   scons -f sconstruct.py /
#   scons -n -f sconstruct.py /
#   scons --debug=explain -f sconstruct.py /
#   scons -j <nprocesses> -f sconstruct.py /

# where / means to build everything (not just stuff in the current working directory .)

# examples:
# cons -n -f sconstruct-features_targets.py /
# cons -f sconstruct-master.py / run_id=[alien|dell|mac|testt]
# cons -f sconstruct-master.py / cusip=68389XAC9
# cons -f sconstruct-master.py / just_reports

import collections
import datetime
import os
import pdb
import pprint

import seven.build
import seven.GetBuildInfo

pp = pprint.pprint
pdb

dir_home = os.path.join('C:', r'\Users', 'roylo')
dir_dropbox = os.path.join(dir_home, 'Dropbox')
dir_working = os.path.join(dir_dropbox, 'data', '7chord', '7chord-01', 'working')
dir_midpredictor_data = os.path.join(dir_dropbox, 'MidPredictor', 'data')

env = Environment(
    ENV=os.environ,
)

env.Decider('MD5-timestamp')  # if timestamp out of date, examine MD5 checksum


def command(*args):
    make_paths = args[0]
    other_args = args[1:]
    scons = seven.build.make_scons(make_paths(*other_args))
    env.Command(
        scons['targets'],
        scons['sources'],
        scons['commands'],
    )


def commands_for():
    'call command(*args) for appropriate targets'
    # for now, just work with one CUSIP and hence one issuer
    issuers = ('AAPL',)
    issuer_cusips = {
        'AAPL': ('037833AG5',),
    }
    feature_dates = ('2017-06-26', '2017-06-27')
    fit_dates = feature_dates[:-1]  # don't fit the last date on which we create features
    hpset = 'grid4'

    # buildinfo.py
    for issuer in ('AAPL',):
        command(seven.build.buildinfo, issuer)

    # features_targets.py
    for issuer, cusips in issuer_cusips.iteritems():
        for cusip in cusips:
            for feature_date in feature_dates:
                command(seven.build.features_targets, issuer, cusip, feature_date)

    # fit.py
    for issuer in issuers:
        gbi = seven.GetBuildInfo.GetBuildInfo(issuer)
        cusips_for_build = set(issuer_cusips[issuer])
        for fit_date in fit_dates:
            # determine issuepriceids for the cusip on the fit_date
            issuepriceids = gbi.get_issuepriceids(fit_date)
            cusips = map(lambda x: gbi.get_cusip(x), issuepriceids)
            relevant_issuepriceids_cusips = filter(lambda x: x[1] in cusips_for_build, zip(issuepriceids, cusips))
            for issuepriceid_cusip in relevant_issuepriceids_cusips:
                issuepriceid, cusip = issuepriceid_cusip
                command(seven.build.fit, issuer, cusip, str(issuepriceid), hpset)


# main program
commands_for()
