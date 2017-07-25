class BuildException(Exception):
    'problem in determining how to build the data; raise by build.py'
    def __init__(self, msg):
        super(BuildException, self).__init__(msg)
        self.msg = msg
        print 'raising BuildException:', msg


class CriticalException(Exception):
    'raised by seven.logging.error, if not debugging'
    def __init__(self, msg):
        super(CriticalException, self).__init__(msg)
        self.msg = msg
        print 'raising BuildException:', msg


class ErrorException(Exception):
    'raised by seven.logging.error, if not debugging'
    def __init__(self, msg):
        super(ErrorException, self).__init__(msg)
        self.msg = msg
        print 'raising BuildException:', msg
