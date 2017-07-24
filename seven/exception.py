class BuildException(Exception):
    'problem in determining how to build the data; raise by build.py'
    def __init__(self, msg):
        super(BuildException, self).__init__(msg)
        self.msg = msg
        print 'raising BuildException:', msg
