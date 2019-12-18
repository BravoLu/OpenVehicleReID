from __future__ import absolute_import 
import os
import sys
from .osutils import mkdir_if_missing

class Logger(object):
    def __init__(self, cfg):
        self.console = sys.stdout
        
        mkdir_if_missing(os.path.join('logs', cfg['NAME']))
        self.file = open(os.path.join('logs', cfg['NAME'], cfg['LOGS']), 'w')
        self.write('Config Setting: \n')
        for key,value in cfg.items():
            self.write("{}: {}\n".format(key, value))

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()
    
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


