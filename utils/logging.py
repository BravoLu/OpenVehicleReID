from __future__ import absolute_import 
import os
import sys
import time 

from .osutils import mkdir_if_missing

class Logger(object):
    def __init__(self, cfg):
        self.console = sys.stdout
        
        mkdir_if_missing(os.path.join('logs', cfg['LOG_DIR']))
        log = time.strftime("%Y_%m_%d_%H_%M.log", time.localtime())
        print(log)
        self.file = open(os.path.join('logs', cfg['LOG_DIR'], log), 'w')
        self.write('Configs Setting: \n')
        for key,value in cfg.items():
            self.write("{} : {}\n".format(key.ljust(20), str(value)))

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


