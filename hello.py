# -*- coding: utf-8 -*-

import datetime
import sys

def main():
    print("** Hello world! ** ")
    print("** Python version ** ")
    print("%s" % sys.version)
    print("** Current time **")
    print("%s" % datetime.datetime.now())    

if __name__ == '__main__':
    main()
