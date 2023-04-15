import os
import sys

DATA_PATH = ".\data\\"

def main(argv):
    if len(argv) == 0:
        printUsage()
    elif argv[0] == "clean":
        partial_path = DATA_PATH + argv[1]
        os.system("del {}*".format(partial_path))

def printUsage():
    pass

if __name__ == "__main__":
    main(sys.argv[1:])