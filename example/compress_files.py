import glob
import subprocess
import sys
from multiprocessing import Pool

FILES_DIR = '/tmp/credo/parts'
CORES = 4


def run_file(fn):
    print('Start compressing: %s' % fn)
    sp = subprocess.Popen(["xz", "-9e", fn])
    sp.wait()
    print('... finish compressing: %s' % fn)


def main():
    # list all files in INPUT_DIR
    files = glob.glob('%s/*.json' % FILES_DIR)
    with Pool(CORES) as pool:
        # each file parsed separately
        pool.map(run_file, files)


if __name__ == '__main__':
    main()
    sys.exit(0)  # not always close
