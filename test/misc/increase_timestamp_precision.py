import sys, os
import numpy as np
import time

def main():

    def fix_time(data):
        return (data[:, 1] + 1e-9 * data[:, 2]) - data[0, 1]

    root = '/raid/barun/data/baxter/realdata/'
    sessions = ['session_'+n+'/motion0/' for n in ['2017-9-7_233102',
                                                   '2017-9-8_02519',
                                                   '2017-9-8_111150',
                                                   '2017-9-8_134717',
                                                   '2017-9-8_152749']]

    """
    # compare state timestamps and meta file timestamps - they are the same
    state_times = []
    for i in range(10):
        with open(root + session + 'state' + str(i) + '.txt','r') as f:
            state_times.append(f.readlines()[-1])

    for mt, st in zip(meta_times, state_times):
        print mt-meta_times[0], st
    """

    ## Update dataset timestamps
    for session in sessions:
        # Read the meta
        print('Updating dataset: ' + session)
        datafol = root + session
        meta = np.loadtxt(datafol + 'trackerdata_meta.txt', skiprows=1)
        meta_times = fix_time(meta)
        meta_times -= meta_times[0]  # shift meta times so initial time is 0

        # Replace state timestamps with meta timestamps - meta have a higher precision
        for i, t in enumerate(meta_times):
            with open(datafol + 'state' + str(i) + '.txt','r') as f:
                lines = f.readlines()
            with open(datafol + 'state' + str(i) + '.txt', 'w') as f:
                f.writelines(lines[:-1] + [str(t)])
            if (i % 10000 == 0):
                print('Frame: {}/{}'.format(i, len(meta_times)))

################ RUN MAIN
if __name__ == '__main__':
    main()
