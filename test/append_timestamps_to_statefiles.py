import csv
import sys

# Read the meta files
load_dir = sys.argv[1] + '/motion0/'
meta_file = load_dir + 'trackerdata_meta.txt'
ids, stamps = [], []
with open(meta_file, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    labels = spamreader.next()
    for row in spamreader:
        ids.append(int(row[0]))
        stamps.append(float(row[1]) + 1e-9 * float(row[2]))

# Append time stamp to state file
def append_timestamp_to_state_file(filename, timestamp):
    with open(filename, 'a') as statefile:
        statefile.write("{} \n".format(timestamp))

# Append timestamps to all state files
init_stamp = stamps[0]
for k in xrange(len(stamps)):
    state_file = load_dir + 'state' + str(ids[k]) + '.txt'
    append_timestamp_to_state_file(state_file, stamps[k] - init_stamp)





