# Get folder names & data statistics for all sub-directories
import os
import data
#load_dir = '/media/barun/Seagate Backup Plus Drive/se3-nets-data/learn_physics_models/data/singlebox_fixsz_visbullet_2016_04_20_20_55_18/'
#load_dir = '/media/barun/Seagate Backup Plus Drive/se3-nets-data/learn_physics_models/data/rand1to3boxes_randszbig_visbullet_fixforce_2016_05_15_22_22_50/'
load_dir = '/media/barun/dot-matrix-ext1/catkin_ws/src/learn_physics_models/data/singlebox_fixsz_visbullet_2016_04_20_20_55_18/'
with open(load_dir + '/postprocessstats.txt', 'w') as pfile:
    dirs = os.listdir(load_dir)
    dirnames, numdata = [], []
    for k in xrange(len(dirs)):
        if k%100 == 0:
            print('{}/{}'.format(k+1, len(dirs)))
        dir = 'bag{}'.format(k)
        path = os.path.join(load_dir, dir) + '/'
        if (os.path.isdir(path)):
            # Get number of images in the folder
            eventsfile   = os.path.join(path, 'events.txt')
            occstatsfile = os.path.join(path, 'occstats.txt')
            assert (os.path.exists(eventsfile) and os.path.exists(occstatsfile))
            # Get end index (find earliest end id avoiding box falling off table & multiple box collisions)
            nexamples = sum(1 for line in open(occstatsfile)) # By default num examples
            events = data.read_events_file(eventsfile)
            if 'BOX_FALL' in events:
                nexamples = min(nexamples, events['BOX_FALL'])
            if 'MULTI_BOX_COLL' in events:
                nexamples = min(nexamples, events['MULTI_BOX_COLL'])
            invalid = (nexamples == 0)
            pfile.write('{} {} {} {} {} {}\n'.format(int(invalid), int(0), int(0), int(nexamples), float(1.0), dir))
            # Add to stats
            dirnames.append(dir)
            numdata.append(int(nexamples))  # We only have flows for these many images!
    print('Found {} motions ({} examples) in dataset: {}'.format(len(numdata), sum(numdata), load_dir))
