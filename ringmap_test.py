from ringmap_statistics import *
import timeit

start = timeit.default_timer()

data_folder = RingMap.data_folder
# Make a list of the available CSDs
days = [os.path.split(fl)[-1] for fl in glob(data_folder + '*') if os.path.split(fl)[-1].isdigit()]
day = days[-1]

data = h5py.File(f'{RingMap.data_folder}{day}/ringmap_intercyl_lsd_{day}.h5', 'r')

stop = timeit.default_timer()
print('Time: ', stop - start)  
