# System utility packages
import sys, os
from glob import glob
import pickle
import h5py
from datetime import datetime, timedelta
import warnings

# Base science packages
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from skimage.transform import resize

# Plotting 
import matplotlib.pyplot as plt
import matplotlib

# chime packages
from ch_util import ephemeris
from ch_pipeline.analysis import flagging

### CHANGE THESE TO LOCAL DIRECTORIES ###
output_directory = '/home/yuhants/rev03/'
plots_directory = f'{output_directory}plots/'

# Load in planet positions for flagging
# This is mostly used to remove times when the moon is up
# I'm honestly not sure how this works, so more investigation might be helpful
planets = ephemeris.skyfield_wrapper.load("de421.bsp")

# This makes the plots bigger and more readable, but can be removed or changed
matplotlib.rcParams.update({'font.size': 40})
matplotlib.rcParams['figure.figsize'] = (30.0, 20.0)

def plot_ringmap(ringmap, title = 'Ringmap'):
    """
    Do a basic plot of the given (single frequency) Ringmap.
    Does NOT show or save the plot, that should be done separately.
    """
    plt.figure(figsize = (30,20))
    matplotlib.rcParams.update({'font.size': 40})
    plt.imshow(ringmap, aspect = 'auto', vmin=-1,vmax=1,origin='lower', cmap = 'RdBu')
    plt.plot(RingMap.cyga_pos[0], RingMap.cyga_pos[1], 'yo', markersize = 20, label = 'Cyg A')
    plt.plot(RingMap.casa_pos[0], RingMap.casa_pos[1], 'go', markersize = 20, label = 'Cas A')
    plt.colorbar()
    plt.legend()
    plt.title(title)
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')

def plot_ringmap_histogram(data, title = "Histogram", no_outliers = False):
    """
    Plot a basic histogram of the given data. 
    Does NOT show or save the plot, that should be done separately.
    
    Parameters
    ----------
    data : 1-d numpy array 
    title : str (optional)
    no_outliers : bool (optional)
        If True, plot the histogram zoomed in within 3-sigma
    """

    plt.figure(figsize = (8,6))
    matplotlib.rcParams.update({'font.size': 12})
    if no_outliers:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        lo, hi = mean - 3 * std, mean + 3 * std
        plt.hist(data, bins = 'fd', histtype = 'step', range = (lo, hi))
    else:
        plt.hist(data, bins = 'fd', histtype = 'step')
    plt.title(title)
    plt.xlabel('Ringmap Data Value')
    plt.ylabel('Counts')

class RingMap:
    """
    A class to collate ringmap data and processing.
    NOTE: Ringmap units are in Jy/beam
    """

    # Change these to whatever revision is being used
    data_folder = '/project/rpp-krs/chime/chime_processed/daily/rev_03/'
    template_file = '/project/rpp-krs/chime/chime_processed/stacks/rev_03/all/ringmap_intercyl.h5'

    # Pickle where the template should be saved locally
    # Feel free to change this
    template_pickle = '/home/yuhants/scratch/template/rev_03_template.pickle'
    
    # The RA positions of Cygnus and Cassiopeia A, the brightest radio sources
    cyga_pos = (3993., 298.)
    casa_pos = (3415., 217.)

    def __init__(self, day = None, intercyl = True, median_subtract = False):
        """
        Initialize the ringmap. Day should be a valid CSD, or None if you just want the template. 
        Median subtraction can be done immediately to remove crosstalk, or later after some 
        masking has already been done (see get_masked_map). 
        """
        if day:
            # If we're given a day, load in the relevant ringmap
            self.day = int(day)
            if not intercyl:
                self.data = h5py.File(f'{self.data_folder}{self.day}/ringmap_lsd_{day}.h5', 'r')
            else:
                self.data = h5py.File(f'{self.data_folder}{self.day}/ringmap_intercyl_lsd_{day}.h5', 'r')
            self.ringmap = np.transpose(self.data['map'][0,0], (0,2,1))
            self.title = f"CSD {self.day} Map"
        else:
            # Otherwise make a template map
            self.day = 0
            self.data = h5py.File(self.template_file, 'r')

            # Since the template is huge and the same every time, we can save a local copy 
            # and load that rather than loading and transforming the base file.
            if os.path.exists(self.template_pickle):
                with open(self.template_pickle, 'rb') as fl:
                    self.ringmap = pickle.load(fl)
            else:
                # If the pickle isn't available, load in the true template and then save a local (resized) copy
                self.ringmap = np.transpose(self.data['map'][0,0], (0,2,1)) 
                self.ringmap = resize(self.ringmap, (1024, 512, 4096), anti_aliasing = False)
                with open(self.template_pickle, 'wb') as fl:
                    pickle.dump(self.ringmap, fl)

            self.title = f"Template Map"
        
        if median_subtract:
            # Median subtract along rows
            self.ringmap -= np.median(self.ringmap, axis = 2, keepdims = True)

        # save a bunch of helpful data from the h5py file
        index_map = self.data['index_map']
        self.rms = self.data['rms']
        self.freq = index_map['freq']
        self.ra = np.array(index_map['ra'])
        self.dec = np.degrees(np.arcsin(np.array([index_map['el'][0],index_map['el'][-1]]))) + ephemeris.CHIMELATITUDE
        
        if self.day:
            # Get unix timestamps corresponding to day and RA
            self._timestamp = ephemeris.csd_to_unix(self.day + self.ra / 360.0)

            # Note: ~ is logical not

            # times that are NOT daytime
            self._daytime_mask = ~flagging.daytime_flag(self._timestamp) 

            # Times when the moon is NOT up
            self._moon_mask = ~flagging.transit_flag(planets['moon'], self._timestamp, nsigma=2.0) 


    def _get_good_ra(self, freq):
        """
        Get the RA indices where the data is nonzero. 
        """
        if self.day:
            # Times when the data is nonzero
            rms_mask = self.rms[0, freq, :] != 0 

            # RA indices where the data is usable
            good_ras = np.nonzero(self._daytime_mask & self._moon_mask & rms_mask)

            return good_ras
            
        # If this is a template map, all the RAs should be fine
        return np.arange(len(self.ra))

    def get_ra_mask(self, freq):
        """
        Get a numpy array mask for the good RAs.
        """
        # True is bad. Assume all points are bad
        ra_mask = np.ones_like(self.ringmap[freq], dtype=bool)
        # Get the indices of good RAs
        good_ras = self._get_good_ra(freq)
        # Set those to be False (good)
        ra_mask[:, good_ras] = False
        return ra_mask

    def get_masked_map(self, freq, median_subtract = False):
        """
        Get a RA masked frequency slice of the Ringmap. 
        Median subtract to remove crosstalk. 
        """
        masked_map = np.ma.array(self.ringmap[freq], mask = self.get_ra_mask(freq))
        if median_subtract:
            # Median subtract along rows
            masked_map -= np.ma.median(masked_map, axis = 1, keepdims = True)
        return masked_map
    
    def subtract(self, other):
        return self.ringmap - other.ringmap

    def rms_flag(self, freq):
        # True means there is no nonzero data
        return np.sum(self.rms[:, freq, :].flatten()) == 0

    def ra_flag(self, freq):
        # True means all RAs in the map at freq are bad
        return len(self._get_good_ra(freq)) == 0

    def plot_map(self, fr_index, other = None):
        """
        Do a basic plot of the ringmap at a given frequency.
        See plot_ringmap above for more details.
        """
        title = self.title
        if other: 
            ringmap = self.subtract(other)[fr_index]
            title = f"{other.title} subtracted from {self.title}"
        else:
            ringmap = self.ringmap[fr_index]
        title = f"{title} at Freq. {fr_index}"
            
        plot_ringmap(ringmap, title)

    def plot_hist(self, fr_index, zoomed = False):
        """
        Plot a basic histogram of the ringmap at a given frequency. 
        Zoomed -> restrict to 3-sigma and closer. 
        See plot_ringmap_histogram above for more details. 
        """
        title = f"Histogram of {self.title} Data"
        plot_ringmap_histogram(self.ringmap[fr_index].flatten(), title, zoomed)


def main(freqs, debug = False):
    """
    Calculate and save a bunch of statistics
    """
    if debug:
        print(f"Main started with freqs {freqs}")

    # Columns of the dataframes
    df_columns = ['CSD', 'mean', 'median', 'stdev', 'skew', 'kurtosis', 'zsum', 'nonzero']

    # Dictionary where key is frequency (index) and value is a pandas dataframe of stats
    freq_dfs = {}
    
    # Load in the template
    template_map = RingMap(median_subtract = True)

    data_folder = RingMap.data_folder
    # Make a list of the available CSDs
    days = [os.path.split(fl)[-1] for fl in glob(data_folder + '*') if os.path.split(fl)[-1].isdigit()]

    # Dictionary where key is CSD and value is a list of frequencies for which there is no data
    no_data_days = {}

    counter = 0
    fr_stats = [[] for freq in freqs]
    for day in days:

        try:
            day_map = RingMap(day)
        except OSError:
            print(f'Day {day} has no ringmap.')
            continue

        no_data_days[day] = []
        for i, freq in enumerate(freqs):
            # Calculate a bunch of statistics on the map
            # Plot every 100 maps (change this if you want)
            stats = ringmap_stats(template_map, day_map, freq, plot = counter % 100 == 0)

            # ringmap_stats returns None if something went wrong
            if stats is None:
                no_data_days[day].append(freq)
                if debug: 
                    print(f"Bad day: CSD {day_map.day} Freq {freq}.")
                continue

            # Add the stats to the list
            fr_stats[i].append(stats)

            # If you want more feedback as it runs, this tells you every 20 maps processed
            if debug and counter % 20 == 0:
                print(f"Done processing CSD {day_map.day} Freq {freq}. {counter+1} maps processed. ")
            counter += 1

    for stats, freq in zip(fr_stats, freqs):
        # make a dataframe from the 2-d list
        freq_dfs[freq] = pd.DataFrame(stats, columns = df_columns)
        # save the dataframe as a CSV
        freq_dfs[freq].to_csv(f'{output_directory}CSD_stats_out_fr{freq}.csv', index = False)
    
    # Pickle the dictionary of dataframes 
    with open(f'{output_directory}freq_stats_dict.pickle', 'wb') as fl:
        pickle.dump(freq_dfs, fl)
    
    # Pickle the dictionary of bad days
    with open(f'{output_directory}bad_days.pickle', 'wb') as fl:
        pickle.dump(no_data_days, fl)

    return 0

def ringmap_stats(template, day_map, freq, plot = False, all_plots = False):
    """
    Get all the statistics for a given ringmap at a given frequency. 
    Run with plot to save a plot of the final processed map and histogram. 
    Run with all_plots to show all the intermediate stages. 
        Only do this in a notebook/as a learning tool. Don't use for full
        stat calculations. 
    """
    
    # If all the data is 0, return None
    if day_map.rms_flag(freq):
        return None

    # If all RAs are bad, return None
    if day_map.ra_flag(freq):
        return None

    # Plot the map and histograms before any processing has been done
    if all_plots:
        day_map.plot_map(freq)
        plt.show()

        day_map.plot_hist(freq)
        plt.show()

        day_map.plot_hist(freq, zoomed = True)
        plt.show()

    # Plot the masked map and histograms (before median subtraction)
    if all_plots:
        masked_map_no_median = day_map.get_masked_map(freq, median_subtract=False)

        title = f"{day_map.title} with Masking at Freq. {freq}"
        plot_ringmap(masked_map_no_median, title)
        plt.show()

        title = f"Histogram of {day_map.title} Data\nwith Masking at Freq. {freq}"
        plot_ringmap_histogram(masked_map_no_median.compressed(), title)
        plt.show()

        title = f"Zoomed Histogram of {day_map.title} Data\nwith Masking at Freq. {freq}"
        plot_ringmap_histogram(masked_map_no_median.compressed(), title, no_outliers=True)
        plt.show()

        
    # Get the masked and median subtracted map
    masked_map = day_map.get_masked_map(freq, median_subtract=True)

    # Plot the map and histograms (after masking and median subtraction)
    if all_plots:
        title = f"{day_map.title} with Masking and Median Subtraction at Freq. {freq}"
        plot_ringmap(masked_map, title)
        plt.show()

        title = f"Histogram of {day_map.title} Data\nwith Masking and Median Subtraction at Freq. {freq}"
        plot_ringmap_histogram(masked_map.compressed(), title)
        plt.show()

        title = f"Zoomed Histogram of {day_map.title} Data\nwith Masking and Median Subtraction at Freq. {freq}"
        plot_ringmap_histogram(masked_map.compressed(), title, no_outliers=True)
        plt.show()
        
    # Mask the template in the same way as the daily map
    # This avoids getting a negative of the template
    masked_template = template.get_masked_map(freq)
    masked_template.mask = masked_map.mask 

    # Subtract the template from the map
    masked_difference = masked_map - masked_template

    # Calculate statistics 
    mean = np.ma.mean(masked_difference)
    median = np.ma.median(masked_difference)
    std = np.ma.std(masked_difference)
    skew_stat = skew(masked_difference, axis = None)
    kurtosis_stat = kurtosis(masked_difference, axis = None)
    zsum = np.sum(np.abs(masked_difference - mean)/(std*masked_difference.count(axis = None)))

    # This is a useful "statistic" for assessing how usable data is
    # If most of the data is zero, we can safely throw it away
    nonzero = masked_difference.count()/masked_difference.size


    # Plot (and save) the final differenced maps and histograms
    if all_plots or plot:
        title = f'{day_map.title} Differenced with Template at Freq. {freq}'
        plot_ringmap(masked_difference, title)
        plt.savefig(f"{plots_directory}diff_map_csd{day_map.day}_fr{freq}.png", bbox_inches = 'tight')
        if all_plots: plt.show()
        else: plt.close()

        title = f"Histogram of {day_map.title} Data\nDifferenced with Template at Freq. {freq}"
        plot_ringmap_histogram(masked_difference.compressed(), title)
        plt.savefig(f"{plots_directory}diff_hist_csd{day_map.day}_fr{freq}.png", bbox_inches = 'tight')
        if all_plots: plt.show()
        else: plt.close()

        title = f"Zoomed Histogram of {day_map.title} Data\nDifferenced with Template at Freq. {freq}"
        # This is kinda wack, basically the zoomed histogram breaks if sigma is not defined (e.g. all data is 0)
        # So to avoid throwing an error I wrapped it in a try
        # There's probably a way to fix plot_ringmap_histogram to avoid this. 
        try:
            plot_ringmap_histogram(masked_difference.compressed(), title, no_outliers=True)
            plt.savefig(f"{plots_directory}zoomed_diff_hist_csd{day_map.day}_fr{freq}.png", bbox_inches = 'tight')
            if all_plots: plt.show()
            else: plt.close()
        except Exception:
            print('Histogram creation failed. Title:', title)

    # Make a nice list of the statistics
    stats_list = [day_map.day, mean, median, std, skew_stat, kurtosis_stat, zsum, nonzero]
    
    # Convert any remaining masked values (stored as "--") to nans for simplicity
    # By default python sends a warning about this but it's fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats_list = np.array(stats_list)

    return stats_list
    



if __name__ == '__main__':
    if len(sys.argv) >=2:
        # If you want to do a test run with a few frequencies 
        # you can do that from the command line
        freqs = [int(fr) for fr in sys.argv[1:]]
    else:
        # Otherwise do all the frequencies
        # Feel free to change this to a subset for testing
        freqs = list(range(0,1024))

    main(freqs, debug = True)
    

"""
Some possible things to look at in the future:
Look at where the sun is on a given day, and compare that with the actual masking
    I'm worried that the masking isn't quite doing the correct thing
Look at closer outliers (see KDE plot) and see what makes those ringmaps different
    Try using the ringmap_stats function with all_plots = True in a notebook
    to look at a given map more closely.
Look at maps for outliers in skew kurtosis and zsum
Examine the zsum double gaussian, see if we have 2 classes of map
    See the "meta" histogram of zsum in the KDE plot 
Try changing the template so you median subtract after masking
Median across freqs on a given day, with RFI masking
Investigate frequency dependence in the statistics more
"""

