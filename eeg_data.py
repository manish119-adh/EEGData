
# bandpass filter the ddata

from collections import namedtuple
from math import ceil, floor
import os
from tkinter import S
from types import NoneType
import numpy as np
from pyedflib import EdfReader
from globals import EEGEvent
from scipy import signal


class EEGData:

    def __init__(self,  data:list[list[np.array]], frequency:float , files:list[str] = [], channels:list[str]=[], annotations=None):
        self._data = data
        self._channels = channels
        self._frequency = frequency
        self._files = files
        self._annotations = annotations

        

    @staticmethod
    def load_from_EEG(directory:str):
        '''
            Load data from USENIX InexpensiveBCI Dataset given the directory
        '''
        # Load data from multiple EDF files along with the associated labels
        signalslist = []
        edffiles = []
        for fil in os.listdir(directory):
            
            if fil[-4:] == ".edf":
                try:
                    edf = EdfReader(os.path.join(directory, fil))
                    
                    print(f"Loaded {os.path.join(directory, fil)}")
                except Exception as ex:
                    print(f"cannot read file {os.path.join(directory, fil)}")
                    continue
                annotations=None
                if edf.filetype == 1:
                    # It is edf+ get annotations
                    annotations = edf.readAnnotations()
                    pass

                # Metadata
                channels = edf.getSignalLabels()
                fs_list = edf.getSampleFrequencies()
                n_channels = edf.signals_in_file
                
                # For simplicity, assume all channels have same fs
                fs = fs_list[0]
                # Read signals
                signals = []
                for ch in range(n_channels):
                    sig = edf.readSignal(ch, start=0, n=None, digital=False)
                    signals.append(sig)
                signalslist.append(signals)
                edf.close()
        return EEGData(signalslist, fs, channels=channels, annotations=annotations)
        

    @staticmethod
    def load_from_csv(filenames, labels, sep=","):
        '''
        load data from csv
        '''
        return 


    def select_channels(self, channels:list[str]):
        '''
        filter data by channels
        '''
        return



    def fir_filter(self, low:float|NoneType=None, high:float|NoneType=None, ntaps=100):
        '''
        bandpass the data with finite impulse response (FIR) algorithm
        If low and high are both None do nothing
        If low is None, it is a lowpass filter. If high is None it is highpass filter
        If neither is None, it is a bandpass filter between low and high

        Note that signal filtering is done on  data itself and no filtered copy
        is made. If you need original signal, you need to make a copy beforehand or load
        the signal again
        '''
        pass_zero = False
        if low is None:
            pass_zero = True
        if low is None and high is None:
            return
        if low is None:
            cutoff = high
        elif high is None:
            cutoff = low
        else:
            cutoff = [low, high]
        filter = signal.firwin(numtaps=ntaps, cutoff=cutoff, fs=self._frequency, pass_zero=pass_zero, window="hamming")
        for d in self._data:
            for i in range(len(self._channels)):
                d[i] = signal.filtfilt(filter, [1.0], d[i])
        

    def iir_filter(self, low:float|NoneType=None, high:float|NoneType=None, algo="butterworth"):
        '''
        IIR filter using given algorithm (default butterworth).
        If low and high are both None do nothing
        If low is None, it is a lowpass filter. If high is None it is highpass filter
        If neither is None, it is a bandpass filter between low and high

        Note that signal filtering is done on  data itself and no filtered copy
        is made. If you need original signal, make a copy beforehand or load
        the data from the source again
        '''
        return

    def bandstop_filter(self, low:float, high:float):
        '''
        Bandstop filter to filter out frequency between low and high. Useful to remove
        noises at narrow frequency ranges such as line noise

        Note that signal filtering is done on  data itself and no filtered copy
        is made. If you need original signal, make a copy beforehand or load
        the data again
        cannot have None
        '''
        return

    def resample(self, new_frequency:float):
        '''
        Resample the frequency. Uses direct sub-sampling for downsampling and interpolation
        for upsampling

        Note that signal resampling is done on  data itself and no filtered copy
        is made. If you need original signal, make a copy beforehand or load
        the data again
        '''
        return

    def copy(self):
        '''
        Copy the data. It is useful when we need access to 
        original signal if we later change them

        '''
        return EEGData([[sig.copy() for sig in dd] for dd in self._data], channels=list(self._channels) if self._channels is not None else None, frequency=self._frequency, files = list(self._files) if self._files is not None else None )

    def select_channels(self, channels):
        deleted_channels = set(self._channels).difference(set(channels))
        for channel in deleted_channels:
            idx = self._channels.index(channel)
            self._channels.pop(idx)
            for d in self._data:
                d.pop(idx)
        
            


        

    

    def epoch_data(self, subjects : list[str|int], event_lists :list[list[EEGEvent]], low:float=-0.5, hi:float=1.0):
        '''
        epoch each eeg data per event in the stream between low (default -1) and hi (default 2)
        seconds after the event onset, note that negative means use time before the event.
        The names/numbers of subjects are given in the subjects array and event_streams which
        give the event onset time and the event label to identify
        The first parameter in each event is timestamp

        '''
        # collect all the signal samples within low and hi of each event
        # assume all relevent signals are captured
        # make size for
        y = []
        X = []
        max_all = int((low - hi)*self._frequency)+1
        data_index = 0
        for (subject, event_list) in zip(subjects, event_lists):
            
            for event in event_list:
                low_time = event.timestamp + low
                high_time = event.timestamp + high_time
                low_sample = ceil(low_time * self._frequency)
                if low_sample < 0 or len(self._data[data_index][0]) < low_sample+max_all):
                    continue
                X.append(self._data[data_index][:][low_sample:low_sample + max_all])
                y.append((subject, event.event_label))


        return np.array(X), y

if __name__ == "__main__":
    edfdata = EEGData.load_from_EDF("data/USENIX_Inexpensive")
    
    edfdata_original = edfdata.copy()
    pass




