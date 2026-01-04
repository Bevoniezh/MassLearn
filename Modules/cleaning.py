# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:11:22 2023

@author: Ronan Le Puil
"""

import gc # garbage collector, to free the RAM
import re
import os
import math
import scipy
import base64
import logging
import pymzml
import peakutils
import numpy as np
import pandas as pd
import time as time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, peak_widths
from functools import partial
from Modules import logging_config

logger = logging.getLogger(__name__)


class Denoise(): 
    """
    Denoise manage file per file removing the noise from mzML files.
    Background noise thresholds (400 counts in MS1 and 200 in MS2 by default) define the minimum
    intensity considered when detecting systematic noise traces, but the raw mzML data are no longer
    filtered by those thresholds. Only the traces identified as noise are removed from the exported
    mzML, preserving lower-intensity signals that are not part of a noise trace.
    
    Denoise() uses Spectra() class to denoise. Denoise() is a class to link Spectra() methods to the mzML files.
    Then, a noise is identified by one property: If a m/z (+-0.005Da) is present at least 20% (Threshold_scans variable, can be more or less depending on the user
    of all the scans), it is identified a noise trace. 
    
    From each noise trace detected, significant peaks are identified following two properties. Peakutils library tool detect a peak based on specific parameters,
    they are defined by default but can be changed in Spectra.detect_peak(). Other property if the ratio between the peak(s) intensity(ies) detected by pekutil and 
    the median of all the Extracted ion chromatogram, meaning the median of all intensities values in this noise trace (Sepctra.noise_terminator() Median_treshold parameter).
    When a significant peak within a noise trace is detected in a nosie trace, all noise trace is deleted from the scans exept from the irght and left borders of this peak.    
    
    It has to be used as follow:
        First, preapre de Spectra objects
        Spectra = Spectra(Filename) # take all the spectra data
        Spectra.extract_peaks(Noise1, Noise2)
        
        After, denoise:
        to_denoise_file = Denoise(Filename, Featurepath)
        to_denoise_file.filtering(Spectra, threshold)
        
    """
    def __init__(self, Filename, Featurepath): # Filenames have to be in relative or absolute path not only the files name
        self.filename = Filename
        self.featurepath = Featurepath
        self.path = Filename
        self.begin = time.time()
        self.binary_count = 0 # attribute for self.filtering() method
        self.encoded_count = 0 # attribute for self.filtering() method
        
        with open(self.path, 'r') as file:
            self.file = file.read()
        self.scanlist = re.split(r'<spectrum index=', self.file) # divide the mzml file as list of spectra, first spectra begin at index 1
        value = re.search(r'scan=(\d+)', self.scanlist[1]).group(1) # take scan number
        if value == '0':
            self.scanlist.replace('scan=' + value, 'scan=1') # the idea is to adjust the scan number if it is 0

        # Add denoising metadata to mzml file
        software_metadata = """
        	  <software id="masslearn_denoising" version="1.0.0">
                <userParam name="mlms" type="xsd:string" value="masslearn denoising software"/>
              </software>
        """
        processing_method = """
        		<processingMethod order="1" softwareRef="masslearn_denoising">
                    <userParam name="denoising" type="xsd:string" value="Denoising with masslearn"/>
                </processingMethod>
        """
        self.scanlist[0] = re.sub('</softwareList>', software_metadata + '\n</softwareList>', self.scanlist[0])
        self.scanlist[0] = re.sub('</dataProcessing>', processing_method + '\n</dataProcessing>', self.scanlist[0])
        
        log = f'{os.path.basename(self.filename)} open.'
        logging_config.log_info(logger, log)

    # Function to encode in binary format the string based m/z and intensities values        
    def encode_binary(self, Peakarray_masked):
        encoded_data = []
        for masses_and_intensities in Peakarray_masked: 
            masses = masses_and_intensities[:, 0]
            intensities = masses_and_intensities[:, 1]
            
            # Convert float arrays to binary data
            data_mz = masses.astype(np.float32).tobytes() 
            data_i = intensities.astype(np.float32).tobytes()
    
            # Encode binary data using Base64
            encoded_mz = base64.b64encode(data_mz)
            encoded_i = base64.b64encode(data_i)
            encoded_data.append((encoded_mz.decode('utf-8'), encoded_i.decode('utf-8')))
        return encoded_data #list of tuples containing masses and intensities
    
    # Function to fliter the noise and re-werite mzml files    
    def filtering(self, Spectra, Threshold_scans = 20, Dash_app = False):
        spectra = Spectra # it is the Spectra() object
        
        logging_config.log_info(
            logger,
            'Starting denoising for %s (MS1 noise=%s, MS2 noise=%s, trace threshold=%s).',
            os.path.basename(self.filename),
            spectra.denoised.get('Noise MS1'),
            spectra.denoised.get('Noise MS2'),
            Threshold_scans,
        )

        try:
            res1, dft1 = spectra.noise_trace(spectra.peaks1, spectra.peakarray1, Threshold = Threshold_scans) # take noise trace information
            res2, dft2 = spectra.noise_trace(spectra.peaks2, spectra.peakarray2, Threshold = Threshold_scans)

            dft1_sorted = dft1.sort_index(ascending=True)  # Sort by index in ascending order
            rounded_values_dft1 = [str(round(value, 4)) for value in list(dft1_sorted.index)]

            dft2_sorted = dft2.sort_index(ascending=True)  # Sort by index in ascending order
            rounded_values_dft2 = [str(round(value, 4)) for value in list(dft2_sorted.index)]

            noise_list_ms1 ="\n".join(rounded_values_dft1)
            noise_list_ms2 ="\n".join(rounded_values_dft2)

            logging_config.log_info(
                logger,
                'Identified %d MS1 noise traces and %d MS2 noise traces for %s.',
                len(dft1_sorted.index),
                len(dft2_sorted.index),
                os.path.basename(self.filename),
            )

            if len(dft1_sorted.index) == 0:
                logging_config.log_warning(
                    logger,
                    'No MS1 noise traces detected for %s. Check noise thresholds or data quality.',
                    os.path.basename(self.filename),
                )
            if len(dft2_sorted.index) == 0:
                logging_config.log_warning(
                    logger,
                    'No MS2 noise traces detected for %s. Check noise thresholds or data quality.',
                    os.path.basename(self.filename),
                )

            log = f'MS1 noise trace for file {os.path.basename(self.filename)} are:\n{noise_list_ms1}'
            logging_config.log_info(logger, log)
            log2 = f'MS2 noise trace for file {os.path.basename(self.filename)} are:\n{noise_list_ms2}'
            logging_config.log_info(logger, log2)

            noise_directory = os.path.join(self.featurepath, 'noise')
            os.makedirs(noise_directory, exist_ok=True)
            with open(
                os.path.join(
                    noise_directory,
                    os.path.splitext(os.path.basename(self.filename))[0] + '.txt'
                ),
                'w'
            ) as fic:
                fic.write(f'{log}\n\n{log2}')

            dfp1 = spectra.detect_peak(dft1) # from the noise traces, we try to detect significant peak for those peaks not to be erased from the peak list
            dfp2 = spectra.detect_peak(dft2)

            # Apply noise masking to full spectra so low-intensity signals are preserved unless part of a detected noise trace
            pkma1, tic1, base1, length1 = spectra.noise_terminator(dfp1, spectra.full_peakarray1) # pkma is the masked array of peaks, the mask substract noise trace from the mass list at the exeption of significant peaks in noise trace if some are present
            pkma2, tic2, base2, length2 = spectra.noise_terminator(dfp2, spectra.full_peakarray2)

            logging_config.log_info(
                logger,
                'Noise terminator produced %d MS1 masked spectra and %d MS2 masked spectra for %s.',
                len(pkma1),
                len(pkma2),
                os.path.basename(self.filename),
            )

            spectra.denoised['Noise trace threshold'] = Threshold_scans # add info that spectra have been proceed with denoising functions already
            #TODO add possibility to indicate all the parameters values to remove noise in spectra.denoise
            encoded1 = self.encode_binary(pkma1)
            encoded2 = self.encode_binary(pkma2)

            logging_config.log_info(
                logger,
                'Encoded %d MS1 spectra and %d MS2 spectra for %s.',
                len(encoded1),
                len(encoded2),
                os.path.basename(self.filename),
            )

            for sp, scan in enumerate(self.scanlist[1:-1]): # sp is the count of spectrum tag
                ms_level = re.search(r'name="ms level" value="(\d+)"', scan).group(1)
                if ms_level == '1':
                    scan = re.sub(r'defaultArrayLength="(\d+)"', r'defaultArrayLength="{}"'.format(str(length1[sp//2])), scan)  #set the new value of the spectrum length, meaning the number of peaks
                    scan = re.sub(r'<binary>.*?</binary>', partial(self.replacer_binary, encoded1[sp//2][0], encoded1[sp//2][1]), scan) # Change the content of the first and second binary tags, meaning the mass and intensities values respectively
                    scan = re.sub(r'<binaryDataArray encodedLength=".*?">', partial(self.replacer_length, str(len(encoded1[sp//2][0])), str(len(encoded1[sp//2][0]))), scan) # Set the number of character in binary, there are the same for mz or intensities
                    # Update base peak and tic
                    scan = re.sub(r'name="base peak m/z" value="([\d.]+)"', r'name="base peak m/z" value="{}"'.format(str(base1[sp//2][0])), scan)
                    scan = re.sub(r'name="base peak intensity" value="([\d.]+)"', r'name="base peak intensity" value="{}"'.format(str(base1[sp//2][1])), scan)
                    scan = re.sub(r'name="total ion current" value="([\d.]+)"', r'name="total ion current" value="{}"'.format(str(tic1[sp//2])), scan)
                else:
                    scan = re.sub(r'defaultArrayLength="(\d+)"', r'defaultArrayLength="{}"'.format(str(length2[sp//2])), scan)  #set the new value of the spectrum length, meaning the number of peaks
                    scan = re.sub(r'<binary>.*?</binary>', partial(self.replacer_binary, encoded2[sp//2][0], encoded2[sp//2][1]), scan) # Change the content of the first and second binary tags, meaning the mass and intensities values respectively
                    scan = re.sub(r'<binaryDataArray encodedLength=".*?">', partial(self.replacer_length, str(len(encoded2[sp//2][0])), str(len(encoded2[sp//2][0]))), scan) # Set the number of character in binary, there are the same for mz or intensities
                    # Update base peak and tic
                    scan = re.sub(r'name="base peak m/z" value="([\d.]+)"', r'name="base peak m/z" value="{}"'.format(str(base2[sp//2][0])), scan)
                    scan = re.sub(r'name="base peak intensity" value="([\d.]+)"', r'name="base peak intensity" value="{}"'.format(str(base2[sp//2][1])), scan)
                    scan = re.sub(r'name="total ion current" value="([\d.]+)"', r'name="total ion current" value="{}"'.format(str(tic2[sp//2])), scan)
                self.scanlist[sp+1] = scan # sp+1 because we begin at 1 in self.scanlist, 0 is the first part of mzML file we do not modify
        except Exception as exc:
            logging_config.log_exception(
                logger,
                'Denoising failed for %s.',
                os.path.basename(self.filename),
                exception=exc,
            )
            raise

        # Merge all the modified scans together into a final file
        self.file = '<spectrum index='.join(self.scanlist)

        # Create exported denoised file
        with open(self.filename[:-5] +'.mzML', 'w') as file: # Save the modified XML file
            file.write(self.file) 
        log = f'Time required to remove noise from {os.path.basename(self.filename)}: {int(time.time()-self.begin)} s'
        logging_config.log_info(logger, log)
        
        # Make the two dictionnaries to store the rt and corresponding spectra info
        ms1_spectra = {}
        for rt, pk in zip(spectra.rt1, pkma1):
            mask = ~(np.all(pk == 0, axis=1))
            ms1_spectra[rt] = pk[mask]
        
        ms2_spectra = {}
        for rt, pk in zip(spectra.rt2, pkma2):
            mask = ~(np.all(pk == 0, axis=1))
            ms2_spectra[rt] = pk[mask]
        
        if Dash_app == True:
            return spectra, ms1_spectra, ms2_spectra
        else:
            return spectra
    
    # Function to add noise masses from each sample file in a text document
    def add_noise(self, Txt):
        with open('noise_list.txt', 'a') as file:
            file.write(f'{Txt}\n\n')
     
     
    # Function to replace <binary> contents bassed on their position
    def replacer_binary(self, Masses, Intensities, match):
        self.binary_count += 1
        if self.binary_count == 1:
            return f'<binary>{Masses}</binary>'
        elif self.binary_count == 2:
            self.binary_count = 0 # rest the count for next scan
            return f'<binary>{Intensities}</binary>'
        else:
            return match.group()
        
    # Function to replace encodedLength contents bassed on their position
    def replacer_length(self, Masses, Intensities, match):
        self.encoded_count  += 1
        if  self.encoded_count  == 1:
            return f'<binaryDataArray encodedLength="{Masses}">'
        elif  self.encoded_count == 2:
            self.encoded_count = 0 # reset
            return f'<binaryDataArray encodedLength="{Intensities}">'
        else:
            return match.group()
        

class Spectra():
    """    
    Object of class spectra store all the spectra data from a file
    mz and intensities are the major values, they are stored as raw and filtered (decimals)
    
    """    
    def __init__(self, Filename):
        self.filename = os.path.basename(Filename)
        self.denoised = {}
        self.path = Filename
        self.peaks1 = [] # extract_peaks have to be called before updating the peaks, mz and intensities
        self.peaks2 = []
        self.all_peaks1 = []  # Peaks without intensity-based background filtering
        self.all_peaks2 = []
        self.peakarray1 = []
        self.peakarray2 = []
        self.full_peakarray1 = []
        self.full_peakarray2 = []
        self.length1 = [] #to store the length of each ms1 spectra (original and filtered)
        self.length2 = [] 
        self.dfp1 = []
        self.dfp2 = []
        self.dft1 = []
        self.dft2 = []
        self.noise_list1 = ""
        self.noise_list2 = ""      
        self.rt1 = []
        self.rt2 = []
        
    def extract_peaks(self, Noise1 = 400, Noise2 = 200): # Noise1 is a treshold for noise of MS1, Noise2 is for MS2
        self.denoised = {'Noise MS1' : Noise1, 'Noise MS2' : Noise2} # to indicate what was remove from noise background
        self.peaks1 = []
        self.peaks2 = []
        self.all_peaks1 = []
        self.all_peaks2 = []
        self.tic1 = []
        self.tic2 = []
        self.length1 = []
        self.length2 = []
        self.rt1 = []
        self.rt2 = []
        peak1_lengths = [] # to have the lenght of the spectra, to at the end having the longest one
        peak2_lengths = []
        all_peak1_lengths = []
        all_peak2_lengths = []
        logging_config.log_info(
            logger,
            'Extracting peaks for %s with MS1 noise=%s and MS2 noise=%s.',
            self.filename,
            Noise1,
            Noise2,
        )
        empty_ms1 = 0
        empty_ms2 = 0
        with pymzml.run.Reader(self.path) as run:
            for count, spectrum in enumerate(run): # iterate through spectra
                if spectrum.__getitem__("MS:1000511") == 1.0: #if the ms level is 1, we used __getitem__ method from pymzml to access the values of all accession
                    peaks = np.column_stack((spectrum.mz, spectrum.i))

                    # Filter the data
                    peaks = peaks[np.where(np.floor(peaks[:, 0] % 1 <= 0.8))] # Masses with removed decimals , < x.8
                    self.all_peaks1.append(peaks)

                    detection_peaks = peaks[np.where(peaks[:, 1] >= Noise1 )] # Masses above Noise treshold
                    if not(detection_peaks.any()):
                        detection_peaks = np.array([[1000.00000, 1]]) # if there is nothing in the spectra we add a mock peak alone in the spectrum originally, for everything below the noise or with wrong decimal
                        empty_ms1 += 1
                    self.peaks1.append(detection_peaks)
                    self.rt1.append(round(spectrum.__getitem__("MS:1000016"), 4)) # add the retention time from the spectra to the list of rt
                    peak1_lengths.append(len(detection_peaks))
                    all_peak1_lengths.append(len(peaks))
                # MS level 2, same as MS level 1:
                else:
                    peaks = np.column_stack((spectrum.mz, spectrum.i))
                    peaks = peaks[np.where(np.floor(peaks[:, 0] % 1 <= 0.8))] # Masses with removed decimals , < x.8
                    self.all_peaks2.append(peaks)

                    detection_peaks = peaks[np.where(peaks[:, 1] >= Noise2 )] # Masses above Noise treshold
                    if not(detection_peaks.any()):
                        detection_peaks = np.array([[1000.00000, 1]]) # We add a mock peak alone in the spectrum originally, everything is below the noise or with wrong decimal
                        empty_ms2 += 1
                    self.peaks2.append(detection_peaks)
                    self.rt2.append(round(spectrum.__getitem__("MS:1000016"), 4))
                    peak2_lengths.append(len(detection_peaks))
                    all_peak2_lengths.append(len(peaks))

        if peak1_lengths:
            max_length = max(peak1_lengths) # max_length is necessary because peakarray is an array, so all dimensions must have same length, meaning the length of the longest peaks1 array
            self.peakarray1 = np.zeros((len(self.peaks1), max_length, 2)) # create an array full of zeros, of the size of the longest spectra in terms of number of different m/z detected)
            # Copy the values from the original arrays to the new array
            for i, arr in enumerate(self.peaks1):
                self.peakarray1[i, :len(arr)] = arr
        else:
            logging_config.log_warning(
                logger,
                'No MS1 spectra were extracted from %s. Downstream denoising may fail.',
                self.filename,
            )
            self.peakarray1 = np.zeros((0, 0, 2))

        # same for ms level 2s
        if peak2_lengths:
            max_length = max(peak2_lengths)
            self.peakarray2 = np.zeros((len(self.peaks2), max_length, 2))
            for i, arr in enumerate(self.peaks2):
                self.peakarray2[i, :len(arr)] = arr
        else:
            logging_config.log_warning(
                logger,
                'No MS2 spectra were extracted from %s. Downstream denoising may fail.',
                self.filename,
            )
            self.peakarray2 = np.zeros((0, 0, 2))

        if all_peak1_lengths:
            max_length = max(all_peak1_lengths)
            self.full_peakarray1 = np.zeros((len(self.all_peaks1), max_length, 2))
            for i, arr in enumerate(self.all_peaks1):
                self.full_peakarray1[i, :len(arr)] = arr
        else:
            self.full_peakarray1 = np.zeros((0, 0, 2))

        if all_peak2_lengths:
            max_length = max(all_peak2_lengths)
            self.full_peakarray2 = np.zeros((len(self.all_peaks2), max_length, 2))
            for i, arr in enumerate(self.all_peaks2):
                self.full_peakarray2[i, :len(arr)] = arr
        else:
            self.full_peakarray2 = np.zeros((0, 0, 2))

        logging_config.log_info(
            logger,
            'Extracted %d MS1 spectra (%d empty) and %d MS2 spectra (%d empty) from %s.',
            len(self.peaks1),
            empty_ms1,
            len(self.peaks2),
            empty_ms2,
            self.filename,
        )
        
       # del self.peaks1 # remove to free space
       # del self.peaks2 
        
        
        
     # Function to filter out all trace noise and keep only peaks FOR BOTH MS LEVELS:
    def noise_terminator(self, Dfp, Peakarray, Median_treshold = 0.1, Delta = 0.005 ): # Peakarray can be self.peakarray1
        level = 'MS1' if Dfp.equals(self.dfp1) else 'MS2'
        logging_config.log_info(
            logger,
            'Applying noise terminator for %s in %s (noise masses=%d, spectra=%d).',
            level,
            self.filename,
            len(Dfp.index),
            Peakarray.shape[0],
        )

        if Dfp.empty:
            logging_config.log_warning(
                logger,
                'No %s noise peaks available for %s. Spectra will remain unmasked.',
                level,
                self.filename,
            )

        if Peakarray.size == 0:
            logging_config.log_warning(
                logger,
                '%s peak array for %s is empty while applying noise terminator.',
                level,
                self.filename,
            )
            empty_arrays = [np.zeros((0, 2)) for _ in range(Peakarray.shape[0])]
            return empty_arrays, [0 for _ in empty_arrays], [(0, 0) for _ in empty_arrays], [0 for _ in empty_arrays]

        # Create a mask for peakarray masses
        mask = np.full((Peakarray.shape[0], Peakarray.shape[1]), False)
        # Iterate through the noise masses
        for noise_mass in Dfp.index:
            left_borders = Dfp.loc[noise_mass, 'Left_borders_peaks']
            right_borders = Dfp.loc[noise_mass, 'Right_borders_peaks']
            relative_median = Dfp.loc[noise_mass, 'Relative_median']
            
            # Create a boolean array to represent the relevant peak ranges for all scans
            in_relevant_peak_range = np.zeros((Peakarray.shape[0], Peakarray.shape[1]), dtype=bool)
            if  relative_median < Median_treshold: # The treshold for relative median wroks only if we have a large rt range, meaning if we do analysis on 1min or 30s of rt range, the feature curve shape will be far much wider and it will be difficult to differentiate from noise with relative median indication
                # TODO chek if the detected peak noise trace intensity at the right and left are removed from the detected peak
                for left, right in zip(left_borders, right_borders):
                    in_relevant_peak_range[left:right+1, :] = True

        
            # Iterate through the peakarray
            for j, masses_and_intensities in enumerate(Peakarray):
                masses = masses_and_intensities[:, 0]
        
                # Find masses within the noise mass range
                mass_in_range = ((masses >= noise_mass - 0.005) & (masses <= noise_mass + 0.005)) | (masses == 0)
        
                # Update the mask if the mass is within the noise mass range and not in relevant peak range
                mask[j] = np.logical_or(mask[j], np.logical_and(mass_in_range, ~in_relevant_peak_range[j]))
        
        # Apply the mask to peakarray masses and intensities columns
        masked_arrays = [Peakarray[k][~mask[k]] for k in range(len(Peakarray))]
        
        tic = []
        base = []
        length = []
        for arr in masked_arrays:
            tic.append(np.sum(arr[:, 1]))
            try:
                max_intensity_index = np.argmax(arr[:, 1]) # index of the peak with the highest intensity in the filtered peaks array using np.argmax()
                base.append((arr[max_intensity_index, 0], arr[max_intensity_index, 1]))       
            except ValueError:
                base.append((0, 0))
            length.append(len(arr[:, 1]))
        logging_config.log_info(
            logger,
            'Noise terminator retained a median of %s peaks per spectrum for %s in %s.',
            np.median(length) if length else 0,
            level,
            self.filename,
        )

        return masked_arrays, tic, base, length
    

    # Function to have all the scans number between left and right border of peak inside noise traces
    def scan_nb(self, left, right):
        l = []
        for i in range(len(left)):
            count = 0            
            for j in range(right[i]-left[i]):
                if count == 0:
                    l.append(left[i])
                    count = 1
                l.append(l[-1] + 1)
        return l # the list of all scan number between the peaks windows        


    #Function to get all the noise trace FOR ONE MS LEVEL based on their grouping woth a specific +- xx.xxx Da delta and a threshold of minimum number of scans when there 
    def noise_trace(self, Peaks, Peakarray, Delta = 0.005, Threshold = 20):
        # Get all the possible noise trace if they appear in a minimum number of scans based on Threshold (% of scans)
        level = 'MS1' if Peaks is self.peaks1 else 'MS2'
        logging_config.log_info(
            logger,
            'Computing %s noise traces for %s (spectra=%d, delta=%s, threshold=%s).',
            level,
            self.filename,
            len(Peaks),
            Delta,
            Threshold,
        )

        if not Peaks:
            logging_config.log_error(
                logger,
                'No %s peaks extracted from %s. Unable to compute noise traces.',
                level,
                self.filename,
            )
            raise ValueError(f'No {level} peaks available for noise trace computation.')

        if Peakarray.size == 0:
            logging_config.log_error(
                logger,
                '%s peak array for %s is empty (shape=%s).',
                level,
                self.filename,
                Peakarray.shape,
            )
            raise ValueError(f'Empty {level} peak array for noise trace computation.')

        try:
            peaks_concat = np.concatenate(Peaks)
        except ValueError as exc:
            logging_config.log_exception(
                logger,
                'Failed to concatenate %s peaks while computing noise traces for %s.',
                level,
                self.filename,
                exception=exc,
            )
            raise
        peaks_concat[:,0] = np.round(peaks_concat[:,0], 3) # round only the mz values
        unique_mass, inverse_indices, counts = np.unique(peaks_concat[:,0], return_counts=True, return_inverse=True) # return count mean how many time a (unique) m/z is counted over all the array, meaning over all the scans
        counts = np.round(counts / len(Peaks) * 100, 2)
        sums = np.bincount(inverse_indices, weights=peaks_concat[:,1]) # Calculate sum of intensities for each unique mass value
        indices = np.where(counts >= Threshold)[0] # Get the indices of mass values that appear in at least 50% of the scans
        result = np.column_stack((unique_mass[indices], counts[indices], sums[indices])) # Combine unique mass, counts, and sums into a single 2D array
        result_sorted = result[result[:, 1].argsort()] # Sort the result array by the second column (counts) in descending order

        # Remove redundant noise trace from the same threshold delta
        if result_sorted.size == 0:
            logging_config.log_warning(
                logger,
                'No %s noise traces met the threshold for %s (delta=%s, threshold=%s).',
                level,
                self.filename,
                Delta,
                Threshold,
            )
            empty_trace = pd.DataFrame(columns=[])
            if Peaks is self.peaks1:
                self.dft1 = empty_trace
            else:
                self.dft2 = empty_trace
            return result_sorted, empty_trace

        masses = result_sorted[:, 0] # here are the masses
        expanded_masses = {} # Create an empty list to store the expanded array
        values = [round(1/(Delta*1000) * x * Delta, 3) for x in range(int(-Delta*1000), int(Delta*1000+1))] # all the mass that will be used aroufn the central noise trace, with a 0.001 gap between each
        for mass in masses: # Iterate over the array
            expanded_masses[mass] = []
            for i in values: # Add the neighboring masses within the specified range
                if i != 0:
                    expanded_masses[mass].append(round(mass + i, 3))
        for mass_to_check in reversed(masses):        
            if mass_to_check in [item for sublist in expanded_masses.values() for item in sublist]:
                del expanded_masses[mass_to_check]
        noise_masses = list(expanded_masses.keys())

        logging_config.log_info(
            logger,
            'Identified %d %s noise traces for %s after filtering overlapping masses.',
            len(noise_masses),
            level,
            self.filename,
        )
        

        # Get all the noise traces highest intensities in a +- Delta range
        chromatogram = np.zeros((Peakarray.shape[0], len(noise_masses))) # Initialize a 2D array to store the chromatogram intensities
        for i, masses_and_intensities in enumerate(Peakarray): # Iterate through the peakarray
            masses = np.round(masses_and_intensities[:, 0],3)
            intensities = np.round(masses_and_intensities[:, 1],0)

            # Iterate through the noise masses
            for j, noise in enumerate(noise_masses):
                # Find the indices of masses within the noise mass range
                mass_indices = (masses >= noise - 0.005) & (masses <= noise + 0.005)

                # Get the intensities corresponding to the found masses
                mass_intensities = intensities[mass_indices]

                # If there are any intensities found, store the maximum intensity for the current noise mass and scan
                if mass_intensities.size > 0:
                    chromatogram[i, j] = np.max(mass_intensities)

        # Create a DataFrame with masses as index and chromatogram intensities as columns
        df_trace = pd.DataFrame(chromatogram, columns=noise_masses).T

        logging_config.log_info(
            logger,
            '%s noise chromatogram for %s has shape %s.',
            level,
            self.filename,
            df_trace.shape,
        )

        if Peaks is self.peaks1:
            self.dft1 = df_trace
        else:
            self.dft2 = df_trace
        
        # TODO below, Peaks == sef.peaks1 is ambigous because we compare arrays
        """
        # Update the class variables
        if Peaks == self.peaks1:
            self.dft1 = df_trace
            MS1_noise_traces = np.round(result[:,0], 3).tolist()
            self.noise_list1 += f'{self.filename}, MS level 1 noise trace:{MS1_noise_traces}\n\n'
        else:
            self.dft2 = df_trace
            MS2_noise_traces = np.round(result[:,0], 3).tolist()
            self.noise_list2 += f'{self.filename}, MS level 2 noise trace:{MS2_noise_traces}\n\n'
        """
        return result, df_trace

    
    # Function to get all the intensity values from a given range for every noise trace, in a df
    def noise_all_delta_intensities(self, Peaks, Expanded_masses, Noise_masses):
        # Create the datafarame of intensities around the Noise traces (at a secific delta, like +- 0.002 Da)
        all_intensities = np.zeros((len(Expanded_masses), len(Peaks))) # Create the output array with the correct shape and fill it with 0s
        for i, mass in enumerate(Expanded_masses): # Iterate over each mass and each scan, setting the appropriate elements in the output array
            for j, peaks in enumerate(Peaks):
                peaks_rounded = np.round(peaks, 3) # Round the mass values in the current scan to the third decimal
                index = np.where(peaks_rounded[:,0] == mass)[0] #index = np.where(peaks_rounded == mass)[0] # Find the index of the current mass in the rounded mass values for the current scan
                if len(index) > 0: # If the current mass is present in the current scan, set the corresponding element in the output row to the rounded intensity value
                    intensity = np.round(peaks[index, 1][0], 0)
                    all_intensities[i, j] = intensity
                else:
                    intensity = 0
        df_all = pd.DataFrame(all_intensities, index = Expanded_masses, columns=[i for i in range(len(Peaks))]) # df wich take all the intensities values
        df_all = df_all[~(df_all == 0).all(axis=1)] # Filter out all rows that contain only 0s
        logging_config.log_info(
            logger,
            'Generated intensity matrix for %s with %d expanded masses and %d scans.',
            self.filename,
            len(df_all.index),
            len(Peaks),
        )
        return df_all

    # Function to get the baseline of the Noise signal based on Algorithm of Least Square (ALS) https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    def baseline_als(self,y, lam, p, niter):
        L = len(y)
        D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = scipy.sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = scipy.sparse.linalg.spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    
    # Function to remove all the flat signal close to 0 which could be present in for example the first halves of the scans, that could bia the median value in a relevant noise that would appear only at the beggining or at the end of the elution
    def deflat(self, Array, Threshold_flat):        
        differences = np.diff(Array) # Calculate the differences between adjacent elements       
        std_diff = np.std(differences) # Calculate the standard deviation of the differences    
        keep_indices = [0] # Identify the indices of the values to keep, always keep the first value
        for i in range(1, len(Array)):
            if abs(differences[i-1]) > Threshold_flat * std_diff:
                keep_indices.append(i)
        filtered_spectra = Array[keep_indices] # Create a new array with only the values that pass the threshold
        return filtered_spectra
    
    
    # Function to detect the significant peaks and take their left and rights borders
    def detect_peak(self, Dft, LAM = 10000, P = 0.01, Niter = 10, window_length = 15, poly_order = 10, Peak_Threshold = 0.5, Threshold_flat = 0.9, scan_threshold = 5):
        level = 'MS1' if Dft.equals(self.dft1) else 'MS2'
        logging_config.log_info(
            logger,
            'Detecting significant peaks for %s noise traces of %s (traces=%d).',
            level,
            self.filename,
            len(Dft.index),
        )

        if Dft.empty:
            logging_config.log_warning(
                logger,
                'Noise chromatogram for %s in %s is empty. Skipping peak detection.',
                level,
                self.filename,
            )
            if level == 'MS1':
                self.dfp1 = Dft
            else:
                self.dfp2 = Dft
            return Dft

        df_peak = pd.DataFrame({'m/z': Dft.index, 'Relative_median': ['NAN' for i in range(len(Dft.index))],
                                'All_Peaks':['NAN' for i in range(len(Dft.index))],
                                'Left_borders_all':[[] for i in range(len(Dft.index))],
                                'Right_borders_all':[[] for i in range(len(Dft.index))],
                                'Peaks':[[] for i in range(len(Dft.index))],
                                'Left_borders_peaks':[[] for i in range(len(Dft.index))],
                                'Right_borders_peaks':[[] for i in range(len(Dft.index))],
                                'Peaks_scans':[[] for i in range(len(Dft.index))]})
        df_peak = df_peak.set_index('m/z')
        skipped_traces = 0
        for i in range(len(Dft)):
            try:
                data = Dft.iloc[i].replace(0, np.nan).interpolate(method='linear', limit_direction='both') # interpolate() here fill the NaN with values between the 0, meaning it artificially fill in some values
            except Exception as exc:
                logging_config.log_exception(
                    logger,
                    'Failed to prepare chromatogram for %s noise trace %.3f in %s.',
                    level,
                    Dft.index[i],
                    self.filename,
                    exception=exc,
                )
                skipped_traces += 1
                continue

            try:
                smooth_signal = savgol_filter(data, window_length, poly_order) # Apply Savitzky-Golay filter to smooth the signal
            except ValueError as exc:
                logging_config.log_exception(
                    logger,
                    'Savitzky-Golay smoothing failed for %s noise trace %.3f in %s.',
                    level,
                    Dft.index[i],
                    self.filename,
                    exception=exc,
                )
                skipped_traces += 1
                continue
            
            # Apply baseline correction to remove the overall behavior and noise
            baseline = self.baseline_als(smooth_signal, LAM, P, Niter)
            corrected_signal = smooth_signal - baseline 
            corrected_data = data - baseline 
            
            deflat_data = self.deflat(corrected_data, Threshold_flat) # remove wide range of uniques values to avoid medians bias
            relative_median = np.median(deflat_data) / max(deflat_data) # the relative median acts as an additional information to investigate what is a peak or not
           
            # Find the peaks in the baseline-subtracted data
            peak_indices = peakutils.indexes(corrected_signal, thres=Peak_Threshold) 
            
            # Update the df with median and all peaks
            df_peak.loc[Dft.index[i],'Relative_median'] = relative_median
            df_peak.loc[Dft.index[i],'All_Peaks'] = peak_indices
            peak_widths_tuple = peak_widths(corrected_signal, peak_indices)
            left_borders =  np.array(peak_widths_tuple[2:], dtype=float).round(1)[0].tolist() # take the left borders of the peaks
            right_borders = np.array(peak_widths_tuple[2:], dtype=float).round(1)[1].tolist() # take the right borders of the peaks
            df_peak.loc[Dft.index[i], 'Left_borders_all'] = left_borders
            df_peak.loc[Dft.index[i], 'Right_borders_all'] = right_borders
            
            # Cluster identified peaks based on their proximity
            groups = []
            for j in range(len(peak_indices)):
                if j == 0:
                    groups.append([peak_indices[j]])
                elif peak_indices[j] - peak_indices[j-1] <= 10:
                    groups[-1].append(peak_indices[j])
                else:
                    groups.append([peak_indices[j]])
            
            # Select the values with the highest intensity for each group
            max_values_index = [] #take the poistion index of the maximum intensity values in the groups, for after to get the right borders values
            left_border_peaks = []
            right_border_peaks = []
            ix = -1
            for group in groups:
                max_intensity = -1
                max_value = None
                left_b = 0
                right_b = 0
                for value in group:
                    ix += 1
                    intensity = Dft.iloc[i, value]
                    if intensity > (0.9 * max_intensity) and (math.ceil(right_borders[ix]) - math.floor(left_borders[ix])) > (right_b - left_b) :
                        max_intensity = intensity
                        max_value = value
                        left_b = math.floor(left_borders[ix])
                        right_b = math.ceil(right_borders[ix])
                if (right_b - left_b) >= scan_threshold: # only when there are scan_treshold (min 5) consecutive scans to be considered as a peak
                    max_values_index.append(max_value) # indicte in a list the maximum intensities values for each group
                    left_border_peaks.append(left_b - 1) # corresponding left border for each max value, we remove 1 to artificially increase the border value due to lack of precision from scipy peak_width algo
                    right_border_peaks.append(right_b + 1)
            
            # Add to the dataframe
            df_peak.loc[Dft.index[i],'Peaks'] = max_values_index                
            df_peak.loc[Dft.index[i], 'Left_borders_peaks'] = left_border_peaks
            df_peak.loc[Dft.index[i], 'Right_borders_peaks'] = right_border_peaks    
            df_peak.loc[Dft.index[i], 'Peaks_scans'] = self.scan_nb(left_border_peaks, right_border_peaks )
        
        # Update the class variables
        if level == 'MS1':
            self.dfp1 = df_peak
        else:
            self.dfp2 = df_peak

        logging_config.log_info(
            logger,
            'Peak detection complete for %s in %s: %d traces processed, %d traces skipped.',
            level,
            self.filename,
            len(Dft.index) - skipped_traces,
            skipped_traces,
        )
        
        return df_peak 
    

    # Function to display a specific chromatogram:
    def plot_mass(self, Target_mass, Peakarray):  # Peakarray can be self.peakarray1 or self.peakarray1_masked   
        # Initialize an empty list to collect intensities
        intensities = []
        
        # Iterate through peakarray_masked
        for i, masses_and_intensities in enumerate(Peakarray):
            masses = masses_and_intensities[:, 0]
            rounded_masses = np.round(masses, 3)
            target_mass_index = np.where(rounded_masses == np.round(Target_mass, 3))
            
            if target_mass_index[0].size > 0:
                intensity = masses_and_intensities[target_mass_index[0][0], 1]
                intensities.append(intensity)
            else:
                intensities.append(0)
        
        # Create the x-axis values (scan indices)
        x = np.arange(len(intensities))

        # Plot the chromatogram
        plt.plot(x, intensities)
        plt.xlabel('Scan Index')
        plt.ylabel('Intensity')
        plt.title(f'Chromatogram for Mass {Target_mass:.3f} Da')
        plt.show()


    #Function to diplay the detected noises
    def plot_noise(self, Dft, LAM = 10000, P = 0.01, Niter = 10, window_length = 15, poly_order = 10, Threshold = 0.5, Threshold_flat = 0.9):  #Window:length and polyorder are smoothing parameters
        for i in range(len(Dft)):
            data = Dft.iloc[i].replace(0, np.nan).interpolate(method='linear', limit_direction='both') # interpolate() here fill the NaN with values between the 0, meaning it artificially fill in some values
            smooth_signal = savgol_filter(data, window_length, poly_order) # Apply Savitzky-Golay filter to smooth the signal
            
            # Apply baseline correction to remove the overall behavior and noise
            baseline = self.baseline_als(smooth_signal, LAM, P, Niter)
            corrected_signal = smooth_signal - baseline 
            corrected_data = data - baseline 
            
            deflat_data = self.deflat(corrected_data, Threshold_flat) # remove wide range of uniques values to avoid medians bias, Threshold_flat represent a threshold of flatiness
            relative_median = np.median(deflat_data) / max(deflat_data)
           
            # Find the peaks in the baseline-subtracted data
            peak_indices = peakutils.indexes(corrected_signal, thres=Threshold) 
            
            # Proceed to plot
            plt.figure(dpi=500)
            plt.title(f'{round(Dft.index[i], 3)} - {relative_median}')
            plt.plot(data, label='Original Signal', linewidth=0.5)
            plt.plot(smooth_signal, label='smooth_signal', linewidth=0.5)
            plt.plot(baseline, label='baseline', linewidth=0.5)
            plt.plot(corrected_signal, label='Corrected Signal', linewidth=0.5)
            plt.plot(peak_indices, corrected_signal[peak_indices], 'o', label='Detected peaks', markersize=4, markerfacecolor='none', markeredgewidth=0.3)
            plt.legend()
            plt.show() 
            
            
    # Function to plot identified peaks and windows ranges on the raw spectra
    def plot_peak(self, Dft, Dfp): # Dft is for df_trace, dfp is for df_peaks
        for i in range(len(Dft)):
            data = Dft.iloc[i].replace(0, np.nan).interpolate(method='linear', limit_direction='both') # interpolate() here fill the NaN with values between the 0, meaning it artificially fill in some values
            plt.figure(dpi=500)
            relative_median = Dfp.loc[Dfp.index[i], 'Relative_median']
            plt.title(f'{round(Dfp.index[i], 3)} - {relative_median}')
            plt.plot(data, label='Raw Signal', linewidth=0.4)
            left = Dfp.loc[Dfp.index[i], 'Left_borders_peaks']
            right = Dfp.loc[Dfp.index[i], 'Right_borders_peaks']
            for bar in range(len(left)):
                plt.axvspan(left[bar], right[bar], alpha=0.3, color='r') # plot the bar for each peak
            plt.show()
            

import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
