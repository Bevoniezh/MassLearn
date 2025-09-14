# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:06:57 2023

@author: Ronan
"""

import os
import pymzml
import warnings
import numpy as np
import pandas as pd
import Modules.CLI_text as cli
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean

class Match():
    """
    This class take concatenated feature list ms1 + ms2 (..._msn_ann.csv), and evaluate the matching correspondance
    between precursor ions and fragment ions.
    """
    def __init__(self, Msn_df, Mzml_folder):

        self.mzml_path = Mzml_folder
        self.df = Msn_df[~Msn_df['label'].str.contains('Pool|Std|blank')]# Remove pool, standard and blank features
        self.df = self.df.sort_values(by = 'rt', ascending = True) # sort based on rt
        group_counts = self.df['feature_group'].value_counts() # Count the occurrences of each feature_groupc, meaning in how many sampels appears a given feature group
        self.df = self.df[self.df['feature_group'].map(group_counts) > 1] # Filter the DataFrame to exclude groups with only one sample
        
        # Apply the function to each row in the dataframe to get similar_rt values
        self.df['similar_rt'] = self.df.apply(self.find_similar_rt, axis=1)
        
        # Fill NaN values with an empty string
        self.df['similar_rt'].fillna("", inplace=True)
        
        # Grouping matched features
        msms_groups = self.df[self.df['similar_rt'] != ""]
        msms_groups = msms_groups.drop_duplicates(subset=['similar_rt'])
        msms_groups = msms_groups.loc[:,['sample', 'similar_rt']]
        
        # Create the feature_grouping DataFrame
        self.feature_grouping = msms_groups
        
        """
        # Create a new column "spectra" to store the extracted information
        self.df["spectra"] = ['' for i in range(len(self.df))]
        
        # Loop through the unique .mzML files in the "sample" column
        file_list = self.df["mzml_name"].unique()
        bar = cli.LoadingBar(len(file_list)) # set loading bar
        for file_name in file_list:
            file_path = os.path.join(self.mzml_path, file_name + ".mzML")
            with pymzml.run.Reader(file_path) as spectra:
                # Filter the DataFrame for the current .mzML file
                current_file_df = self.df[self.df["mzml_name"] == file_name].copy()
                # Loop through the spectra only once
                for spectrum in spectra:
                    ms_level = spectrum.ms_level
                    rt = spectrum.scan_time_in_minutes()
                    condition = (current_file_df["peak_rt_start"] <= rt) & (current_file_df["peak_rt_end"] >= rt) & (current_file_df["MS_level"] == f'ms{ms_level}')
                    if condition.any():
                        concerned_features = current_file_df.loc[condition, ] # take only the features which have signal in the concerned rt
                        updated_intensities = self.process_spectrum(spectrum, concerned_features)                      
                        self.df['spectra'] = self.df.apply(self.update_spectra, args=(concerned_features, updated_intensities), axis=1) 
            bar.loading()
            
        self.df['spectra'] = self.df.apply(self.clean, axis=1)
        self.df.to_csv('df.csv', index=False)
      
        # Apply the replace missing value function to the "spectra" column
        self.df["spectra"] = self.df["spectra"].apply(lambda x: ";".join(map(str, self.replace_missing_values([float(val) for val in x.split(";")]))))
        
        # Apply the function to the DataFrame
        self.df["peak_rt_adjusted"], self.df["height_adjusted"], self.df["noise_risk"] = zip(*self.df.apply(self.compute_adjusted_values, axis=1, method='poly', poly_degree=5))
        self.df.to_csv('df.csv', index=False)
        """
        
        
        # Pivot the data to have feature_sample as index, sample as columns, and height_adjusted as values
        self.df_pivot = self.df.pivot_table(index='feature_group', columns='sample', values='height')
        self.df_pivot.fillna(0, inplace=True)
        self.df['similar_rt'] = self.df['similar_rt'].fillna('')
        
        # Compute correlations based on similar retention times (RT). This step identifies potential connections between feature groups based on RT.
        correlations = {}
        for feature_group in self.df_pivot.index:
            all_similar_rt = self.df[self.df['feature_group'] == feature_group]['similar_rt'] # take similar rt features, meaning all features with similar rt for a same sample
            similar_features = []
            for similar_rt in all_similar_rt:
                similar_features.extend([int(x) for x in str(similar_rt).split(';') if x != '']) # take similar feature of a same sample
            similar_groups = self.df[self.df['feature_ID'].isin(similar_features)]['feature_group'].unique() # similar_groups is all groups of feature which match at least one of the sample feature from the feature_group
            for similar_group in similar_groups:
                correlations[(feature_group, similar_group)] = self.compute_correlation(feature_group, similar_group)
        self.df_correlations = pd.DataFrame(list(correlations.items()), columns=['Feature_Groups', 'Correlation'])
        
        # Create a matrix to capture correlations between feature groups. This matrix provides a visual representation of the relationships between feature groups.
        self.correlation_df = pd.DataFrame(index=self.df_pivot.index, columns=self.df_pivot.index)
        
        # Populate the matrix with the computed correlations
        for row in self.df_correlations.iterrows():
            feature_group_1, feature_group_2 = row[1]['Feature_Groups'] # row[1] corresponds to df.correlations rows, because in .iterrows() you always have two parameters which iterates, index and row
            self.correlation_df.at[feature_group_1, feature_group_2] = row[1]['Correlation']
        
        # Fill the diagonal of the matrix with 1s. The correlation of a feature group with itself is always 1.
        np.fill_diagonal(self.correlation_df.values, 1)
        
        # Replace any remaining NaN values with zeros. This ensures that the matrix is fully populated and ready for analysis.
        self.correlation_df.fillna(0, inplace=True)        
        self.correlation_df = self.correlation_df.loc[self.df['feature_group'].unique()] # make the correlation sample rows sorted based on the feature rt
        self.correlation_df = self.correlation_df.reindex(columns=self.df['feature_group'].unique()) # make the correlation sample columns sorted based on the feature rt

        
    # Function to compute the correlation between two specific feature groups. This function will be used to compare feature groups that have similar retention times.
    def compute_correlation(self, primary_group, similar_group):
        # Identify common samples between the two feature groups
        overlapping_samples = self.df_pivot.loc[primary_group][self.df_pivot.loc[primary_group] != 0].index.intersection(
                              self.df_pivot.loc[similar_group][self.df_pivot.loc[similar_group] != 0].index) # take only the samples signal which are in both features, because sometime a signal could be missing due to it was not obtain in the feature list because of a too low signal
    
        # Extract intensity values for the overlapping samples
        primary_intensities = self.df_pivot.loc[primary_group, overlapping_samples].values
        similar_intensities = self.df_pivot.loc[similar_group, overlapping_samples].values
        
        # Weight the correlation score with the difference of columns where the features are detected in primary group and similar group
        nb_samples_prim = len(self.df_pivot.loc[[primary_group], (self.df_pivot.loc[primary_group] != 0).values].columns)
        nb_samples_sim = len(self.df_pivot.loc[[similar_group], (self.df_pivot.loc[similar_group] != 0).values].columns)
        
        ratio = min([nb_samples_sim, nb_samples_prim])/max([nb_samples_sim, nb_samples_prim])
        if ratio > 0.66:
            weight = 'A'
        elif 0.33 < ratio <= 0.66:
            weight = 'B'
        else:
            weight = 'C'
        
        # Calculate the correlation for the overlapping samples
        if len(overlapping_samples) <= 1:
            return np.nan
        else:
            return weight + str(int(round(np.corrcoef(primary_intensities, similar_intensities)[0, 1], 2)*100))


    # Function to find feature with similar rt
    def find_similar_rt(self, Row, Threshold=0.05):
        # Filter for features within the same sample and MS level
        mask = (self.df['sample'] == Row['sample'])
        
        # Find features with similar rt
        similar_features = self.df[mask & (self.df['rt'].between(Row['rt'] - Threshold, Row['rt'] + Threshold))]
    
        # If there are similar features, sort them by m/z and return their feature_IDs as a string
        if len(similar_features) > 1:
            sorted_ids = similar_features.sort_values(by='m/z', ascending=False)['feature_ID'].tolist()
            return ';'.join(map(str, sorted_ids))
        return ""
    

    # Function to plot spectra. Need to have have all column from the other funcitons
    def plot_spectra_with_baseline(self, df, index, method='poly', poly_degree=3, num_points=100):
        """
        Plot the spectra along with a baseline for a given row in the DataFrame.
    
        Args:
        - df (DataFrame): The DataFrame containing the spectra data.
        - index (int): The index of the row to plot.
        - method (str): Method to compute the baseline. Options are 'poly' or 'savgol'.
        - poly_degree (int): Degree of the polynomial (used only if method='poly').
        """
        # Extract necessary information from the DataFrame
        spectra_str = df.loc[index, "spectra"]
        spectra = [float(intensity) for intensity in spectra_str.split(';')]
        rt_start = df.loc[index, "peak_rt_start"]
        rt_end = df.loc[index, "peak_rt_end"]
        mz = df.loc[index, "m/z"]
        peak_rt = df.loc[index, "peak_rt"]
        sample_name = df.loc[index, "sample"]
        feature_id = df.loc[index, "feature_ID"]
        height = df.loc[index, "height"]
        height_adjusted = df.loc[index, "height_adjusted"]
        
        # Original array of RT values
        rt_values_original = [rt_start + i * (rt_end - rt_start) / len(spectra) for i in range(len(spectra))]
        
        # Convert rt_values to a linspace of 100 values
        rt_values_linspace = np.linspace(rt_start, rt_end, num_points)
        
        # Interpolate the spectra data to match the linspace
        interpolator = interp1d(rt_values_original, spectra, kind='linear', bounds_error=False, fill_value='extrapolate')
        spectra_interpolated = interpolator(rt_values_linspace)
        
        # Compute the baseline
        if method == 'poly':
            coefficients = np.polyfit(rt_values_linspace, spectra_interpolated, poly_degree)
            baseline = np.polyval(coefficients, rt_values_linspace)
        elif method == 'savgol':
            baseline = savgol_filter(spectra, window_length=5, polyorder=2)
        else:
            raise ValueError("Invalid method. Choose 'poly' or 'savgol'.")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(rt_values_original, spectra, '-o', markersize=4, label='Spectra')
        plt.plot(rt_values_linspace, baseline, '--', color='red', label='Baseline')
        plt.axvline(peak_rt, color='green', linestyle='--', label='Peak RT')
        plt.title(f"Sample: {sample_name} | Feature ID: {feature_id}")
        plt.suptitle(f"Detected m/z: {mz}  -  Feature ID : {df.loc[index, 'feature_ID']} - height/adjuted:{height} / {height_adjusted}")
        plt.xlabel("RT (min)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    # Function to have adjusted values of height based on the baseline of the feature signals
    def compute_adjusted_values(self, row, method='poly', poly_degree=3, num_points=100):
        """
        Compute the adjusted RT, height values, and low quality risk based on the smoothed and vectorized baseline 
        for a given row in the DataFrame.

        Args:
        - row (Series): A row from the DataFrame.
        - method (str): Method to compute the baseline. Options are 'poly' or 'savgol'.
        - poly_degree (int): Degree of the polynomial (used only if method='poly').
        - num_points (int): Number of data points for vectorization.

        Returns:
        - tuple: (peak_rt_adjusted, height_adjusted, noise_risk)
        """
        spectra_str = row["spectra"]
        spectra = [float(intensity) for intensity in spectra_str.split(';')]
        rt_start = row["peak_rt_start"]
        rt_end = row["peak_rt_end"]
        
        # Original array of RT values
        rt_values_original = [rt_start + i * (rt_end - rt_start) / len(spectra) for i in range(len(spectra))]
        
        # Convert rt_values to a linspace of 100 values
        rt_values_linspace = np.linspace(rt_start, rt_end, num_points)
        
        # Interpolate the spectra data to match the linspace
        interpolator = interp1d(rt_values_original, spectra, kind='linear', bounds_error=False, fill_value='extrapolate')
        spectra_interpolated = interpolator(rt_values_linspace)
        
        # Compute the baseline on interpolated data
        # Suppress the warning
        warnings.filterwarnings('ignore', 'Polyfit may be poorly conditioned*', np.RankWarning)

        if method == 'poly':
            coefficients = np.polyfit(rt_values_linspace, spectra_interpolated, poly_degree)
            baseline = np.polyval(coefficients, rt_values_linspace)
        elif method == 'savgol':
            baseline = savgol_filter(spectra_interpolated, window_length=5, polyorder=2)
        else:
            raise ValueError("Invalid method. Choose 'poly' or 'savgol'.")
        
        # Identify the RT and height values for the maximum point on the baseline
        index_of_max = np.argmax(baseline)
        peak_rt_adjusted = rt_values_linspace[index_of_max]
        height_adjusted = baseline[index_of_max]
        
        # Determine if the peak is in the first 10% or last 10% of the RT range
        first_10_percent = rt_start + 0.1 * (rt_end - rt_start)
        last_10_percent = rt_end - 0.1 * (rt_end - rt_start)
        noise_risk = peak_rt_adjusted <= first_10_percent or peak_rt_adjusted >= last_10_percent
        
        return peak_rt_adjusted, height_adjusted, noise_risk

    

    # Function to process a single spectrum and update the intensities for matching features.
    def process_spectrum(self, Spectrum, Features_df):
        """
        Process a single spectrum and update the intensities for matching features.

        Args:
        - spectrum: Current MS spectrum from pymzml.
        - features_df (DataFrame): Subset of the main DataFrame corresponding to the current .mzML file.

        Returns:
        - Updated intensities for the matching features.
        """
        intensities_list = []

        # Loop through the features (rows) for the current .mzML file
        for _, row in Features_df.iterrows():
            # Check if the MS level of the spectrum matches the feature's MS level
            if Spectrum.ms_level == int(row["MS_level"][-1]):
                # Check if the spectrum is within the rt range of the current feature
                if row["peak_rt_start"] <= Spectrum.scan_time_in_minutes() <= row["peak_rt_end"]:
                    # Filter peaks within the m/z range and get the maximum intensity
                    peaks = np.column_stack((Spectrum.mz, Spectrum.i))
                    # Use boolean indexing to filter the peaks within the desired m/z range
                    peaks_within_range = peaks[(peaks[:, 0] >= row["peak_mz_min"]) & (peaks[:, 0] <= row["peak_mz_max"])]
                    if peaks_within_range.any():
                        highest_intensity = np.max(peaks_within_range[:, 1]) # Retrieve the highest intensity from the filtered peaks
                        intensities_list.append(highest_intensity)
                    else:
                        intensities_list.append(0)
                else:
                    intensities_list.append(None)
            else:
                intensities_list.append(None)

        return intensities_list
    
    # Function to replace the missing value from low abundant spectra
    def replace_missing_values(self, Spectra_list):
        """
        Replace missing values (0s) in the spectra list with the average of their adjacent values.

        Args:
        - spectra_list (list): List of spectra intensities.

        Returns:
        - list: Modified list with missing values replaced.
        """
        # Check if the list is not empty and has more than 2 elements
        if not Spectra_list or len(Spectra_list) <= 2:
            return Spectra_list
        
        # Create a copy of the spectra list
        modified_spectra = Spectra_list.copy()
        
        # Iterate over the list, skipping the first and last elements
        for i in range(1, len(Spectra_list) - 1):
            if Spectra_list[i] == 0:
                modified_spectra[i] = (Spectra_list[i - 1] + Spectra_list[i + 1]) / 2
        
        return modified_spectra

    
    # Function to add intensities values in the spectra for each concerned feature
    def update_spectra(self, Row, Concerned_features, Intensities):
        index = Row.name
        if index in Concerned_features.index and Intensities[Concerned_features.index.get_loc(index)] is not None:
            return Row['spectra'] + str(Intensities[Concerned_features.index.get_loc(index)]) + ';'
        else:
            return Row['spectra']
    
    # Function to remove the last ';' from the spectra
    def clean(self, Row):
        return Row['spectra'][:-1]


   

