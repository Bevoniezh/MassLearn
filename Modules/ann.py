# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:35:18 2023

@author: Ronan
"""
import os
import threading
import numpy as np
import pandas as pd
import NeatMS as ntms
from collections import Counter


Featurelists = ['./Feature_list/features.csv','./Feature_list/features1.csv', './Feature_list/features2.csv']

"""
comb = ann.FeatureCombiner(Featurelists) #create object and first feature list
comb.combining() # combine all feature list index, feature_goup and rows, not values!!
comb.quantitative() # get the average of quantitative variables
c = comb.combined.copy() # if you want to have a copy of self.combined variable, use copy() to have a fixed one, otherwise variable will point to hte last updated version of self.combined
comb.categorical() # to get categorical variable most abunt or relevant
comb.adjust_neatms() # to have comb.combined_neatms created
comb.combined_neatms.to_csv(YOURNAME.CSV)

ANN = ann.QualityFilter('ft.csv', '../Test_files/Denoised/')
ANN.neatms() # to use neatms
neatms_output = ANN.neatms_output # to generate the feature list output

"""

ft = os.path.join('..', '..', 'ANN', 'ft_neatms.csv')
output = os.path.join('..', '..', 'ANN', 'neatms_export_with_extra_properties.csv')

class FeatureCombiner():
    def __init__(self, Featurelists):
        self.list = []
        for file in Featurelists:
             df = pd.read_csv(os.path.abspath(file), low_memory=False)
             self.list.append(df) # self.list ontain all features lists as df  
        self.combined = self.list[0].loc[self.list[0]['charge'] != 2, :] # it is the final df which combine all feature lists, begin with the first feature list present in self.list. We remove all charge == 2 from the list because it make it more difficut to combine the feature list (risk of errors) 
        self.samples = []
        for variable in self.combined.columns.to_list():
            if 'datafile' in variable and (variable.split(':')[1] not in self.samples): # variable.split(':')[1] is the sample name 
                self.samples.append(variable.split(':')[1]) # add sample to sample list

        self.quantitative_variables = ['area', 'rt', 'mz_range:min', 'mz_range:max', 'alignment_scores:weighted_distance_score',
                          'alignment_scores:mz_diff_ppm', 'alignment_scores:mz_diff', 'alignment_scores:rt_absolute_error',
                          'rt_range:min', 'rt_range:max', 'mz', 'intensity_range:min', 'intensity_range:max',
                          'height', 'ion_identities:iin_relationship', 'ion_identities:consensus_formulas', 
                          'ion_identities:simple_formulas', 'datafile:sample:area', 'datafile:sample:rt',
                          'datafile:sample:mz_range:min', 'datafile:sample:mz_range:max',
                          'datafile:sample:fwhm', 'datafile:sample:rt_range:min', 'datafile:sample:rt_range:max',
                          'datafile:sample:mz', 'datafile:sample:intensity_range:min',
                          'datafile:sample:intensity_range:max', 'datafile:sample:asymmetry_factor', 'datafile:sample:isotopes',
                          'datafile:sample:tailing_factor', 'datafile:sample:height']
        to_remove = [] 
        to_add = []
        for var in self.quantitative_variables:
            if 'datafile' in var:
                to_remove.append(var) # remove sample concerning variable which have only 'sample' as name, it was temporary
                for sample in self.samples:
                    var = var.split(':')
                    var[1] = sample # put the sample name instead of 'sample' like in datafile:sample:intensity_range:max
                    var = ':'.join(var)
                    to_add.append(var) # add the existing variable name for each sample in the list
        self.quantitative_variables = [x for x in self.quantitative_variables if x not in to_remove]    
        self.quantitative_variables += to_add
        
        self.categorical_variables = ['alignment_scores:rate', 'alignment_scores:aligned_features_n', 'alignment_scores:align_extra_features',
                                      'ion_identities:consensus_formulas', 'ion_identities:simple_formulas',
                                      'datafile:sample:feature_state'
                                      ]
        to_remove = [] 
        to_add = []
        for var in self.categorical_variables:
            if 'datafile' in var:
                to_remove.append(var) # remove sample concerning variable which have only 'sample' as name, it was temporary
                for sample in self.samples:
                    var = var.split(':')
                    var[1] = sample # put the sample name instead of 'sample' like in datafile:sample:intensity_range:max
                    var = ':'.join(var)
                    to_add.append(var) # add the existing variable name for each sample in the list
        self.categorical_variables = [x for x in self.categorical_variables if x not in to_remove]    
        self.categorical_variables += to_add
        
        self.special_variables = ['id', 'charge', 'feature_group', 'ion_identities:iin_id', 'ion_identities:ion_identities',
                                  'ion_identities:list_size', 'ion_identities:neutral_mass', 'ion_identities:partner_row_ids',
                                  'ion_identities:iin_relationship'] # they are variables which are combined in other ways than other quantitative and categorical variables
        
        self.combined_neatms = self.combined # it is the feature list with column filterd to be usable in neatms
        
    # Function to perform the addition of values in variables for each feature      
    def adding(self, Feature, Idx, Row_idx):        
        for var in self.quantitative_variables:            
            combined_value = self.combined.loc[Idx, var]
            feature_value = Feature.loc[Row_idx, var]
            self.combined.loc[Idx, var] = f"{combined_value};{feature_value}"
        for var in self.categorical_variables:            
            combined_value = self.combined.loc[Idx, var]
            feature_value = Feature.loc[Row_idx, var]
            self.combined.loc[Idx, var] = f"{combined_value};{feature_value}"
                
    # Function to adjust the columns for neatms input
    def adjust_neatms(self):
        self.combined_neatms = self.combined.copy()
        self.combined_neatms = self.combined_neatms.rename(columns={'rt': 'row retention time'})
        self.combined_neatms = self.combined_neatms.rename(columns={'mz': 'row mz'})
        for var in self.combined_neatms.columns.to_list():
            splited_var = var.split(':')
            if len(splited_var) > 1:
                samplename = splited_var[1] # it is really the sample name only for the variable containing sample names, meaning all the one beginning with datafile:...
            if 'datafile' in var and splited_var[2] == 'mz':                
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak m/z'})
            elif 'datafile' in var and splited_var[2] == 'rt':
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak RT'})
            elif 'datafile' and ':rt_range:min' in var:
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak RT start'})
            elif 'datafile' and ':rt_range:max' in var: 
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak RT end'})    
            elif 'datafile' and ':intensity_range:max' in var: 
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak height'})
            elif 'datafile' and ':area' in var: 
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak area'})
            elif 'datafile' and ':mz_range:min' in var: 
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak m/z min'})
            elif 'datafile' and ':mz_range:max' in var: 
                self.combined_neatms = self.combined_neatms.rename(columns={var: f'{samplename} Peak m/z max'})
            else:
                if var != 'row retention time' and var != 'row mz' :
                    self.combined_neatms = self.combined_neatms.drop(var, axis=1)              
        
    # Function to make the average of the values after self.adding()
    def average(self, cell):
        floats = [float(x) for x in str(cell).split(';') if x.lower() != 'nan']    
        return np.mean(floats) if floats else np.nan
    
    
    
    
    # Function to apply the average of all the features lists quantitative variables
    def quantitative(self): # TODO there is a bug when htere is no identity ion detected, resulting in a key error because ion_identities_iin_if and other related variable are not found in the feature list. Make here an expetion in this case.
        self.combined[self.quantitative_variables] = self.combined[self.quantitative_variables].applymap(self.average)
        for var in self.quantitative_variables:
            if var == 'rt' or 'rt_range' in var or ':rt' in var or ':fwhm' in var: # for rt related variables
                self.combined[var] = self.combined[var].round(3) 
            elif var == 'mz' or 'mz_range' in var or ':mz' in var: # for mz related variables
                self.combined[var] = self.combined[var].round(4)
            else:
                self.combined[var] = self.combined[var].round(0) # for all other variables


    # Function to get the most abundant of categories
    def categories_most_abundant(self, Categorical_variables, Rank_order):
        Categorical_variables = str(Categorical_variables).split(';')        
        if self.is_float(Categorical_variables[0]): # if there are float values, covnersion is done in float
            Categorical_variables = [float(x) for x in Categorical_variables]        
        counter = Counter(Categorical_variables)
        max_count = max(counter.values())
        most_abundant = [category for category, count in counter.items() if count == max_count]
        for category in Rank_order:
            if category in most_abundant:
                return category # Return the most important category based on the given rank
            
    def categorical_unique_values(self, string):
        return [float(x) for x in set(str(string).split(';'))]
    
    # Function to chose the most abundant/relevant categorical variable after used self.adding
    def categorical(self):
        for var in self.categorical_variables:
            if 'feature_state' in var:   
                self.combined[var] = self.combined[var].apply(lambda x: self.categories_most_abundant(x, ['DETECTED', 'ESTIMATED', 'UNKNOWN'])) # If there is an equivalent abundance of two categorical variable, we retain only the most important one in the rank order DETECTED > ESTIMATED > UNKNOWN
            elif 'formula' not in var:
                unique_values = self.combined[var].apply(self.categorical_unique_values) # get all uniue values from the categorical variable
                all_unique_values = set().union(*unique_values)
                rank = sorted(all_unique_values, reverse=True) # rank all the unique values in descending order (because those variables are float and importance is linked to higher values)
                self.combined[var] = self.combined[var].apply(lambda x: self.categories_most_abundant(x, rank))
            else:
                pass
            
    # Function to combine the features list to the combination of feature list (self.combined)
    def combining(self):
        for features in self.list[1:]: # iterate over the feature list from the second, because the first was taken as basis for self.combined
            features_reduced = features.loc[features['charge'] != 2] # remove charge = 2 rows
            features_reduced.index = range(len(features_reduced))
            for row in range(len(features_reduced)): 
                idx, charge = self.match_feature(features_reduced, row) # check if there is already the feature in self.combined
                if idx != 'Null': # if feature already exists
                    self.combined.loc[idx, 'charge'] = charge #update the charge value if necessary 
                    feature_group = features_reduced.loc[row,'feature_group'] # feature row corresponding group (if exist)
                    combined_group = self.combined.loc[idx,'feature_group'] # equivalent from self.combined
                    if not np.isnan(feature_group) and not np.isnan(combined_group): # if both are already assigned to groups it changes nothing
                        self.adding(features_reduced, idx, row) 
                    elif np.isnan(feature_group) and not np.isnan(combined_group): # if there is a group in self.combined but not in the new feature it changes nothing
                        self.adding(features_reduced, idx, row) 
                    elif not np.isnan(feature_group) and np.isnan(combined_group): # feature is assigned to a group in its feature list and not in the combined group we have new information to update   
                        self.combined.loc[idx,'feature_group'] = max(self.combined['feature_group']) + 1 # group number will be the next integer coming in a increasing rank
                        self.adding(features_reduced, idx, row) 
                    else: # if both are not already assigned to groups it changes nothing
                        self.adding(features_reduced, idx, row) 
                else:
                    self.new_row(features_reduced, row)  
                    
    # Function to test in an object is convertible to a float:
    def is_float(self, Obj):
        try:
            float(Obj)
            return True
        except ValueError:
            return False
            
    # Function to get the index in combined dataframe from a corresponding external feature in a feature list, or create a new line of the combined dataframe                
    def match_feature(self, Feature, Row_idx):
        rt = Feature.loc[Row_idx, 'rt']     
        mz = Feature.loc[Row_idx, 'mz'] 
        charge =  Feature.loc[Row_idx, 'charge']
        for idx in range(len(self.combined)):
            RT = self.combined.loc[idx, 'rt']
            RT = [float(x) for x in str(RT).split(';')] # take all values RT to make the average
            RT = sum(RT) / len(RT) # # take the average value of RT
            MZ = self.combined.loc[idx, 'mz']
            MZ = [float(x) for x in str(MZ).split(';')] # take all values MZ to make the average
            MZ = sum(MZ) / len(MZ) # # take the average value of MZ          
            if (rt - 0.03) <= RT <= (rt + 0.03) and (mz - 0.003) <= MZ <= (mz + 0.003): # 0.02 min and 0.002 Da are estimated to be suitable values to check siilarities for the same feature coming from different batch
                return idx, charge                    
                
        # if there is no match, it is probably a new feature which is still unknown in the combined feature list and we return None
        return 'Null', 'Null'
    
    # Function to get the index in combined dataframe from the corresponding group that a new feature row should match with
    def match_group(self, Feature, Row_idx):
        ori_group = Feature.loc[Row_idx, 'feature_group']    # get the original group feature member from the feature list
        group_feature = Feature.loc[Feature['feature_group'] == ori_group, :]  # take only the features from the group              
        for i in group_feature.index:
            idx, charge = self.match_feature(group_feature, i)
            if idx != 'Null' and charge < 2: # if the feature is already included in a group in self.combined
                group = self.combined.loc[idx, 'feature_group'] # take the corresponding group number
                return group
            else:
                return 'Null'  
            
    # Function to create a new row in the combined feature list
    def new_row(self, Feature, Row_idx): # Row_idx is the row unknown from Features in self.combined
        self.combined = self.combined.append(Feature.iloc[Row_idx], ignore_index=True)
        if not np.isnan(self.combined.loc[len(self.combined)-1, 'feature_group']): # if the new row is already included in a group (it can be a group hich exist in self.commbined or a total new group)
            group_id = self.match_group(Feature, Row_idx) 
            if group_id != 'Null': # If the group already exist in self.combined
                self.combined.loc[len(self.combined)-1,'feature_group'] = group_id
            else: # if the group is totally new, it create a new group number
                new_group_id = max(self.combined['feature_group']) + 1 # group number will be the next integer coming in a increasing rank
                self.combined.loc[len(self.combined)-1,'feature_group'] = new_group_id                
        else: # if the new row is not included in a group
            pass
           
    # Function to add the no_label tag to features without a ANN tag. If NeatMS failed, it gives it to all features
    def nolabel_output(self, ANN_failure = False, Files = None, Ann_fl = None):
        if ANN_failure == True:        
            ann_df = pd.DataFrame({'feature ID':[], 'sample':[], 'm/z':[], 'rt':[], 'height':[], 'area':[], 'label':[], 'peak_rt':[], 'peak_rt_start':[], 'peak_rt_end':[], 'peak_mz':[], 'peak_mz_min':[], 'peak_mz_max':[]})
            f_id = 0
        else:
            ann_df = pd.read_csv(Ann_fl)
        for ix, row in self.combined_neatms.iterrows():
            rt = row['row retention time']
            mz = row['row mz']
            if ANN_failure == True:
                f_id += 1
            else:
                mask = (ann_df['m/z'] == mz) & (ann_df['rt'] == rt) # to test if rt and mz match a feature in the ann feature list
                if mask.any():
                    f_id = int(ann_df.loc[(ann_df['m/z'] == mz) & (ann_df['rt'] == rt), 'feature ID'].unique()[0])
                else:
                    f_id = ann_df['feature ID'].max() + 1   
            filenames = [os.path.basename(f) for f in Files]  # take all file names without .mzML extension      
            for fi in filenames: # take all file names without .mzML extension
                sample = os.path.splitext(fi)[0]
                area = row[f'{fi} Peak area']
                height = row[f'{fi} Peak height']
                rt_s = row[f'{fi} Peak RT start']
                rt_e = row[f'{fi} Peak RT end']
                p_rt = row[f'{fi} Peak RT']
                mz_min = row[f'{fi} Peak m/z min']
                mz_max = row[f'{fi} Peak m/z max']
                p_mz = row[f'{fi} Peak m/z']
                if not(np.isnan(row[f'{fi} Peak area'])): # if the feature exist
                    if sample not in ann_df.loc[(ann_df['m/z'] == mz) & (ann_df['rt'] == rt), 'sample'].to_list(): # if the sample name is not present in ann df
                        row_to_add = pd.DataFrame({'feature ID':[f_id],
                                               'sample':[sample],
                                               'm/z':[mz],
                                               'rt':[rt],
                                               'height':[height],
                                               'area':[area],
                                               'label':['no_label'],
                                               'peak_rt':[p_rt],
                                               'peak_rt_start':[rt_s],
                                               'peak_rt_end':[rt_e],
                                               'peak_mz':[p_mz],
                                               'peak_mz_min':[mz_min],
                                               'peak_mz_max':[mz_max]
                                               })
                        ann_df = pd.concat([ann_df, row_to_add], ignore_index=True) # add the new row to the df
        ann_df.to_csv(Ann_fl, index=False)
        


class QualityFilter():
    """
    Class to manage Neatms software and create an ouput list of the feature
    with their quality based on pre-trained ANN model
    Then, filter the feature_list input based on quality results from ANN model
    """
    
    def __init__(self, Combined_adjusted_featurelist_path, Mzml_folder_path):
        self.neatms_featurelistpath = Combined_adjusted_featurelist_path
        self.neatms_featurelist = pd.read_csv(Combined_adjusted_featurelist_path)
        self.mzml = Mzml_folder_path
        self.neatms_output = None
        
    def start_run(self, Log):
        # This method starts the run method in a new thread
        thread = threading.Thread(target=self.neatms_dashapp, args=(Log,))
        thread.start()
        return thread
     
    # Function to get a quality tag for each feature based on a pre-trained ANN model from NeatMS software
    def neatms(self, Model_path = "./Cache/neatms_default_model.h5"): 
        raw_data_folder_path = self.mzml 
        feature_table_path = self.neatms_featurelistpath # Using peaks that have been aligned across samples
        input_data = 'mzmine' #neatms takes mzmine files as input, because it cans also takes XCMS 
        experiment = ntms.Experiment(raw_data_folder_path, feature_table_path, input_data)
        for sample in experiment.samples:
            print('Sample {} : {} peaks'.format(sample.name,len(sample.feature_list))) #data exploration
        nn_handler = ntms.NN_handler(experiment)
        # Here we use the first default base model from NeatMS
        model_path = Model_path
        nn_handler.create_model(model = model_path)
        # Set the threshold to 0.22
        threshold=0.22
        # Run the prediction
        nn_handler.predict_peaks(threshold)
        # Default properties will be overwritten, so make sure to add them to the list as well
        export_properties = ["mz", "rt",  "height", "area", "label", "peak_rt", "peak_rt_start", "peak_rt_end", "peak_mz", "peak_mz_min", "peak_mz_max"]
        NeatMS_output = experiment.export_to_dataframe(export_properties = export_properties)
        NeatMS_output = NeatMS_output.rename(columns={'retention time': 'rt'})
        NeatMS_output.rename(columns={'rt': 'temp', 'm/z': 'rt'}, inplace=True)
        NeatMS_output.rename(columns={'temp': 'm/z'}, inplace=True)
        self.neatms_output = NeatMS_output

    def neatms_dashapp(self, Model_path = "./Cache/neatms_default_model.h5"): 
        raw_data_folder_path = self.mzml 
        feature_table_path = self.neatms_featurelistpath # Using peaks that have been aligned across samples
        input_data = 'mzmine' #neatms takes mzmine files as input, because it cans also takes XCMS 
        experiment = ntms.Experiment(raw_data_folder_path, feature_table_path, input_data)
        for sample in experiment.samples:
            print('Sample {} : {} peaks'.format(sample.name,len(sample.feature_list))) #data exploration
        try:
            nn_handler = ntms.NN_handler(experiment)
            # Here we use the first default base model from NeatMS
            model_path = Model_path
            nn_handler.create_model(model = model_path)
            # Set the threshold to 0.22
            threshold=0.22
            # Run the prediction
            nn_handler.predict_peaks(threshold)
            # Default properties will be overwritten, so make sure to add them to the list as well
            export_properties = ["mz", "rt",  "height", "area", "label", "peak_rt", "peak_rt_start", "peak_rt_end", "peak_mz", "peak_mz_min", "peak_mz_max"]
            NeatMS_output = experiment.export_to_dataframe(export_properties = export_properties)
            NeatMS_output = NeatMS_output.rename(columns={'retention time': 'rt'})
            NeatMS_output.rename(columns={'rt': 'temp', 'm/z': 'rt'}, inplace=True)
            NeatMS_output.rename(columns={'temp': 'm/z'}, inplace=True)
            self.neatms_output = NeatMS_output
            print('success')
        except Exception as e:
            self.neatms_output = 'Fail'
        

    # Function to adjust mz
    def adjust_mz(self):
        # We have a problem because NeatMS is not able to provide satisfying output, indeed mz for each feature are not indicated in the ouput, it is replaced by an index number (we dont know why)
        # We decide to consider an approach when first rt between the original non filtered feature list and output, filtered have same rt indication
        # But for a same rt, it can have multiple m/z. For this we have to consider a feature from the output to be from the original feature list when the mass is the closet from all the different mass from the original feature list with a same rt
        unique_rt = self.neatms_featurelist['row retention time'].unique()
        count = 0
        for rt in unique_rt:
            if count %10 == 0:
                print(f'{count} / {len(unique_rt)}')
            count += 1
            rt_filter = self.neatms_featurelist.loc[self.neatms_featurelist['row retention time'] == rt, :]
            masslist_ori = rt_filter['row mz'].tolist()
            if len(masslist_ori) > 1:                  
                masslist_output = self.neatms_output.loc[self.neatms_output['rt'] == rt,'peak_mz'].tolist()
                for mass in masslist_output:
                    closest_value = min(masslist_ori, key=lambda x: abs(x - mass)) # get the closest value in masslist origin, meaning the real mass from the original feature list
                    self.neatms_output.loc[(self.neatms_output['rt'] == rt) & (self.neatms_output['peak_mz'] == mass), 'm/z'] = closest_value 
            else:
                mass = masslist_ori[0]
                self.neatms_output.loc[self.neatms_output['rt'] == rt, 'm/z'] = mass
        
