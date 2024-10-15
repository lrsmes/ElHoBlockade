#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:09:52 2023

@author: lucca
"""

# Library for CD data analysis 

import h5py
import numpy as np


# ############################################################################
# ------------------------ Functions to read data ----------------------------
# ############################################################################        

# function to read file without time traces, data is stored in group "Data" in hdf file
def read_file(file_path, group, dataset, channels, information):
    with h5py.File(file_path, 'r') as file:
        channel_data_lists = {}  # Creating an empty dictionary for measurement data
        group_name = group 
        dataset_name = dataset  
        channel_names = channels
        
        if information:
            def print_name(name):
                print(name)

            # Use the visit method to traverse all groups and datasets
            print("\n Groups and datasets in the HDF5 file:")
            file.visit(print_name)

            # List all datasets in the file
            dataset_names = list(file.keys())
            print("\n Datasets in the HDF5 file:", dataset_names)
    
            # Print attributes of the group
            for attr_name, attr_value in file[group].attrs.items():
                print(f"\n Attribute: {attr_name} = {attr_value}")
            
        else:
            if group_name in file:
                group = file[group_name]
                data_dataset = file[dataset_name]
                channel_names_dataset = file[channel_names]
                
                # Initialize an empty list to store channel names
                channel_names = []

                # Iterate through the dataset and access the elements of each tuple
                for item in channel_names_dataset:
                    channel_name = item[0]  # Access the first element of the tuple
                    channel_names.append(channel_name)

                channel_names = [channel_name.decode('utf-8') for channel_name in channel_names]

                # Convert the dataset to a NumPy array
                data = data_dataset[:]

                # Create lists with channel names containing the corresponding data
                for i, channel_name in enumerate(channel_names):
                    channel_data = data[:, i, :]  # Extract data for the current channel
                    channel_data_lists[channel_name] = np.array(channel_data.tolist())

                # Dictionary where keys are channel names, and values are lists of corresponding data
                for channel_name, channel_data in channel_data_lists.items():
                    print(f"Channel Name: {channel_name}")
                    print('Shape of data: ')
                    print(np.shape(channel_data))
                    print()
                    
    return channel_data_lists

# function to read data containing time traces. Data is stored in group "Traces" in hdf file
def read_dual_pulse_file(file_path, group, dataset, time_spacing, measurement_time, n_points, information):
    with h5py.File(file_path, 'r') as file:
        group_name = group 
        dataset_name = dataset  
        time_spacing_name = time_spacing
        
        if group_name in file:
            print('-----------------group_name')
            group = file[group_name]
                
        if dataset_name in file and time_spacing_name in file: 
            print('-----------------dataset name')
            data = file[dataset_name][:]
            time_spacing = file[time_spacing_name][:]
            
            n_traces = len(data[0][0])   
        
        if information:
            def print_name(name):
                print(name)

            # Use the visit method to traverse all groups and datasets
            print("Groups and datasets in the HDF5 file:")
            file.visit(print_name)

            # List all datasets in the file
            dataset_names = list(file.keys())
            print("Datasets in the HDF5 file:", dataset_names)
    
            # Print attributes of the group
            for attr_name, attr_value in group.attrs.items():
                print(f"Attribute: {attr_name} = {attr_value}")
        
            time_array, traces_array = [], []
            
            print("t0 - dt:")
            print([file['Traces/UHFLI - TraceDemod1R_t0dt'][i] for i in range(n_traces)])
            
        else:
            
            time_spacing_list = []
            traces_list = []
                 
            for i in range(n_traces):
                traces_list.append([]) # adding n_traces empty lists to array, which will be populated with data
                time_spacing_list.append(time_spacing[i][1]) # Sometimes time spacing is the same for all traces, then i=0 for all traces
                #time_spacing_list.append(time_spacing[0][1]) 
                       
            # Extract different traces from data
            for i in data:
                val = i[0]
                for j in range(n_traces):
                    t_val = val[j]
                    if np.isnan(t_val) == False: 
                        traces_list[j].append(t_val)
                    else: 
                        #print("Is nan")
                        continue
                    # If there's a problem with the data, you can filter data here
                    
                # if t_val >= 0.004:
                #     traces_list[j].append(t_val)
                # else: 
                #     traces_list[j].append(0.00574) # Hilfswert, da die Messung hier kaputt war
                
            traces_array = np.array(traces_list)

            # Create time array according to determined spacing
            #time_array = [np.linspace(0, t*n_points, n_points) for t in time_spacing_list]
            time_array = [np.linspace(0, t*len(traces_array[i]), len(traces_array[i])) for t,i in zip(time_spacing_list, range(n_traces))]
            #print("time:")
            #print([len(traces_array[i]) for i in range(n_traces)])
            #print([max(t) for t in time_array])
               
            averaged_data = [np.mean(x) for x in zip(*traces_array)]
            averaged_time = np.linspace(0,np.mean(time_spacing_list)*len(averaged_data), len(averaged_data))
                
            #plt.figure()
            #plt.plot(averaged_time, averaged_data)
    
    average = False
    
    if average == True:
        return time_array, traces_array,  averaged_time, averaged_data
    else: 
        return time_array, traces_array

