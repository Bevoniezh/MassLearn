# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:26:13 2023

@author: Ronan
"""
import tkinter as tk 
from tkinter import ttk 
import tkinter.messagebox
import os
import pandas as pd

path = r"C:\Users\Ronam\Desktop\Working_folder\Bioinformatic_projects\Molecular Network project\MassLearn\Cache\All_standards.csv"

class GroupingTool():
    """
    Class for grouping tool. Here samples from a dataframe can be easily labelled. 
    The labels will be used in the PCA tool to color samples.

    # Assuming we have the following DataFrame.
    tool = GroupingTool(df, df_name, export_path)  # Creating an instance of the GroupingTool class.
    tool.run()  # Running the GroupingTool instance.


    """
    
    def __init__(self, Table, Featurelistname, Export_path, Type = 'label'): # Type is a parameter to define the grouping tool to be used as a labeller or to group based on blanks
        self.name = Featurelistname
        self.all_std = pd.read_csv(path) # TODO refers the standard here to hte All_standard file
        self.type = Type
        self.treatment = {} # dictionnary for all treatment labels
        if self.type == 'label':
            self.df = Table # Msn df
            self.df['label'] = ""  # Adding a new column 'label' to the dataframe corresponding to the treatment/control labels, will be color in the PCA
            self.df.loc[self.df['blank'] == True,'label'] = 'blank'
        elif self.type == 'blank':
            self.df = pd.DataFrame({'sample':Table.columns.to_list(), 'blank':""}) # Table is a
        elif self.type == 'std':
            self.df = Table
        
        self.scale = 20 # It is the the threshold in % for the blank detection   
        self.std_counter = [i for i in range(1, 100)]
        self.pool_counter = [i for i in range(1, 100)]
        self.export_path = Export_path  # Path to export CSV.
        self.label_boxes = []  # List to hold the new label boxes.
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.destroy = False
        if self.type == 'label':
            self.root.title("Attribute labels to samples")
        elif self.type == 'blank':
            self.root.title("Attribute blanks to samples")
        elif self.type == 'std':
            self.root.title("Attribute standards to samples")
        
        # Set combobox
        self.combobox_values = [str(i) for i in range(1,6)]  # Initial combobox values
        
        # Static frame for the left listbox
        self.static_frame = tk.Frame(self.root)
        self.static_frame.pack(side="left", fill="both", expand=False)  # Adjust as necessary
        
        # Create the canvas, which will automatically be to the right of the scrollbar due to packing order
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side="right", fill="both", expand=True)
        
        
        # Create the scrollbar and pack it first, so it's on the left
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.xview)
        self.scrollbar.pack(side="right", fill="y")


        # The scrollable frame and its content go here
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        # Make sure to bind the canvas configuration to a function that updates the scroll region
        self.scrollable_frame.bind("<Configure>", self.update_scrollregion)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        
        # Creating a label on top of the window.
        self.title_label = tk.Label(self.static_frame, text=f'You must sort all samples from the list to a {Type} group:')
        self.title_label.grid(row=3, column=0, pady=(20, 0))  # Displaying the label at the top.

        # Before creating the listbox:
        if self.type == 'label':
            items = list(self.df.loc[self.df['blank'] == False, 'sample'].unique())
        elif self.type == 'blank':
            items = self.df['sample'].tolist()
        elif self.type == 'std':
            items = list(self.df.loc[self.df['blank'] == False, 'sample'].unique())
        
        self.max_width = max(len(item) for item in items) # take the longest name to set the width of listboxes
        
        # Creating a listbox on the left side of the window.
        self.left_listbox = tk.Listbox(self.static_frame, selectmode=tk.EXTENDED, height=20, width=(self.max_width + 1))
        if self.type == 'label':
            for i in items:  # Adding each sample to the listbox.
                self.left_listbox.insert('end', i)
        elif self.type == 'blank':
            for i in self.df['sample']:  # Adding each sample to the listbox.
                self.left_listbox.insert('end', i)
        self.left_listbox.grid(row=4, column=0, pady=(0, 0))  # Displaying the listbox on the left side.

        # Creating the 'Add Label or Add blank' button.
        if self.type == 'label':
            self.add_label_button = tk.Button(self.static_frame, text="Add Label group", command=lambda: self.add_label("label"))
            self.add_label_button.grid(row=0, column=0, pady=5)  # Displaying the button at the top.
            self.add_pool_button = tk.Button(self.static_frame, text="Add Pool group", command=lambda: self.add_label("pool"))
            self.add_pool_button.grid(row=1, column=0, pady=5)  # Displaying the button at the top.
            self.add_standard_button = tk.Button(self.static_frame, text="Add Standard group", command=lambda: self.add_label("std"))
            self.add_standard_button.grid(row=2, column=0, pady=5)  # Displaying the button at the top.
        elif self.type == 'blank':
            self.add_label_button = tk.Button(self.static_frame, text="Add blank group", command=lambda: self.add_label("blank"))
            self.add_label_button.grid(row=0, column=0, pady=0, columnspan=3)  # Displaying the button at the top.
        elif self.type == 'std':
            self.add_label_button = tk.Button(self.static_frame, text="Add Standard group", command=lambda: self.add_label("std"))
            self.add_label_button.grid(row=0, column=0, pady=5)  # Displaying the button at the top.

        # Creating a frame for the filename entry and 'Generate Table with Label' button.
        self.filename_frame = tk.Frame(self.static_frame)
        self.filename_frame.grid(row=6, column=0, pady=20, sticky='ew')  # Displaying the frame at the top with some padding.

        # Create a Scale widget
        self.slider = tk.Scale(
            self.filename_frame,  # parent widget
            from_=0,  # minimum value of the scale
            to=100,  # maximum value of the scale
            orient=tk.HORIZONTAL,  # horizontal orientation
            label="Select a blank threshold (% of average sample):",  # label text
            length=260,  # length of the slider (in pixels)
            command=self.update_scale
        )
        # Set the default value of the slider
        self.slider.set(self.scale)
        if self.type == 'blank':
            self.slider.grid(row=5, column=0, pady=20, columnspan=2)

        # Creating the 'Generate Table with Label' button.
        if self.type == 'label':
            self.generate_table_button = tk.Button(self.filename_frame, text='Generate table with label', command=self.generate_table_with_label)
            self.generate_table_button.grid(row=5, pady=3)  # Displaying the button on the left side of the frame.
        elif Type == 'blank':
            self.generate_table_button = tk.Button(self.filename_frame, text='Remove the blank signal', command=self.generate_table_with_blank)
            self.generate_table_button.grid(row=6, column=0, pady=3, columnspan=2)  # Displaying the button on the left side of the frame.  
        elif Type == 'std':
            self.generate_table_button = tk.Button(self.filename_frame, text='Associate Standards to samples', command=self.generate_table_with_blank)
            self.generate_table_button.grid(row=6, column=0, pady=3, columnspan=2) 
    
        self.last_selected_value = {}  # Dictionary to track last selected number for each combobox
    
    def on_combobox_select(self, event, combobox): # combobox is one of the comboboxes object among multiple
        # Save the selected value
        self.last_selected_value[combobox] = combobox.get()

    def on_combobox_custom_input(self, event, combobox):
        custom_value = combobox.get()
        original_value = self.last_selected_value.get(combobox)
        self.combobox_values = [custom_value if x == original_value else x for x in self.combobox_values]
        self.update_comboboxes()
        print(self.last_selected_value)
        print(self.combobox_values)
        print('')

    def update_comboboxes(self):
        for label_box in self.label_boxes:
            combobox = label_box["combobox"]
            # Update combobox options with the current values from shared data source
            combobox['values'] = list(self.combobox_values)
    

        
    # Method to retrieve the current value of the slider
    def update_scale(self, value):    
        self.scale = int(value)
    
    # Update the scroll region to fit the content of the scrollable_frame.
    def update_scrollregion(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    # Function to add a new label box.
    def add_label(self, Subtype):  
        frame = tk.Frame(self.scrollable_frame)  # Creating a new frame for the label box.
        frame.pack(side="left")  # Displaying the frame on the left side.          
        subtype = Subtype
        if Subtype == 'std':
            listbox_std = tk.Listbox(frame, selectmode=tk.EXTENDED, height=4, width=(self.max_width + 1))
            listbox_std.pack(pady=30)
            # Creating a new Toplevel window (popup window)
            top = tk.Toplevel(self.root)
            top.title("Add standard Label")
            
            # Adjusting the width of the Toplevel based on the title
            title_width = len(top.title())
            top.geometry(f"{title_width*10}x200")  # Assuming each character in the title is approximately 10 pixels wide
    
    
            # Creating a Listbox for unique values from self.all_std['name']
            self.std_listbox = tk.Listbox(top, selectmode=tk.SINGLE)
            for item in self.all_std['name'].unique():
                self.std_listbox.insert(tk.END, item)
            self.std_listbox.pack(pady=10)
    
            # Creating a scrollable frame for checkboxes
            canvas = tk.Canvas(top)
            scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
            self.check_frame = tk.Frame(canvas)
            self.check_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=self.check_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Variable to store checkbox values
            self.check_vars = []
            self.checkboxes = []
            
            def update_checkboxes(event):
                # Clearing old checkboxes
                for checkbox in self.checkboxes:
                    checkbox.destroy()
                self.checkboxes = []
                self.check_vars = []
    
                # Getting selected item from listbox
                selected = self.std_listbox.get(self.std_listbox.curselection())
                
                # Creating new checkboxes
                selected_std = list(self.all_std.loc[self.all_std['name'] == selected, 'Name2'])
                compounds = [] # we propose to select the compounds, and not the adducts to simplify
                for st in selected_std:
                    c = st.split(' [')[0] # separate the name from the adducts property based on "name [H+H2O]"
                    if c not in compounds:
                        compounds.append(c)
                for item in compounds:
                    var = tk.BooleanVar()
                    self.check_vars.append(var)
                    cb = tk.Checkbutton(self.check_frame, text=item, variable=var)
                    cb.pack(anchor=tk.W, padx=10, pady=5)
                    self.checkboxes.append(cb)
    
            self.std_listbox.bind("<<ListboxSelect>>", update_checkboxes)
    
            # Confirm button
            def on_confirm():
                selected_items = [checkbox.cget("text") for checkbox, var in zip(self.checkboxes, self.check_vars) if var.get()]
                # Populating the main Listbox with the selected items
                for it in selected_items:
                    listbox_std.insert('end', it)
                top.destroy()
    
            confirm_button = tk.Button(top, text="Confirm", command=on_confirm)
            confirm_button.pack(pady=10)

         
        pool_name = f"Pool_{self.pool_counter[0]}"
        std_name = f"Std_{self.std_counter[0]}"
        if Subtype == 'pool':
            self.pool_counter.pop(0)
        elif Subtype == 'std':
            self.std_counter.pop(0)           

        

        # Creating an entry for the label box title.
        entry = tk.Entry(frame)
        if self.type == 'label' and Subtype == 'label':
            entry.insert(0, "Enter label here")  # Default title.
            entry.pack(pady=30)  # Displaying the entry.
            # Create a Combobox
            value_combobox = ttk.Combobox(frame, values=list(self.combobox_values))
            value_combobox.set("Treatment level")  # Set the default value
            value_combobox.pack(pady=10)  # Displaying the combobox
            # Bind the combobox select event to the callback function
            value_combobox.bind("<<ComboboxSelected>>", lambda event, cb=value_combobox: self.on_combobox_select(event, cb))
            value_combobox.bind("<Return>", lambda event, cb=value_combobox: self.on_combobox_custom_input(event, cb))
        elif self.type == 'label' and Subtype == 'pool':
            entry.insert(0, pool_name)  # Inserting the default text.
            entry.config(state='disabled')  # This makes the entry uneditable.
            entry.pack(pady=30)  # Displaying the entry.
        elif self.type == 'label' and Subtype == 'std':
            entry.insert(0, std_name)  # Inserting the default text.
            entry.config(state='disabled')  # This makes the entry uneditable.
            entry.pack(pady=30)  # Displaying the entry.
        else:
            listbox_blank = tk.Listbox(frame, selectmode=tk.EXTENDED, height=4, width=(self.max_width + 1))
            listbox_blank.pack(pady=30)  # Displaying the listbox.

        # Creating a listbox for the label box.
        listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, width=(self.max_width + 1))
        listbox.pack(pady=2)  # Displaying the listbox.
        
        # After adding the new listbox, explicitly update the scroll region
        self.update_scrollregion()

        # Function to move an item from the left listbox to this label box.
        def move_item_to_box():
            items = self.left_listbox.curselection()  # Getting the selected items from the left listbox.
            if items:  # If any item is selected.
                for item in reversed(items):  # Loop through each selected item.
                    selected_item = self.left_listbox.get(item)  # Get the text of the selected item.
                    all_items = listbox.get(0, tk.END)
                    if selected_item not in all_items:
                        if self.type == 'blank' or Subtype == 'pool' or Subtype == 'std':
                            self.left_listbox.delete(item)  # Remove the item from the left listbox.
                        listbox.insert('end', selected_item)  # Add the item to this label box.
                    
        def move_item_to_blank():
            items = self.left_listbox.curselection()  # Getting the selected items from the left listbox.
            if items:  # If any item is selected.
                for item in reversed(items):  # Loop through each selected item.
                    selected_item = self.left_listbox.get(item)  # Get the text of the selected item.
                    self.left_listbox.delete(item)  # Remove the item from the left listbox.
                    listbox_blank.insert('end', selected_item)  # Add the item to this label box.

        # Function to move an item from this label box back to the left listbox OR the blank list.
        def move_item_to_list():
            items = listbox.curselection()  # Getting the selected items from this label box.
            items_blank = []
            if self.type == 'blank':
                if listbox_blank:
                    items_blank = listbox_blank.curselection()
                if items_blank:
                    for item in reversed(items_blank):  # Loop through each selected item.
                        selected_item = listbox_blank.get(item)  # Get the text of the selected item.
                        listbox_blank.delete(item)  # Remove the item from this label box.
            else:
                for item in reversed(items):  # Loop through each selected item.
                        selected_item = listbox.get(item)  # Get the text of the selected item.
                        listbox.delete(item)  # Remove the item from this label box.
                        if subtype == 'pool' or subtype == 'std' :
                            self.left_listbox.insert('end', selected_item)  # Add the item back to the left listbox.
    
        # Function to delete a sample from a label list
        def del_item():
            items = listbox.curselection()  # Getting the selected items from this label box.
            if items:  # If any item is selected.
                for item in reversed(items):  # Loop through each selected item.
                    listbox.delete(item)  # Remove the item from this label box.
            
        # Creating the 'Move Item to This Label' button.        
        move_item_to_box_button = tk.Button(frame, text='Move item to this group', command=move_item_to_box)
        move_item_to_box_button.pack(pady=5)  # Displaying the button.
        
        if self.type == 'blank':
            move_item_to_blank_button = tk.Button(frame, text='Move item to blank list', command=move_item_to_blank)
            move_item_to_blank_button.pack(pady=5)  # Displaying the button.
            # Creating the 'Move Back to List' button.
            move_item_to_list_button = tk.Button(frame, text="Move back to list", command=move_item_to_list)
            move_item_to_list_button.pack(pady=5)  # Displaying the button.

        if self.type == 'label':
            if Subtype == 'label':
                del_item_button = tk.Button(frame, text="Remove from this list", command=del_item)
                del_item_button.pack(pady=5)  # Displaying the button.
            else:
                del_item_button = tk.Button(frame, text="Move back to list", command=move_item_to_list)
                del_item_button.pack(pady=5)  # Displaying the button.
        

        # Adding this label box to the list of label boxes.
        if self.type == 'label':
            if Subtype == 'label':
                self.label_boxes.append({"entry": entry, "listbox": listbox, "combobox": value_combobox})
            else:
                self.label_boxes.append({"entry": entry, "listbox": listbox, "combobox": 0})
            # Delete button functionality
            def delete_label_box():
                # Check if the listbox has items                   
                if listbox.size() > 0 and (Subtype == 'pool' or Subtype == 'std'):
                    tk.messagebox.showwarning("Warning", "This list is not empty, empty the list before deleting it.")
                    return
                if Subtype == 'pool':
                    iteration_value = int(entry.get().split('_')[1]) # take the value of iteration from Pool_1, Pool_2, Std_1 etc
                    self.pool_counter.insert(0, iteration_value) # now a user have deleted a pool list, next time it creates a pool it takes the right iteration value
                    self.pool_counter.sort() # in case of multiple group are deleted, sorting make sure the new iteration value will be in ascending order
                elif Subtype == 'std':
                    iteration_value = int(entry.get().split('_')[1]) # take the value of iteration from Pool_1, Pool_2, Std_1 etc
                    self.std_counter.insert(0, iteration_value)
                    self.std_counter.sort()
                frame.pack_forget()  # Remove the frame and its children (i.e., the listbox and buttons)
                
                # Correctly removing the label box from the list of label boxes
                box_to_remove = None
                for box in self.label_boxes:
                    if box["entry"] == entry and box["listbox"] == listbox:
                        box_to_remove = box
                        break
                if box_to_remove:
                    self.label_boxes.remove(box_to_remove)
        elif self.type == 'blank':
            self.label_boxes.append({"Blank groups": listbox_blank, "listbox": listbox})
            # Delete button functionality
            def delete_label_box():
                # Check if the listbox has items
                if listbox.size() > 0:
                    tk.messagebox.showwarning("Warning", "This list is not empty, empty the list before delete it.")
                    return
                frame.pack_forget()  # Remove the frame and its children (i.e., the listbox and buttons)
                
                # Correctly removing the label box from the list of label boxes
                box_to_remove = None
                for box in self.label_boxes:
                    if box["Blank groups"] == listbox_blank and box["listbox"] == listbox:
                        box_to_remove = box
                        break
                if box_to_remove:
                    self.label_boxes.remove(box_to_remove)
            
        # Adding a delete button
        delete_button = tk.Button(frame, text="Delete", command=delete_label_box)
        delete_button.pack(pady=5)  # Displaying the delete button.

    # Function to generate a table with labels.
    def generate_table_with_label(self):
        for box in self.label_boxes:  # Loop through each label box.
            label = box["entry"].get()  # Get the label box title.
            level = box["combobox"].get()
            if label == "Enter label here" or label == "":  # If the title is not changed or empty.
                # Showing an error message.
                tk.messagebox.showerror("Error", "One label box lacks a label")
                return
            if level == 'Treatment level' or level in '12345': # everify if the user have attributed a name to the treatment levels
                tk.messagebox.showerror("Error", "At least one label box lacks a treatment level with a name")
                return
        response = tkinter.messagebox.askyesno("Confirm Treatments", "Have you well defined all levels of treatments?")
        if response: # if it is yes, all treatments are defined
            for box in self.label_boxes:  # Loop through each label box.
                label = box["entry"].get()  # Get the label box title.
                level = box["combobox"].get()
                self.treatment[label, level] = []
                for i in range(box["listbox"].size()):  # Loop through each item in the label box
                    sample = box["listbox"].get(i)  # Get the item text.
                    self.treatment[label, level].append(sample)
                    # Assigning the label to the sample in the dataframe.
                    if (self.df.loc[self.df['sample'] == sample, 'label'] == '').any():
                        self.df.loc[self.df['sample'] == sample, 'label'] = label
                    else:
                        self.df.loc[self.df['sample'] == sample, 'label'] += f';{label}'
            tk.messagebox.showinfo("Success", f"{self.name} successfully labelled!")
            self.root.destroy()
        
        
    # Function to generate a table with labels.
    def generate_table_with_blank(self):  
        if self.left_listbox.size() > 0:  # If there are still items in the left listbox.
            # Showing an error message.
            tk.messagebox.showerror("Error", "You have to sort all samples in a label before exporting table!")
            return

        for box in self.label_boxes:  # Loop through each label box.
            blank = box["Blank groups"].size()  # Get the label box title.
            if blank == 0 :  # If the title is not changed or empty.  
                blanks = ['No_blank']             
                tk.messagebox.showerror("Warning", "At least one group box lacks blank(s) samples, by default we consider no features detected in blank for the concerned group(s).")
            else:
                blanks = []
                for i in range(box["Blank groups"].size()):
                    b = box["Blank groups"].get(i)
                    blanks.append(b)
                
            for i in range(box["listbox"].size()):  # Loop through each item in the label box.
                sample = box["listbox"].get(i)  # Get the item text.
                # Assigning the label to the sample in the dataframe.
                self.df.loc[self.df['sample'] == sample, 'blank'] = ','.join(blanks)

        self.exported_csv_path = os.path.join(self.export_path,  f"{self.name}_deblanked.csv") # TODO use file manager to generate new path
        self.df.to_csv(self.exported_csv_path, index=False)
        
        tk.messagebox.showinfo("Success", f"{self.name} successfully deblanked!")
        self.root.destroy()
    
    # Function to prevent unwanted closings of the window
    def on_closing(self):
        response = tkinter.messagebox.askyesno("Confirm closing", "Are you sure you want to quit? It will interrupt the whole untargeted metabolomic pipeline")
        if response == True:
            self.destroy = True # indicate a premature interuption
            self.root.destroy()
        else:
            pass
            
    # Function to start the tkinter main event loop.
    def run(self):  
        self.root.mainloop()  # Starting the tkinter main event loop.
        if self.destroy == True:
            return self.df, True
        if self.type == 'label':
            return self.df, self.treatment
        elif self.type == 'blank':
            return self.df, self.scale
