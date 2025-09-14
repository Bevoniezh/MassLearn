# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:22:02 2023

@author: Ronan
"""

from tqdm import tqdm
import time

class DisplayText():
    def __init__(self, Text, Menulist = {}, Definition = [], Complementary = ''):
        self.text = Text
        self.menu = Menulist
        self.definition = Definition
        self.complementary = Complementary
        
    # Function to display definitions
    def definitions(self):
        print(self.square_info())
        print(f'{self.definition["Definition"]}\
              \n\nSource: {self.definition["Source_ID"]}\
              \nLink: {self.definition["Source_link"]}')
        
             
    #Function to display a multiple choice menu
    def menu_choice(self, Module = 'module', Complementary = ''): # Complementary is an option to add explanation text bbetween the title and choose a module...
        adjustment = "#"*len(self.text)
        text = self.text.upper() # title of the menu
        
        # Take the maximum length of the terms to optimize the menu list display
        max_length = 0  # variable to store the length of the longest odd term
        for nb, term in self.menu.items():
            if int(nb) % 2 != 0:  # check if the term index is odd
                if len(term) > max_length:  # check if the length of the term is greater than the current max
                    max_length = len(term)        
        
        menu = ''
        count = 0
        for nb, module in self.menu.items():
            if len(self.menu) == 1: # if the length of terms is pair, there will be always two terms per row
                menu += f'[{nb}] - {module}\n'
                break
            elif count % 2 == 0:
                if int(nb) == len(self.menu): # if the length of terms is pair, there will be always two terms per row
                    menu += f'[{nb}] - {module}\n'
                    break
                space = (max_length - len(module))*' ' # equilibrate the space between terms to create two straight columns
                odd_nb = str(int(nb)+1)
                menu += f'[{nb}] - {module}{space}   [{odd_nb}] - {self.menu[odd_nb]}\n'
            count += 1  
            
        complementary = f'\n{Complementary}\n'
            
        return f'\n\n\n\n\t\t\t##{adjustment}##\
               \n\t\t\t# {text} #\
               \n\t\t\t##{adjustment}##\n\
               {complementary}\
               \nChoose a {Module} by indicating a number\
               \n\n{menu}\
               \n---> '
    
    #Function to write simple text
    def simple(self):
        return f'\n\n\t --------------------------------\n\n{self.text}'
    
    # Function to display text in a square
    def square_info(self):
        adjustment = "#"*len(self.text)
        return f'\n\n\t\t\t##{adjustment}##\
               \n\t\t\t# {self.text} #\
               \n\t\t\t##{adjustment}##\n\n'
               
    # Function to display text as warning
    def warning(self):
        print(f'\n/!\\/!\\\
               \nWARNING: {self.text}')
    
    # Function to display starting title
    def title(self):
        print('\n\n███╗░░░███╗░█████╗░░██████╗░██████╗██╗░░░░░███████╗░█████╗░██████╗░███╗░░██╗\
       \n████╗░████║██╔══██╗██╔════╝██╔════╝██║░░░░░██╔════╝██╔══██╗██╔══██╗████╗░██║\
       \n██╔████╔██║███████║╚█████╗░╚█████╗░██║░░░░░█████╗░░███████║██████╔╝██╔██╗██║\
       \n██║╚██╔╝██║██╔══██║░╚═══██╗░╚═══██╗██║░░░░░██╔══╝░░██╔══██║██╔══██╗██║╚████║\
       \n██║░╚═╝░██║██║░░██║██████╔╝██████╔╝███████╗███████╗██║░░██║██║░░██║██║░╚███║\
       \n╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═════╝░╚═════╝░╚══════╝╚══════╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝\n\n\
       🇮​​​​​🇲​​​​​🇵​​​​​🇷​​​​​🇴​​​​​🇻​​​​​🇪​​​​​ 🇾​​​​​🇴​​​​​🇺​​​​​🇷​​​​​ 🇷​​​​​🇪​​​​​🇸​​​​​🇪​​​​​🇦​​​​​🇷​​​​​🇨​​​​​🇭​​​​​ 🇸​​​​​🇰​​​​​🇮​​​​​🇱​​​​​🇱​​​​​🇸​​​​​')
       
class LoadingBar():
    def __init__(self, Iteration):
        self.total_iterations = Iteration
        self.current_iteration = 0
        self.progress_bar = tqdm(total=self.total_iterations, unit=' iterations', bar_format='\t{l_bar}{bar}| Estimated time to the end: {remaining}')

    def loading(self):
        self.progress_bar.update(1)
        self.current_iteration += 1
        if self.current_iteration == self.total_iterations: 
            self.progress_bar.close()
            self.progress_bar = ''





