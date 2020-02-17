# CMAC - Neural Network
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/Supermarket-Cleaning-Robot/blob/master/LICENSE)

## Overview

This project implements the Cerebellar Model Articulation Controller (CMAC) described by James Albus in 1975.
CMAC is a type of neural network based on the mammalian cerebellum. 

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Numpy and Matplotlib

## Install Dependencies

- Install *Python3*, *Python3-tk*, and the necessary libraries: (if not already installed)
````
sudo apt install python3 python3-tk
pip3 install numpy matplotlib
````
- Check if your system successfully installed all the dependencies
- Open terminal using ````Ctrl+Alt+T```` and enter ````python3````
- The terminal should now present a new area represented by ````>>>```` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ````Ctrl+Z```` if an error pops up while 
running the below commands)
````
import tkinter
import numpy as np
import matplotlib.pyplot as plt
````

## Run

- Using the terminal, clone this repository, go into the project directory, and run the main program:
````
git clone https://github.com/urastogi885/cmac-neural-network
cd cmac-neural-network
python3 main.py
````
- The program launches 8 plots: next one will pop up once you close the current one
- Alternatively, if you work with the PyCharm editor, you can add this project to the PyCharm workspace and run the
*main.py* program through the editor. 
- The advantage of using PyCharm in this project is that you will be able to access all the plots.  

