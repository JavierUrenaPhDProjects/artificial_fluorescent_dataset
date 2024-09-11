# Bacillus Subtilis artificial dataset creator

This project allows the creation of an artificial fluorescent bacteria dataset oriented for bacteria counting.

## Description

By using template backgrounds, which are microscopic images from a blank agar plate, the program will alter pixel values
defined by two Gaussian ellipses with different, random distributions, rotations, and XY positions in the image. 

![Alt text](figures/artificial_dataset.png?raw=true "Concept behind fake bacteria generation")
After
running the main script, generated images will be saved in the specified folders and a csv with the name, number of
instances and centroid locations wil be created.

![Alt text](figures/500_cells.png?raw=true "Example of an artificial fluorescent image with 500 cells")

NOTE: the generated images will have a resolution of 3280x2464 pixels.

## Getting Started

### Executing scripts

There are only two hyperparameters to consider, that are defined in the "main.py" file:

* N: Total number of images desired to generate for the dataset
* max_cells: Maximum number of cells that the script will add in the images

Running the main.py script will create 'N' images using the different backgrounds in the DATASET/Background folder. The
images will have from 0 to 'max_cells' number of bacteria.

## Authors

Contributors names and contact info

ex. Javier Ureña Santiago  
ex. javier.urena@uibk.ac.at
