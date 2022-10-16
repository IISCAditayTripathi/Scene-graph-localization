The code is provided for training and testing VL-MPAG Net on Visual Genome and COCO-Stuff datasets.

# Installation

The installation requirements are given in `requirements.txt` file.

# Data prepration

## Visual Genome dataset

Download the Visual Genome dataset images [link](https://visualgenome.org/).

The splits of the Visual Genome dataset used in our analysis can be accessed by use the following **anonymous link**

`https://github.com/anonymous9039/SGL_data`

## COCO-stuff dataset
 Downoad the COCO-stuff dataset from [link](https://github.com/nightrome/cocostuff)

# Training

The training instructions are provided for *Visual Genome* and *COCO-stuff* datasets.

## Training on Visual Genome

The training code is provided for both Visual Genome Full Overlap(VGFO) and Visual Genome Partial Overlap (VGPO) setting.

For training the model on VG dataset run the following command.

`python train_detector_vg.py`

## Training on COCO-stuff dataset

Run the following commands for training.

`python train_detector_coco.py`

You can find the hyperparameters of the model inside the training files.

# Testing

## Visual Genome
For testing the model on VG run the following command.

`python test_detector_vg.py`

## COCO-stuff
`python test_detector_coco.py`


