# NanoTranslator by HyperbeeAI
Copyrights © 2023 Hyperbee.AI Inc. All rights reserved. hello@hyperbee.ai

This repository contains the Spanish-to-English translation utility by HyperbeeAI called NanoTranslator. **The model takes up less than 400 KBs of RAM and provides accurate translation for casual conversations.**

To run the demo, which communicates with a MAX78000 FeatherBoard over a USB serial port, see explanations in "demo.ipynb". This notebook acts as the serial terminal to communicate with the ai85 from the host PC. Further explanations are provided below as well as in the notebooks.

If you only want to evaluate model accuracy, run "evaluation.ipynb". Note that this notebook does an exact simulation of the MAX78000 arithmetic, so the accuracy should be exactly the same as what would be achieved on a hardware run. 

![Demo](./assets/ai8x-nlp-demo.gif)

## Installation:

    bash setup.sh 

Note1: If torchtext defaults to older versions in your container (e.g., v0.8), the .legacy submodule path needs to be removed from the import directives in the .py files and Jupyter notebooks.

Note2: There are multiple python packages on pip that provide serial port implementation, with conflicting function/object names too. Although the package used here gets imported with "import serial", it needs to be installed via "pip install pyserial", not "pip install serial". Make sure you get the correct version. See requirements.txt for more info.

## Contents:

- **.py files:** python modules used by the Jupyter notebooks. These files define a simulation environment for the MAX78000 CNN accelerator hardware + some peripheral tools that help evaluation. Note that the simulator only includes the chip features that are relevant to this project (e.g., pooling not implemented because this project does not need it).  

- **evaluation.ipynb:** this Jupyter notebook provides an interface to try out different sentences from the test set on the model in the simulation environment, and compute the BLEU score of the model over the test set.

- **demo.ipynb:** this Jupyter notebook acts as the serial interface with the chip. A sentence in the source language is sent over to the chip for translation via the serial port, the implementation on the chip translates this and sends it back via the same serial port in the target language, and the result is displayed on the notebook cell. This needs to be run together with the "assets/demo.elf" program on the chip, which does the actual translation job on the ai85. There is a specific cell on the notebook that needs to be run before the ai85 demo.elf is started. Check the notebook for further info.

- **assets/demo.elf:** C program running the actual translation application. Run this together with the demo.ipynb notebook for the translation demo. See further explanations inside demo.ipynb.


## Extras/Notes:

- the demo C program does not require any extra modules/libraries, it can be directly run the same way as the Maxim SDK examples (i.e., using the arm gdb, defining the target as "remote localhost:3333", doing "load" etc.). However, note that the Jupyter notebook demo.ipynb needs to be run together with the C program for meaningful output. There is a specific cell on the notebook that needs to be run before the ai85 demo.elf is started. Check the notebook for further info.

- The demo.ipynb notebook needs to run on the same host PC that programs the ai85 since it uses the on-board (USB) serial port (that programs the ai85) to communicate with the chip while the translation application is running.

- Although the program should run on both the EVKit and the FeatherBoard without errors (since it uses common functionality), it was only explicitly tested with the FeatherBoard for now.
