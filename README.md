# A Sequence Learning Model for Decision Making in the Brain
# Source code for the network model and the analyses.

The Python code for the analyses in Zhang et al.(2020)

### Installation
Download the source code into a local directory.

### Add path
Please add the folder path and its subfolder in the PATH system variable

### Environment
This code has been tested on python 3.6.8 in linux. All the required packages are listed in the file 'reqirements.txt'

### Running the code
specify the config file and run `main.py` in the terminal
$ python main.py --config 'config_shape'

### Datasaving
The trained model is saved in folder ./save
The log of the behavioral results and neuronal response is saved in ./log

### Data analysis

	** probability reasoning task **
		- 'analysis_shape.py': plotting the psychometric curve (fig3a/4a) and the reaction time distribution  (fig3c/4c)
							    the subjective weight of each shape (fig3b/4b) and temporal effect (fig3d/4d)
							    the psth of neurons with different selectivity (fig5a/b/c)
							    the variance of units' response (fig5d)
							    the prediction of output units on the appearance of shapes (fig7)
		
		- 'analysis_whenwhich.py': plotting the effect of when/which lesion and graph analysis (fig6)

	** multisensory integration task **
		-'analysis_multisensory.py': contains all the analysis about the multisensory integration task


	** Post-decision wagering task **
		-'analysis_sure.py': contains all the analysis about the Post-decision wagering task

	** Two step task **
		-'analysis_twostep.py': contains all the analysis about the two step task

	* analysis results/figs are saved in folder ../figs





