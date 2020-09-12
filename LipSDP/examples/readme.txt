#First please download the code from https://github.com/arobey1/LipSDP and install the requirements
#The usage of LipSDP is also on this github website.
#Second,copy all files in this directory to /LipSDP/LipSDP/examples
#Hint: the subnet 0 of CNN1 in saved_weights is too large to upload to github, please unzip it manually

saved_info			#Directory stores all local infomation
saved_model			#Directory stores all models trained by pytorch
saved_weights			#Directory stores all weights infomation that can be used by LipSDP

box_analyser_mnist.py		#extract some data points and their coresponding Alpha and Beta.These infomation is stored in saved_info
				#When we have alpha and beta, we can calculate the local lipschitz constant using LipSDP by add parameter "--alpha " and "--beta"
							
batch_run.py			#This program verifies a batch of inputs according to the lipschitz constant we have.

experiment.txt			#All experiment results is stored in experiment.txt