# Guide to set up environment
(Will be updated trhoughout the course)

## Basics
The course uses Python and but there might also be a bit of R for data wrangling (primarily dplyr) and the slides are written with the [xaringan](https://slides.yihui.org/xaringan/) library.

The course generally uses the spyder IDE in the anaconda environment. But feel free to use the tools you prefer. 

There are plenty of guides online for how to install this: https://youtu.be/-sNX_ZMVpQM?si=jQknaT0sN3WYS7VW

## Conda environment (feel free to use something else than Spyder)
`conda create --name sentimentF23`  
`conda activate sentimentF23`  
`conda install spyder`  
`conda install nltk`  
`conda install numpy matplotlib pandas seaborn requests`

## Install pytorch to run on cuda 11.8
`pip install transformers`  
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`  

### Verify the pytorch installation and that it is running on cuda 11.8
`python -c "import torch; print(torch.cuda.get_device_name(0))`

installing spacy, see: https://spacy.io/usage

The two main libraries: nltk and spaCy

## NLTK data
We will use nltk data extensively. Download everything with the following python code    
`import nltk`  
`nltk.download()`  


