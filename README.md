# synapseParkinson
[Parkinsons Disease Digital Biomarker DREAM Challenge](https://www.synapse.org/#!Synapse:syn8717496)

## The write-up can be found in PDF [here](https://github.com/patbaa/synapseParkinson/blob/master/writeup.pdf). 
It is recommended to download as the github's PDF viewer doesn't render it properly.

## Instructions to run the code:
Using clean conda environment is suggested.
```
conda create --name MYENV python=3.6.2
python -m ipykernel install --user --name MYENV --display-name "MYENV"
```

Use _python 3.6.2_<br>
Install the dependencies
```
pip install -r requirements.txt
```

Run the wrapper [notebook](https://github.com/patbaa/synapseParkinson/blob/master/wrapper.ipynb)<br>
this notebook downloads the acceleration tracks and creates all the features and submission files.<br>
<b> In the wrapper notebook you need to set the baseDIR </b>
