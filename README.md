# synapseParkinson
[Parkinsons Disease Digital Biomarker DREAM Challenge](https://www.synapse.org/#!Synapse:syn8717496)

## The write-up can be found [here](https://github.com/patbaa/synapseParkinson/blob/master/writeup.pdf). 
It is recommended to download as the github's PDF viewer doesn't render it properly.

## Instructions to run the code:
1. run the data downloader [notebook](https://github.com/patbaa/synapseParkinson/blob/master/dataDownloader/dataDownloader.ipynb)<br>
this notebook downloads the acceleration tracks and creates .tsv tables with their metadata and file locations.

2. run the feature creator notebooks from 1 to 4 for the selected subchallenge. (eg for [tremor](https://github.com/patbaa/synapseParkinson/tree/master/sub2.1_tremor))

3. run the feature selector notebook for the selected subchallange (eg for [LINK TO TREMOR](LINK))

In the first cell of each notebook you need to set the baseDIR variable according to your directory system.
