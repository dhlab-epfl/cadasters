# Napoleonic Venetian cadaster processing
This work was presented at the Digital Humanities 2019 conference in Utrecht, [_A deep learning approach to Cadastral Computing_](https://infoscience.epfl.ch/record/268282/files/dh2019-cadaster-computing.pdf).

To see the old version of the work presented in 2017 ([_Machine vision algorithms on cadaster plans_](https://infoscience.epfl.ch/record/254960/files/final_abstract.pdf)), go to [this branch](TODO)

### Segmentation (dhSegment)
The segmentation uses a fully convolutional neural network, dhSegment, which is available at [github.com/dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment)

### Transcription (CRNN)
The code for training the convolutional recurrent neural network used to transcribe the labels can be found at [github.com/solivr/tf-crnn](https://github.com/solivr/tf-crnn)

### Usage
In order to use this code, two models need to be trained: a segmentation model (dhSegment) and a transcription model (CRNN). 
Once the models are trained, it is possible to process the casdaster maps by running:

`python process_cadaster.py -im image1.tif image2.jpg -out output_dir -sm segmentation_model_path -tm transcription_model_path`
 
`-im`: image(s) to process (can be of type .tif, .jpg or .png)  
`-out`: output directory where the geojson containing the geometries and their transcriptions will be saved  
`-sm`: path to the Tensorflow saved model for segmentation  
`-tm`: path to the Tensorflow saved model for transcrition 
 
### Requirements
Create a new environement using the `requirements.txt`.  
Then install [`dhSegment`](https://github.com/dhlab-epfl/dhSegment) and [`tf-crnn`](https://github.com/solivr/tf-crnn).  

