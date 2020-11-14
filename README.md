# Style_transfer_2

The code of this project is a simple implementation of the paper————【ECCV-2016】Perceptual Losses for Real-Time Style Transfer
Paper Address:https://arxiv.org/pdf/1603.08155.pdf

##Environment
---------------------

Because the VGG19 parameter file is relatively large, the resource has been put on CSDN for free, please put it under the vgg19 folder after downloading, please see the detailed address：https://download.csdn.net/download/qq_40298054/13082438

Parametric model trained with Van Gogh’s starry sky as a style picture:
##File description
-------

For a detailed explanation of the code, please see：https://blog.csdn.net/qq_40298054/article/details/109599354
## Running the code
### Training a network for a particular style

python train_network.py --style "style/1.jpg" --train-path "<path to training images>"  --save-path <directory to save network>
 
### Using a trained network to transfer image
 
python stylize_image.py --content "content/lion.jpg" --output-path "1.jpg" --network "network"
 
### Using a trained network to transfer video
  
If you want to convert a video file: 
  
python stylize_video.py --video-path "<path to the video>" --output-path "output.avi" --network "network"
 
If you want to open the camera for real-time conversion:

python stylize_video.py --video-path "camera" --output-path "output.avi" --network "network"
