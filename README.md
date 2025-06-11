# Artificial Image Generation
Synthetic image generation of fruits using Generative Adversarial Network(GAN) model. This code provides the model for generation of images of apples and calculate the FID score for the generated images.

A few samples of the input images are,<br/>
<img src="input_data/input_images/apples/r0_0.jpg" width="150" />
<img src="input_data/input_images/apples/r1_1.jpg" width="150" />
<img src="input_data/input_images/apples/r1_300.jpg" width="150" />

Images were generated with FID Score of 61. Samples of the generated images are: <br/>
<img src="output/generated_images/apples/generated_image_0.png" width="150" />
<img src="output/generated_images/apples/generated_image_1.png" width="150" />
<img src="output/generated_images/apples/generated_image_10.png" width="150" />

The FID score could be improved using hyperparameter tuning, which requires high computational resources. :disappointed:

For installing required libraries run, 
```
cd apple_images
pip install - r requirements.txt
```
For training and generation of images, run:
```
python3 main.py
```
