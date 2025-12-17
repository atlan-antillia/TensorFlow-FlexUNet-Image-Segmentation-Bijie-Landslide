<h2>TensorFlow-FlexUNet-Image-Segmentation-Bijie-Landslide (2025/12/18)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Bijie-Landslide</b> (Singleclass) based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and an Upscaled PNG 
<a href="https://drive.google.com/file/d/1S_y_S2guvvgkH4S1y_XsBvFD_YBX4EbN/view?usp=sharing">
<b>Augmented-Bijie-Landslide-ImageMask-Datase.zip</b></a>
which was derived by us from <br><br>
<a href="http://gpcv.whu.edu.cn/data/Bijie_pages.html">
<b>Bijie Landslide Dataset</b>
</a>.
<br> Bijie is a city in northwestern Guizhou, China<br>
<br>
<hr>
<b>Actual Image Segmentation for the Upscaled Bijie-Landslide Images</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our augmented dataset appear similar to the ground truth masks.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10009.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10255.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10255.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10255.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10388.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10388.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10388.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from <br><br>
<a href="https://zenodo.org/records/3775870">
<a href="http://gpcv.whu.edu.cn/data/Bijie_pages.html">
<b>Bijie Landslide Dataset</b>
</a>.
<br><br>
We create an open remote sensing landslide dataset called Bijie landslide dataset for developing automatic landslide detection methods. The dataset consists of satellite optical images, shapefiles of landslides’ boundaries and digital elevation models. All the images in this dataset, i.e. 770 landslide images (red points) and 2003 non-landslide images were cropped 
from the TripleSat satellite images captured from May to August 2018.
<br><br>
For the landslide instances, we provide the landslide images (*.png), the landslide shapefiles (mask files, *.png), 
the corresponding DEM data (*.png) and each landslide’ boundary coordinates (polygon, *.txt). <br>
For the non-landslide samples, the images and the corresponding DEM data were provided. <br>
All the data was prepared with a careful three-fold inspection to ensure its reliability.<br><br>
 More details can be found in:<br>
 Ji, S., Yu, D., Shen, C., Li, W., & Xu, Q.<br>
<b> Landslide detection from an open satellite imagery and digital elevation model dataset <br>
  using attention boosted convolutional neural networks. Landslides, 1-16, 2020.</b>
<br><br>
<b>License</b><br>
Unknown
</a><br>
<br>
<h3>
2 Bijie-Landslide ImageMask Dataset
</h3>
<h4>2.1 Download Bijie-Landslide dataset</h4>
 If you would like to train this Bijie-Landslide Segmentation model by yourself,
 please download the augmented <a href="https://drive.google.com/file/d/1S_y_S2guvvgkH4S1y_XsBvFD_YBX4EbN/view?usp=sharing">
 <b>Augmented-Bijie-Landslide-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Bijie-Landslide
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Bijie-Landslide Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Bijie-Landslide/Bijie-Landslide_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br><br> 
<h4>2.2 Bijie-Landslide Derivation</h4>
The renamed folder structure of the original dataset is the following. <br>
<pre>
./Bijie Landslide dataset
├─landslide
│  ├─dem
│  ├─image
│  ├─mask
│  └─polygon_coordinate
└─non-landslide
    ├─dem
    └─image
</pre>
We used the following 2 Python scripts to generate the Upscaled, width and height >=512 pixcels, and Augmented Bijie Landslide dataset from 
 770 files in <b>image</b> and corresponding mask files in <b>mask</b> in <b>landslide</b> folder.<br>
<ul>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>
<br>
<h4>2.3 Train Image and Mask samples </h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Bijie-Landslide TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Bijie-Landslide/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Bijie-Landslide and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and a large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
num_classes    = 2

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Bijie-Landslide 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                     Landslide: yellow
rgb_map = {(0,0,0):0, (255,255,0):1,}
</pre>


<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = False
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 31,32,33)</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 64,65,66)</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 66 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/train_console_output_at_epoch66.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Bijie-Landslide/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Bijie-Landslide/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Bijie-Landslide</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Bijie-Landslide.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/evaluate_console_output_at_epoch66.png" width="880" height="auto">
<br><br>Bijie-Landslide
<a href="./projects/TensorFlowFlexUNet/Bijie-Landslide/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Bijie-Landslide/test was low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0129
dice_coef_multiclass,0.9928
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Bijie-Landslide</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Bijie-Landslide.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the Upscaled Bijie-Landslide Images</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
augmented dataset appear similar to the ground truth masks.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10040.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10138.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10138.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10138.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10242.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10409.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10409.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10409.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10498.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10498.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10498.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/images/10559.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test/masks/10559.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Bijie-Landslide/mini_test_output/10559.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Landslide detection from an open satellite imagery
and digital elevation model dataset using attention <br>
boosted convolutional neural networks</b><br>
Shunping Ji, Dawen Yu, Chaoyong Shen, Weile Li, Qiang Xu<br>
<a href="https://sci-hub.ru/10.1007/s10346-020-01353-2">https://sci-hub.ru/10.1007/s10346-020-01353-2</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide
</a>

