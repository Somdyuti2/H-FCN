# VP9 Intra Encoding using H-FCN

## Prerequisities

Python 3 and NumPy  
Keras with Tensorflow backend   
Tensorflow C API   

The code provided in this project has been tested with Tensorflow version 1.12. The GPU version has been used for training and the CPU version for encoding. 

## Install Tensorflow C API
Install the Tensorflow [C API](https://www.tensorflow.org/install/lang_c). The CPU version has been used for integration with the VP9 encoder. For best performance, compile the library [from source](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md)  enabling all the CPU extensions (AVX, AVX2, SSE 4.2, FMA etc.) supported by your processor. Additionally, if you are using an Intel processor, build with support for Intel's [MKL-DNN](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide) by setting the build config as `--config=mkl` to get a further boost in performance.  

## Build libvpx library
libvpx v1.6.0 has been used in this project.  

`git clone https://github.com/Somdyuti2/H-FCN.git`  
`cd H-FCN`  
`mkdir libvpx-build`  
`cd libvpx-build`  
`../libvpx/configure --enable-debug --enable-debug-libs`  

In the `Makefile` created by the last step, find each occurrence of the line `(qexec)$$(LD) $$(strip $$(INTERNAL_LDFLAGS) $$(LDFLAGS) -o $$@ $(2) $(3) $$(extralibs))` and add `-ltensorflow` to its end (there should be two occurrences at lines 257 and 263 of the `Makefile` when build with the above configuration). Then run:

`make`  
`cd ..`

## Generate some data
To get started, collect some VP9 encoded videos and the corresponding sources, and generate some data for training and validation. A few example videos can be found [here](https://drive.google.com/drive/folders/1UCJ0qCQDSn90b-CFFwULor9xxkMkuI9m?usp=sharing). 

`mkdir Dummy_data/`  
`mkdir Dummy_data/Stat_files/`  
`mkdir Dummy_data/Training/`  
`mkdir Dummy_data/Validation/`  

Record superblock partition data using VP9 decoder (`vpxdec`):  
`./libvpx-build/vpxdec Example_Videos/Encodes/park_joy.ivf --codec=vp9  --blockstats=Dummy_data/Stat_files/park_joy_partitions.txt -o /dev/null`  
`./libvpx-build/vpxdec Example_Videos/Encodes/mob_cal.ivf --codec=vp9  --blockstats=Dummy_data/Stat_files/mob_cal_partitions.txt -o /dev/null`

Generate training and validation data using the partition data collected in previous step:  
`python3 Python_scripts/create_database.py --ffmpeg_path /usr/bin/ffmpeg --ffprobe_path /usr/bin/ffprobe --source_path Example_Videos/Sources/park_joy_420_720p50.y4m --encode_path Example_Videos/Encodes/park_joy.ivf --stats_path Dummy_data/Stat_files/park_joy_partitions.txt --output_path Dummy_data/Training/ --key park_joy`  

`python3 Python_scripts/create_database.py --ffmpeg_path /usr/bin/ffmpeg --ffprobe_path /usr/bin/ffprobe --source_path Example_Videos/Sources/720p50_mobcal_ter.y4m --encode_path Example_Videos/Encodes/mob_cal.ivf --stats_path Dummy_data/Stat_files/mob_cal_partitions.txt --output_path Dummy_data/Validation/ --key mob_cal`  

*note - the data generated above is just intended to set up a working example of training the model and is by no means sufficient to train the model. 

## Train H-FCN model

Train model:  
`mkdir Python_scripts/logs_fcn`  
`python3 Python_scripts/HFCN_train.py --train_path Dummy_data/Training/ --val_path Dummy_data/Validation/ --hist_path Python_scripts/logs_fcn/ --model_path Python_scripts/trained_model/`

Plot loss and accuracy:  
`tensorboard --logdir Python_scripts/logs_fcn/`

## Convert trained Keras model to Tensorflow  
`python3 Python_scripts/convert_model_26336_params.py --model Python_scripts/trained_model/hfcn.hdf5 --numout 4 --outdir Python_scripts/trained_model/ --name hfcn.pb`


## Use trained Tensorflow H-FCN model to encode videos in intra mode   

`mkdir H-FCN_encodes`  
`cd libvpx-build`  
`time ./vpxenc  --threads=1 --passes=1 --cpu-used=1 --good  --end-usage=q --cq-level=30  --auto-alt-ref=0 --kf-min-dist=1 --kf-max-dist=1 --tile-rows=0 --tile-columns=0 --codec=vp9 -o ../H-FCN_encodes/mobcal_HFCN_encoded.ivf  ../Example_Videos/Sources/720p50_mobcal_ter.y4m`  

By default, the encoder uses our pretrained Tensorflow model trained on `11 990 384` samples (available as `Python_scripts/trained_model/hfcn_trained.pb`). In order to use a different trained model, run the following commands from `libvpx-build`:  
`sed -i 's#../Python_scripts/trained_model/hfcn_trained.pb#../Python_scripts/trained_model/hfcn.pb#' ../libvpx/vp9/encoder/vp9_encodeframe.c`  
`make`

The new trained model is assumed to be in location `/Python_scripts/trained_model/hfcn.pb`. Change the path as required. 

## Citation  
If this code is used for research, please cite [Speeding up VP9 Intra Encoder with Hierarchical Deep Learning Based
  Partition Prediction](http://arxiv.org/abs/1906.06476). 

## License
Different parts of the source code are distributed under different open source licenses and the individual source files should be checked for details. 
