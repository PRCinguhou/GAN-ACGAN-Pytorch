# GAN-ACGAN-Pytorch
Implementation of GAN &amp; ACGAN with face dataset

# DataSet 
<p float="left">
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00000.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00001.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00002.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00003.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00004.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00005.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00006.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00007.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00008.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00009.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00010.png" width=64 height=64 >
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/00011.png" width=64 height=64 >
</p>  

# Download Dataset 
```
bash ./get_dataset.sh 
```

# Training 
```
example :  
python3 gan.py -epoch 200 -batch 200 -lr 1e-4 
```
fixed noise image will save to ./gan_result

# Result
<p float="left">
<h3>gan</h3>
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/gan.jpg" width=200 height=200 >
<h3>ac-gan</h3>
<img src="https://github.com/PRCinguhou/GAN-ACGAN-Pytorch/blob/master/readme_img/acgan.jpg" width=200 height=200 >
</p>  
