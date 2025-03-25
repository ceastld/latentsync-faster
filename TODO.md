* parallel and stream inference
* stable audio2feat stream inference, use overlapping window
* face_preprocess speed up, use onnxruntime-gpu
* faster restore_img, 
* restore_img faster by wrapAffine in GPU
* 排查每一步数据是否在GPU上
* docker https://hub.docker.com/r/pytorch/pytorch/tags

* 大分辨率图像的 restore_img 优化，ROI 策略，提前计算出仿射变换影响区域，只将需要的部分放入GPU进行计算

* latentsync 1.5, new model

* v1 L4 25fps bs8, v1.5 A100 25fps bs16

* TODO: 优化没有识别到人脸情况下的输出，直接输出原始图片


# speed test
* ori var and unet
```
prepare_face_batch:
  Calls: 25
  Avg time (trimmed): 90.70ms

process_batch:
  Calls: 25
  Avg time (trimmed): 372.00ms

restore_batch:
  Calls: 25
  Avg time (trimmed): 77.27ms
```

* torch.compile unet and vae
```
prepare_face_batch:
  Calls: 25
  Avg time (trimmed): 92.52ms

process_batch:
  Calls: 25
  Avg time (trimmed): 318.16ms

restore_batch:
  Calls: 25
  Avg time (trimmed): 79.58ms
```