
stream inference
* audio2feat
* faster whipser stream inference
* face_preprocess
* face_postprocess
* restore_img faster in affine_transform.py
* may use [kornia](https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine)

* 排查每一步数据是否在GPU上

* docker https://hub.docker.com/r/pytorch/pytorch/tags

* 串行 14.7 fps
* 并行 17.2 fps

* 大分辨率图像的 restore_img 优化，ROI 策略，提前计算出仿射变换影响区域，只将需要的部分放入GPU进行计算

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