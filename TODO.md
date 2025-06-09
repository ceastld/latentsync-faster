* parallel and stream inference
* stable audio2feat stream inference, use overlapping window
* face_preprocess speed up, use onnxruntime-gpu
* faster restore_img, 
* restore_img faster by wrapAffine in GPU
* check if each step data is on GPU
* docker https://hub.docker.com/r/pytorch/pytorch/tags

* optimize restore_img for high-resolution images, ROI strategy, pre-calculate affine transformation affected areas, only put necessary parts into GPU for computation

* latentsync 1.5, new model

* v1 L4 25fps bs8, v1.5 A100 25fps bs16

* optimize output when no face detected, directly output original image
* skip processing when pose changes

* potential issue: multiple people face-swapping inference simultaneously



* auto inference when no audio input for over 1s?


Initial Delay ~= 1.5s

---
* use warmup strategy, after model loading, push one frame data to complete LatentSync module warmup, takes about 0.55s
* added FPS controller to control output stream at ~25fps
* change batch_size to 12 to reduce latency and improve stability

## Performance Test Results (240 frames)
* **First frame delay**: 1.383s 
* **Overall FPS**: 25.04 fps ✅ (target achieved)
