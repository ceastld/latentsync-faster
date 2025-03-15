
stream inference
* audio2feat
* faster whipser stream inference
* face_preprocess
* face_postprocess
* restore_img faster in affine_transform.py
* may use [kornia](https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine)

* 排查每一步数据是否在GPU上

# speed test
* ori var and unet
```
process_audio_with_pre:
  Calls: 25
  Avg time (trimmed): 15.33ms

prepare_latents:
  Calls: 25
  Avg time (trimmed): 0.20ms

prepare_masks_and_masked_images:
  Calls: 25
  Avg time (trimmed): 3.09ms

prepare_mask_latents:
  Calls: 25
  Avg time (trimmed): 5.65ms

prepare_image_latents:
  Calls: 25
  Avg time (trimmed): 3.67ms

_denoising_step:
  Calls: 75
  Avg time (trimmed): 86.93ms

decode_latents:
  Calls: 25
  Avg time (trimmed): 3.96ms

_run_diffusion_batch:
  Calls: 25
  Avg time (trimmed): 282.37ms

process_batch:
  Calls: 25
  Avg time (trimmed): 453.20ms
```

* torch.compile unet and vae
```
process_audio_with_pre:
  Calls: 25
  Avg time (trimmed): 15.29ms

prepare_latents:
  Calls: 25
  Avg time (trimmed): 0.20ms

prepare_masks_and_masked_images:
  Calls: 25
  Avg time (trimmed): 3.21ms

prepare_mask_latents:
  Calls: 25
  Avg time (trimmed): 5.85ms

prepare_image_latents:
  Calls: 25
  Avg time (trimmed): 3.72ms

_denoising_step:
  Calls: 75
  Avg time (trimmed): 42.86ms

decode_latents:
  Calls: 25
  Avg time (trimmed): 4.12ms

_run_diffusion_batch:
  Calls: 25
  Avg time (trimmed): 133.83ms

process_batch:
  Calls: 25
  Avg time (trimmed): 398.39ms
```