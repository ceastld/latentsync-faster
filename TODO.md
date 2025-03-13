
stream inference
* audio2feat
* faster whipser stream inference
* face_preprocess
* face_postprocess
* restore_img faster in affine_transform.py
* may use [kornia](https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.warp_affine)


# speed test
* ori unet
```
Batch 5 processing completed, total time: 0.68 seconds
   Step times:
  - Read frames: 0.01 seconds
  - Facial preprocessing: 0.27 seconds
  - Audio feature preparation: 0.02 seconds
  - Diffusion inference: 0.36 seconds
  - Prepare latent variables: 0.00 seconds
  - Set timesteps: 0.00 seconds
  - Prepare audio embeddings: 0.00 seconds
  - Prepare face masks: 0.00 seconds
  - Prepare mask latent variables: 0.02 seconds
  - Prepare image latent variables: 0.01 seconds
  - Denoising process: 0.26 seconds
  - Average denoising per step: 0.09 seconds
  - Decode and post-process: 0.08 seconds
```

* torch.compile unet
```
Batch 3 processing completed, total time: 0.66 seconds
   Step times:
  - Read frames: 0.01 seconds
  - Facial preprocessing: 0.30 seconds
  - Audio feature preparation: 0.03 seconds
  - Diffusion inference: 0.31 seconds
  - Prepare latent variables: 0.00 seconds
  - Set timesteps: 0.00 seconds
  - Prepare audio embeddings: 0.00 seconds
  - Prepare face masks: 0.00 seconds
  - Prepare mask latent variables: 0.02 seconds
  - Prepare image latent variables: 0.01 seconds
  - Denoising process: 0.10 seconds
  - Average denoising per step: 0.03 seconds
  - Decode and post-process: 0.18 seconds
```

* torch.compile unet and vae
```
Batch 5 processing completed, total time: 0.64 seconds
   Step times:
  - Read frames: 0.01 seconds
  - Facial preprocessing: 0.28 seconds
  - Audio feature preparation: 0.02 seconds
  - Diffusion inference: 0.31 seconds
  - Prepare latent variables: 0.00 seconds
  - Set timesteps: 0.00 seconds
  - Prepare audio embeddings: 0.00 seconds
  - Prepare face masks: 0.00 seconds
  - Prepare mask latent variables: 0.02 seconds
  - Prepare image latent variables: 0.01 seconds
  - Denoising process: 0.10 seconds
  - Average denoising per step: 0.03 seconds
  - Decode and post-process: 0.18 seconds
```