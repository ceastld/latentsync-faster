03/08/25: The dependencies are very confusing because of the application of the newest `diffusers` and `transformers`. But it is necessary for `torch.compile`. It can be run by installing every newest dependency seperately now. Later I will try to create a stable requirements file.

03/09/25: The correct requirements have been settled down. Just `sh setup_env.sh`

03/11/25: The inference process has been packaged into several functions in batch mode. Search `core function` to find the key components of face_processor, audio_processor (todo) and diffusion_processor.