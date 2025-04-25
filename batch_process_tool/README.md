
# LatentSync Data Preprocessing and Batch Inference

## Architecture

* The overall process consists of three stages: raw video collection, video preprocessing, and batch inference
* Each module operates independently, forming a complete processing pipeline

---

## Processing Workflow

* **Raw Video Collection**
  * Place unprocessed videos in the `testset/raw_video` directory

* **Video Preprocessing Tool `process_video_translate.py`**
  * Workflow:
    1. Video Processing: Use FFmpeg to convert videos to the target frame rate (default 25fps)
    2. Audio Extraction: Extract audio from videos and optimize for speech recognition
    3. Speech Recognition: Use Gemini API to convert audio to text
    4. Text Translation: Translate recognized text into multiple target languages
    5. Speech Synthesis: Use ElevenLabs API to convert translated text into audio for each language
  * Output: Creates subdirectories in `testset/preprocess_video`, containing video files and multilingual audio files, for example:
  ```
    testset/
    └── preprocess_video/                     
        ├── video1/                    
        │   ├── video1.mp4              
        │   ├── video1_zh.wav           
        │   ├── video1_en.wav           
        │   ├── video1_es.wav           
        │   ├── video1_fr.wav           
        │   ├── video1_ru.wav           
        │   ├── video1_ja.wav           
        │   ├── video1_ko.wav           
        │   ├── video1_de.wav           
        │   ├── video1_it.wav           
        │   ├── video1_uk.wav           
        │   ├── video1_pt.wav           
        │   ├── video1_pt-br.wav        
        │   ├── video1_tr.wav           
        │   └── video1_hi.wav           
        │
        ├── video2/                     
        │   ├── video2.mp4
        │   ├── video2_zh.wav
        │   ├── video2_en.wav
        │   └── ...                           
        │
        ├── video3/                     
        │   └── ...
        │
        └── ...                               
    ```
  * Key Components:
    * `GeminiSTT`: Speech recognition
    * `GeminiTextTranslator`: Text translation
    * `ElevenLabsTTS`: Text-to-speech conversion

* **Multilingual Batch Inference `batch_inference_multilang.py`**
  * Workflow:
    1. Model Initialization: Load LipsyncContext and LipsyncModel
    2. File Matching: Find video-audio pairs in the preprocessing directory
    3. Batch Processing: Perform lip synchronization for each video-audio pair
  * Output: Files in the `output/enhanced_multilang` directory, with format `video_name_language_code_enhanced.mp4`

* **Single Audio Batch Inference `batch_inference_single_audio.py`**
  * Workflow:
    1. Uses the same model but with a single audio input
    2. Finds all preprocessed videos and performs lip synchronization with the same audio
  * Purpose: Comparative testing to observe the effect of the same audio on different videos
  * Output: Files in a custom output directory, with format `video_name_enhanced.mp4`

* **Output Examples**
```
output/
└── enhanced_multilang/                   
    ├── video1_zh_enhanced.mp4      
    ├── video1_en_enhanced.mp4      
    ├── video1_es_enhanced.mp4      
    ├── ...     
    ├── video2_zh_enhanced.mp4      
    ├── video2_en_enhanced.mp4      
    ├── video2_es_enhanced.mp4      
    ├── ...   
    ├── video3_zh_enhanced.mp4           
    └── ...                            

output/
└── single_audio_test/                    
    ├── video1_enhanced.mp4         
    ├── video2_enhanced.mp4
    └── ...                               
```

---

## Usage Instructions

* **Video Preprocessing**
  ```bash
  python batch_process_tool/process_video_translate.py \
    --input_dir testset/raw_video \
    --output_dir testset/preprocess_video \
  ```

* **Multilingual Batch Inference**
  ```bash
  python batch_process_tool/batch_inference_multilang.py \
    --input_dir testset/preprocess_video \
    --output_dir output \
  ```

* **Single Audio Comparative Batch Inference**
  ```bash
  python batch_process_tool/batch_inference_single_audio.py \
    --input_dir testset/preprocess_video \
    --driving_audio_path path/to/common_audio.wav \
    --output_dir output/single_audio_test \
  ```

---

## Important Notes

* **Directory Structure**
  * Preprocessing directory format: `testset/preprocess_video/videoID_segNumber/`
  * Each subdirectory contains one video file and multiple language audio files
