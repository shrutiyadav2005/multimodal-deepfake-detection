# Multimodal Deepfake Datasets Guide

This document describes the benchmark datasets supported by this project, their official acquisition procedures, licensing, citations, and data directory structure.

---

## 1. Supported Benchmark Datasets

### A. Video Deepfake Datasets

#### 1. FaceForensics++ (FF++)
- **Description**: Standard benchmark containing 1,000 original video sequences manipulated with four automated state-of-the-art face manipulation methods: Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
- **Manipulation Types**: Face swap, facial reenactment, synthetic generation.
- **Access Policy**: Restricted academic access. Requires filling an application form to obtain download credentials from the Technical University of Munich (TUM).
- **Official Terms & Form**: [FaceForensics GitHub Repository](https://github.com/ondyari/FaceForensics)
- **Acquisition Procedure**:
  1. Fill out the Google Form provided in the FaceForensics repo with your university/research affiliation.
  2. Once approved, you receive an automated python download script (`download-FaceForensics.py`).
  3. Execute:
     ```bash
     python download-FaceForensics.py <destination_path> -d all -c c23 -t videos
     ```
  4. Place extracted files under `data/FaceForensics/`.

#### 2. Celeb-DF (v2)
- **Description**: A large-scale challenging deepfake video dataset containing 590 original YouTube videos of celebrities and 5,639 high-quality synthetic deepfake videos generated using an improved DeepFake algorithm.
- **Access Policy**: Academic research only.
- **Official Repository**: [Celeb-DF GitHub](https://github.com/danmohaha/celeb-deepfakeforensics)
- **Acquisition Procedure**:
  1. Request access via the official agreement linked in the repository.
  2. Unpack the dataset into `data/Celeb-DF/` with subfolders `Celeb-real`, `Celeb-synthesis`, and `YouTube-real`.

---

### B. Audio Deepfake & Spoofing Datasets

#### 1. ASVspoof (2019 / 2021)
- **Description**: The benchmark dataset for automatic speaker verification spoofing and countermeasure evaluation. Includes Logical Access (LA) containing speech synthesized using state-of-the-art text-to-speech (TTS) and voice conversion (VC) algorithms.
- **Access Policy**: Open access under Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Official Portal**: [ASVspoof.org](https://www.asvspoof.org/) / [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
- **Acquisition Procedure**:
  ```bash
  # Download ASVspoof 2019 LA training & development partitions
  curl -O https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
  unzip LA.zip -d data/ASVspoof/
  ```

---

### C. Image Deepfake Datasets

#### 1. Deepfake Detection Challenge (DFDC) & 140k Real and Fake Faces
- **Description**: High-resolution real and AI-generated portraits (StyleGAN, StyleGAN2, Diffusion-based faces).
- **Access Policy**: Available via Kaggle / Open Academic mirrors.
- **Directory**: Place under `data/images/real/` and `data/images/fake/`.

---

## 2. Directory Structure for Training

```text
data/
├── README.md
├── download_datasets.py        # Automated helper script for public datasets
├── sample/                    # Verified sample media for zero-setup demo & testing
│   ├── images/
│   │   ├── sample_real.jpg
│   │   └── sample_fake.jpg
│   ├── videos/
│   │   ├── sample_real.mp4
│   │   └── sample_fake.mp4
│   └── audio/
│       ├── sample_real.wav
│       └── sample_fake.wav
├── raw/                       # Staging area for raw extracted downloads
└── processed/                 # Train/Val/Test splits with identity-leakage protection
    ├── images/
    │   ├── train/ (real/fake)
    │   ├── val/   (real/fake)
    │   └── test/  (real/fake)
    ├── audio/
    │   ├── train/ (real/fake)
    │   ├── val/   (real/fake)
    │   └── test/  (real/fake)
    └── video_frames/
        ├── train/
        ├── val/
        └── test/
```

---

## 3. Data Leakage Prevention

When preparing splits:
1. **Identity Disjointness**: The same subject/celebrity identity MUST NOT appear in both the training set and the validation or test set.
2. **Video Sequence Separation**: Consecutive frames from the same source video must strictly reside within a single split to avoid optimistic bias.

---

## 4. Academic Citations

```bibtex
@inproceedings{roessler2019faceforensicspp,
  author = {Andreas R{\"o}ssler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
  title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{Celeb_DF_cvpr20,
  author = {Yuezun Li and Xin Yang and Pu Sun and Honggang Qi and Siwei Lyu},
  title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

@inproceedings{todisco2019asvspoof,
  author = {Massimiliano Todisco and Xin Wang and Ville Vestman and Md Sahidullah and Héctor Delgado and Andreas Nautsch and Junichi Yamagishi and Nicholas Evans and Tomi Kinnunen and Kong Aik Lee},
  title = {{ASVspoof 2019}: Future Horizons in Spoofed and Fake Audio Detection},
  booktitle = {Interspeech},
  year = {2019}
}
```
