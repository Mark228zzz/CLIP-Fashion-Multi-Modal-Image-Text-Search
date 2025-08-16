# CLIP Fashion: Multi-Modal Image-Text Search

This project implements a multi-modal image-text search system using CLIP (Contrastive Language–Image Pretraining) and lightweight fine-tuning on a fashion product dataset.

## Features
- Zero-shot and fine-tuned CLIP evaluation
- Image and text retrieval with Recall@K and nDCG@K metrics
- Data preprocessing and augmentation
- Training and evaluation scripts in a Jupyter notebook
- Gradio app for interactive demo (see `app/`)

## Dataset
[Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

## Project Structure
```
CLIP_fashion/
├── app/                # Gradio app and API
├── data/               # CSVs and processed data
├── models/             # Saved model checkpoints
├── notebook/           # Main notebook
├── requirements.txt    # Python dependencies
└── README.md           # Project info
```

## Quick Start
1. Clone the repo:
   ```bash
   git clone Mark228zzz/CLIP-Fashion-Multi-Modal-Image-Text-Search
   cd CLIP-Fashion-Multi-Modal-Image-Text-Search
   ```

2. Setup the venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook:

   Open `notebook/MultimodelImageTextSearch.ipynb` in Jupyter or VS Code.
   > **It will run the entire pipeline from data loading to evaluation and training so be careful with your resources of RAM and GPU.**

5. (Optional) Launch the Gradio app:
   ```bash
   python app/app.py
   ```

## Usage
- The notebook covers data download, preprocessing, model training, evaluation, and test set reporting.
- The Gradio app provides a simple UI for searching images by text and vice versa.

## Citation
If you use this project, please cite the original CLIP paper: https://arxiv.org/abs/2103.00020

## License
This project is licensed under the MIT License with an additional clause: commercial sale of this software is not permitted. See the LICENSE file for details.
