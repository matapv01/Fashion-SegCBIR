# Fashion-SegCBIR

Fashion-SegCBIR is an advanced Content-Based Image Retrieval (CBIR) system tailored for fashion images. It leverages state-of-the-art segmentation and deep learning techniques to enable efficient and accurate retrieval of visually similar fashion items.

## Features

- **Image Segmentation:** Automatically detects and segments clothing items from complex backgrounds using pretrained deep learning models.
- **Feature Extraction:** Extracts robust visual features (color, texture, shape) from segmented items using CNN-based architectures.
- **Similarity Search:** Computes similarity scores between images using extracted features, enabling fast and relevant retrieval.
- **User Interface:** Provides a web-based interface for uploading queries, browsing results, and visualizing segmentation masks.
- **Extensible Pipeline:** Modular design allows easy integration of new models or feature extraction methods.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/matapv01/Fashion-SegCBIR.git
    cd Fashion-SegCBIR
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download pretrained models:**  
   Place segmentation and feature extraction models in the `models/` directory. Refer to the documentation for download links.

## Usage

1. **Prepare your dataset:**  
   Place fashion images in the `data/` folder. Supported formats: JPG, PNG.
2. **Run segmentation and feature extraction:**  
    ```bash
    python segment_and_extract.py
    ```
   This script processes images, segments fashion items, and extracts features. Outputs are saved in `data/features/`.
3. **Start the retrieval system:**  
    ```bash
    python app.py
    ```
   The FastAPI server will launch the web interface.
4. **Access the interface:**  
   Open your browser and go to [http://localhost:5000](http://localhost:5000).
5. **Query images:**  
   Upload a fashion image to retrieve visually similar items from your dataset.

## Project Structure

```
Fashion-SegCBIR/
├── data/                # Fashion image dataset and extracted features
├── models/              # Pretrained segmentation and feature models
├── segment_and_extract.py # Segmentation and feature extraction script
├── app.py               # Main FastAPI application
├── requirements.txt     # Python dependencies
├── static/              # UI assets (CSS, JS, images)
├── templates/           # HTML templates for web interface
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch or TensorFlow (for deep learning models)
- OpenCV (image processing)
- FastAPI (web server)
- NumPy, scikit-learn (feature processing)
- Jinja2 (templating for UI)

## Customization

- **Model Selection:**  
  You can swap segmentation or feature extraction models by updating the `models/` directory and modifying `segment_and_extract.py`.
- **Dataset Expansion:**  
  Add more images to `data/` and rerun the extraction script to update features.
- **UI Modification:**  
  Edit files in `templates/` and `static/` to customize the web interface.


## License

This project is licensed under the MIT License. See `LICENSE` for details.
