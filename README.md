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
    uv sync
    ```
3. **Download vector database from:**: https://drive.google.com/file/d/1Ff32PYdxlN_OUCtCfLb13xljeDm-cL5e/view?usp=sharing
   - put it into folder vector_database

4. **Create .env**:
   ```
   OPENAI_API_KEY = ""
   OPENAI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
   OPENAI_API_MODEL = "gemini-2.5-flash"
   ```

## Usage

5. **Start the retrieval system:**  
    ```bash
    uv run python app.py
    ```
   The FastAPI server will launch the web interface.
6. **Access the interface:**  
   Open your browser and go to [http://localhost:8000](http://localhost:8000).
7. **Query images:**  
   Upload a fashion image to retrieve visually similar items from your dataset.


## Evaluation
   ```bash
   uv run python eval.py
   ```


## License

This project is licensed under the MIT License. See `LICENSE` for details.
