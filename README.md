# Landcover App

This project uses Streamlit, TensorFlow, and OpenCV for land cover classification and visualization.

## Setup (macOS ARM/Apple Silicon)

1. Install Python 3.11 (recommended).
2. Clone this repository:
   ```sh
   git clone <your-repo-url>
   cd landcover_app
   ```
3. Create and activate a virtual environment:
   ```sh
   python3.11 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

**Important:**
- Do NOT upgrade numpy or opencv-python beyond the versions in requirements.txt.
- Do NOT install `tensorflow-metal` unless you know it works for your setup.
- If you encounter segmentation faults, try commenting out OpenCV (`cv2`) imports.

## Troubleshooting
- If you get segmentation faults, check your package versions and Python version.
- For Apple Silicon, only use the versions specified above.
