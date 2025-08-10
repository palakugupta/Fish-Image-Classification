# Fish Image Classification  
This project focuses on classifying fish images into multiple species using deep learning techniques. It includes training a Convolutional Neural Network (CNN) from scratch, applying transfer learning with EfficientNetB0, evaluating model performance, and deploying a Streamlit app for real-time fish image classification.

## Dataset
The dataset contains fish images categorized by species. Due to its large size, the dataset is **not included** in this repository. You can find details and instructions for preparing your own dataset in the Jupyter notebook.

## Project Structure
- `notebooks/training_and_evaluation.ipynb` — Jupyter notebook for training, evaluating, and saving models.
- `models/fish_classifier_effnetb0.h5` — Pretrained EfficientNetB0 model weights.
- `app.py` — Streamlit app for uploading images and getting species predictions.
- `requirements.txt` — List of Python packages required to run the project.

## How to Run
### Clone the repository
git clone https://github.com/your_username/Fish-Image-Classification.git
cd Fish-Image-Classification

### Install dependencies
pip install -r requirements.txt

### Run the Streamlit app
streamlit run app.py
Then open the URL provided by Streamlit (usually http://localhost:8501) in your browser.

## Model Performance
- CNN from scratch: accuracy ~XX%
- EfficientNetB0 transfer learning: accuracy ~XX%
For detailed metrics and analysis, check the Jupyter notebook.

## Contact
If you have any questions or want to collaborate, feel free to contact me at palakugupta@gmail.com.




