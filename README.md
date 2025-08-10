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

### Deployment of streamlit on Terminal 

<img width="758" height="925" alt="image" src="https://github.com/user-attachments/assets/1e82a7bc-2fd6-4588-b037-ef1266256b9f" />

### Streamlit on Local URL: http://localhost:8501
<img width="1320" height="395" alt="image" src="https://github.com/user-attachments/assets/952493ae-270d-44c8-bc5e-484162d7725c" />
<img width="1267" height="769" alt="image" src="https://github.com/user-attachments/assets/87b1cd0f-c9c4-4b36-8327-43c633575229" />

#### The model was trained for only one epoch due to limitations imposed by using the free version of Google Colab, which provides access to CPU resources instead of a GPU. CPU-based training is significantly slower compared to GPU acceleration, resulting in prolonged training times. Consequently, training was restricted to a single epoch to avoid excessive runtime and potential session timeouts inherent to the free Colab environment. Training for only one epoch is insufficient for the model to adequately learn from the dataset, leading to underfitting and suboptimal performance. Access to more powerful hardware, such as GPU acceleration, would enable training over multiple epochs and improve model accuracy.

#### As a result of limited training (only one epoch), the model's predictive confidence is relatively low for various classes. For example, predictions for classes such as fish sea_food horse_mackerel (14.37%), fish sea_food black_sea_sprat (14.07%), and animal fish (13.94%) reflect low certainty. This underperformance is directly linked to insufficient training, which prevents the model from effectively distinguishing between classes with higher confidence.





