Spam Message Detection
This project is a Spam Message Detection system using machine learning techniques. It uses a dataset of SMS messages labeled as either "spam" or "not spam" to train a model that can classify new messages as spam or not spam.

Table of Contents
Installation

Project Structure

How It Works

Training the Model

Using the Model

License

Installation
To run the application locally, follow these steps:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/spam-detection.git
Navigate to the project directory:

bash
Copy
Edit
cd spam-detection
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy
Edit
venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset spam_dataset.csv (from Kaggle or other sources) and place it in the project directory.

Ensure you have the following Python libraries installed:

pandas

scikit-learn

streamlit

Project Structure
bash
Copy
Edit
spam-detection/
│
├── app.py                  # Main Streamlit application file
├── model.py                # Model training and prediction file
├── spam_dataset.csv        # The dataset file (SMS messages)
├── model.pkl               # Trained model file (generated after training)
├── vectorizer.pkl          # Vectorizer file for transforming text
├── requirements.txt        # List of dependencies
└── README.md               # This file
How It Works
Data Preprocessing: The dataset consists of SMS messages that are labeled as either spam or ham (not spam). The text data is processed using the TfidfVectorizer from scikit-learn to convert text into numerical format. The labels are binary: 1 for spam, 0 for not spam.

Model Training: The dataset is split into training and testing sets. A Multinomial Naive Bayes model is trained on the training set, and the trained model is saved to disk as model.pkl. The vectorizer used to transform the text into numerical features is also saved as vectorizer.pkl.

Model Prediction: After the model is trained, it is used for making predictions. The user can enter a message, and the app will predict whether the message is spam or not. This is done using a simple Streamlit web interface.

Training the Model
To train the model, run the following command in your terminal:

bash
Copy
Edit
python model.py
This will:

Load the dataset (spam_dataset.csv)

Preprocess the text data

Train the Multinomial Naive Bayes model

Save the model as model.pkl

Save the vectorizer as vectorizer.pkl

Using the Model
Once the model is trained, you can use it for predictions through the Streamlit interface. To run the application:

bash
Copy
Edit
streamlit run app.py
This will start a local server and open the application in your default web browser. You can enter any SMS message in the input box and click "Predict" to see whether it's spam or not.

License
This project is licensed under the MIT License - see the LICENSE file for details.