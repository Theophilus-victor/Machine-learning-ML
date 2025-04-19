# 📰 Fake News Detection Using Machine Learning

A simple yet effective machine learning project to detect whether a news article is **Real** or **Fake**, built with `PassiveAggressiveClassifier` and deployed using **Streamlit**.

---

## 🚀 Features

- 🔍 Detects fake news based on the news title and body text
- 🧠 Machine Learning model: `PassiveAggressiveClassifier`
- 📊 Real-time prediction with user input
- 💾 Model serialized using `pickle`
- 🖥️ Deployed using **Streamlit** for a clean, interactive UI

---

## 📂 Dataset

- Used a modified version of the Kaggle [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Dataset includes: `title`, `text`, `subject`, `date`
- Target labels: `REAL` = 0, `FAKE` = 1

---

## 🧰 Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
pandas
numpy
scikit-learn
streamlit
```

---

## ⚙️ File Structure

```
.
├── data_preprocessing.py     # Cleans and prepares the dataset
├── model_training.py         # Trains and saves the model + vectorizer
├── main.py                   # Streamlit app for fake news prediction
├── train.csv                 # Dataset (you can add more rows!)
├── model.pkl                 # Trained model file
├── vectorizer.pkl            # TF-IDF Vectorizer
└── README.md                 # You’re reading it!
```

---

## 🧪 How to Run

1. Train the model and generate `model.pkl` and `vectorizer.pkl`:
   ```bash
   python model_training.py
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

3. Open your browser and interact with the UI!

---

## ✨ Future Enhancements

- Add more training data to improve accuracy
- Integrate live news feed API for real-time detection
- Improve UI with animations and dark mode
- Deploy on cloud (e.g., Streamlit Cloud / Heroku / Hugging Face Spaces)

---

## 🧠 Author

**Theophilus Victor KJ**  
Blockchain Dev • ML Enthusiast • Builder  
Made with ❤️, logic, and lots of caffeine ☕
