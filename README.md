# 🐞 Bug Detection and Fixing - The HAWKS

**Intel® Unnati Industrial Training Project**  
**Institution:** Karunya Institute of Technology and Sciences

## 🚀 Overview

This project presents an AI-powered solution for detecting and fixing bugs in source code using a **Bidirectional LSTM** model for classification and **Hugging Face's StarCoder** for code correction. The entire solution is deployed as a user-friendly **Streamlit** web application for real-time interaction.

## 👥 Team Members

| Name           | Responsibility       |
|----------------|----------------------|
| Moses Paul A   | Data Preprocessing   |
| Balaji A       | Model Building       |
| Madanika N     | Model Deployment     |

## 🧠 Abstract

Our system classifies code snippets as either buggy or bug-free using a BiLSTM model trained on preprocessed code data. Upon detecting a bug, it utilizes the Hugging Face StarCoder model to generate a corrected version. The application enables developers to identify and fix bugs with minimal manual effort, offering a faster debugging experience via an intuitive Streamlit interface.

---

## 🛠️ Technologies Used

- **Machine Learning (ML)** – Bug classification with BiLSTM
- **Deep Learning (DL)** – Model architecture using TensorFlow/Keras
- **Natural Language Processing (NLP)** – Tokenization, sequence padding
- **Hugging Face API** – For code correction using StarCoder
- **Streamlit** – Web app frontend
- **Scikit-learn & Imbalanced-learn** – Preprocessing and oversampling
- **Pickle** – Tokenizer persistence
- **Python** – Core language

---

## 📊 Dataset & Preprocessing

- Dataset includes labeled code snippets: buggy (`1`) or bug-free (`0`)
- Tokenization via TensorFlow’s Tokenizer
- Padding applied to ensure uniform sequence lengths
- Oversampling with `RandomOverSampler` to handle class imbalance

You can find the dataset [here](./Dataset.csv).


---

## 🔁 Workflow

1. User submits code via the Streamlit interface.
2. The BiLSTM model classifies it as buggy or bug-free.
3. If buggy, the code is passed to the **StarCoder** model:
   ```
   Fix this buggy code: <user_code>
   Fixed Code:
   ```
4. The fixed output is returned and displayed to the user.

---

## 📂 Project Structure

```bash
.
├── Output_images/                    # Output screenshots
│   ├── Output-1.png
│   ├── Output-2.png
│   ├── Output-2(continuation).png
├── Sample_Testing_code_files/        # Test code samples
│   ├── BugFree_Testcode.py
│   ├── Buggy_Testcode.py
├── Dataset.csv                       # Labeled dataset
├── deploy.py                         # Streamlit deployment script
├── model_building.py                 # Model training and saving
├── THE HAWKS -BUGGY CODE FIX.pdf     # Project Presentation
├── THE HAWKS Final Report.pdf        # Project Report
├── tokenizer.pkl                     # Saved tokenizer object
├── Trained_model.h5                  # Trained BiLSTM model
```

---

## 💻 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the web app:
   ```bash
   streamlit run deploy.py
   ```

---

## 🌐 Deployment

The Streamlit application can be deployed on:
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [AWS EC2](https://aws.amazon.com/ec2)

---

## 📎 Resources

- [TensorFlow](https://www.tensorflow.org/)
- [Hugging Face StarCoder](https://huggingface.co/bigcode/starcoder)
- [Streamlit Docs](https://docs.streamlit.io/)

---

> _This project was developed as part of the Intel® Unnati Industrial Training Program, aiming to build real-world AI solutions with practical impact._
