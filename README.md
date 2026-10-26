# Email Intent Classification Using BiLSTM

## Project Overview
This academic mini-project uses Deep Learning and Natural Language Processing (NLP) to classify emails into four intent categories: Request, Complaint, Feedback, and Inquiry.

The model uses a Bidirectional Long Short-Term Memory (BiLSTM) network to learn patterns in email text and predict the appropriate intent.

## Objectives
- Classify emails into four intent categories.
- Preprocess email text and remove duplicate records.
- Train and evaluate a BiLSTM deep learning model.
- Predict the intent of a new email.

## Technologies Used
- Python
- TensorFlow and Keras
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)

## Dataset
The dataset contains email text and corresponding intent labels. Duplicate email records were removed during data cleaning.

- Original dataset: 2,000 records
- Cleaned dataset: 1,210 unique email records
- Intent categories: Request, Complaint, Feedback, and Inquiry

## Model Architecture
The model consists of:
1. An Embedding layer
2. A Bidirectional LSTM layer
3. A Dense layer with ReLU activation
4. An output Dense layer with Softmax activation for four-class classification

## Results
The model achieved **100% test accuracy** on the available test set.

The classification report showed precision, recall, and F1-score of 1.00 for all four classes. These results apply to this dataset and test split; further testing on new, independent emails would help assess generalization.

## Project Structure

```text
email-intent-classification-bilstm/
├── Dataset/
│   ├── README.md
│   └── cleaned_email_intent.csv
├── email_intent.csv
├── email_intent_bilstm.keras
├── main.py
└── README.md
```


## How to Run
1. Install Python 3.12.
2. Install the required Python libraries.
3. Run the program:

```bash
py -3.12 main.py
```

## Project Type
Academic Deep Learning Mini-Project

## Author
B.Tech Computer Science and Engineering (AI & ML) Student
