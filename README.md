# News-Category-Classification-GLOVE--LSTM-
News classification model using Deep Learning (LSTM) to categorize HuffPost headlines into 10 topics. Achieved 77% accuracy using TensorFlow, Keras, and NLTK for text preprocessing (Stemming/Stopwords). Features Bidirectional LSTM layers and balanced class weights. 🚀

# News Category Classification using LSTM

Αυτό το project υλοποιεί ένα μοντέλο Βαθιάς Μάθησης (Deep Learning) για την ταξινόμηση ειδήσεων σε 10 διαφορετικές κατηγορίες με βάση τους τίτλους και τις περιλήψεις τους. Το μοντέλο βασίζεται σε αρχιτεκτονική **LSTM (Long Short-Term Memory)** και έχει αναπτυχθεί με τη χρήση του **TensorFlow/Keras**.

## Δεδομένα (Dataset)
Χρησιμοποιήθηκε το [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) από το Kaggle, το οποίο περιλαμβάνει περίπου 210.000 τίτλους ειδήσεων από το HuffPost. Για τις ανάγκες του project επιλέχθηκαν οι εξής 10 κατηγορίες:
* Politics, Wellness, Entertainment, Travel, Style & Beauty, Parenting, Healthy Living, Queer Voices, Food & Drink, Business.

##  Ροή Εργασιών (Workflow)
1.  **Προεπεξεργασία Κειμένου**: 
    * Καθαρισμός κειμένου (πεζά, αφαίρεση σημείων στίξης, stopwords).
    * Stemming με τη χρήση του `SnowballStemmer` (NLTK).
    * Tokenization και Padding των ακολουθιών.
2.  **Αρχιτεκτονική Μοντέλου**:
    * **Embedding Layer**: Μετατροπή λέξεων σε διανύσματα.
    * **Bidirectional LSTM**: Για την κατανόηση του πλαισίου (context) της πρότασης.
    * **Dropout & Batch Normalization**: Για αποφυγή overfitting και σταθερότητα.
    * **Dense Layers**: Με συνάρτηση ενεργοποίησης Softmax για την ταξινόμηση.
3.  **Εκπαίδευση**: 
    * Χρήση των callbacks `ReduceLROnPlateau` και `ModelCheckpoint`.
    * Διαχείριση της ανισορροπίας των κλάσεων (class imbalance) με `class_weights`.

##  Αποτελέσματα
Το μοντέλο πέτυχε συνολική ακρίβεια (**Accuracy**) **77%** στο test set.
* **Precision**: Εξαιρετικά αποτελέσματα σε κατηγορίες όπως *Parenting* (0.95) και *Queer Voices* (0.85).
* **Εργαλεία**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras.

##  Πώς να το τρέξετε
1. Κατεβάστε το αρχείο `.ipynb`.
2. Βεβαιωθείτε ότι έχετε εγκαταστήσει τις απαραίτητες βιβλιοθήκες:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn nltk scikit-learn
