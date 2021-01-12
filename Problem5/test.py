from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

nr_cuv_diferite = 5000
dim_max = 500
if __name__ == '__main__':
  # Load the dataset
  (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words =
  nr_cuv_diferite)
  model = load_model("model.h5")
  X_test = sequence.pad_sequences(X_test, maxlen=dim_max)
  scores = model.evaluate(X_test, Y_test)
  model.summary()
  print('Loss: %.3f' % scores[0])
  print('Accuracy: %.3f' % scores[1])