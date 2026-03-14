from sklearn.metrics import accuracy_score

def evaluate(predictions, labels):
    acc = accuracy_score(labels, predictions)
    print("Accuracy:", acc)
