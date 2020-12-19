import numpy as np
from joblib import load


class InitialClassification:
    def __init__(self, vectorizer_path, model_path, threshold):
        self.vectorizer = load(vectorizer_path)
        self.model = load(model_path)
        self.threshold = threshold
        self.label_dict = {2: 'Прізвище', 1: 'По батькові', 0: 'Ім’я'}

    def classify(self, surname, name, patronymic, id):
        sample = [surname, name, patronymic]
        sample_vectors = self.vectorizer.transform(sample)
        sample_predictions = self.model.predict_proba(sample_vectors)
        predicted_classes = np.argmax(sample_predictions, axis=1)
        sorted_predictions = [x for _, x in sorted(
            zip(predicted_classes, sample))]
        return sorted_predictions


if __name__ == '__main__':
    vectorizer_path = 'model_weights/text_vectorizer.joblib'
    model_path = 'model_weights/vanilla_linreg.joblib'

    classifier = InitialClassification(vectorizer_path, model_path, 0.5)
    preds = classifier.classify('ШЕВЧЕНКО', 'МИКОЛА', 'ІВАНОВИЧ', 0)
    print(preds)

    preds = classifier.classify('МИКОЛА', 'ШЕВЧЕНКО', 'ІВАНОВИЧ', 0)
    print(preds)
