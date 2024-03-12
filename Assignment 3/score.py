# score.py
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import expit

def score(text: str, model, vectorizer: TfidfVectorizer, threshold: float) -> Tuple[int, float]:
    """Scores a text using a trained SVC model and a threshold.

    Args:
        text (str): The input text to be scored.
        model (SVC): The trained SVC model.
        vectorizer (TfidfVectorizer): The vectorizer used for transforming the text.
        threshold (float): The threshold for classification.

    Returns:
        Tuple[int, float]: A tuple containing the prediction (1 for True/0 for False) and the propensity score.
    """
    # Vectorize the text
    vectorized_text = vectorizer.transform([text])

    # Get the decision function value (distance from the separating hyperplane)
    decision_function_value = model.decision_function(vectorized_text)[0]

    # Convert the decision function value to a probability using a logistic function
    # Note: This is a common practice for obtaining probabilities from SVM decision function values
    propensity = expit(decision_function_value)

    # Make the prediction based on the threshold
    prediction = propensity > threshold

    return int(prediction), propensity
