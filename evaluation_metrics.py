# evaluation_metrics.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import evaluate
from bert_score import score as bert_score

# Load Hugging Face evaluation metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")

def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def embedding_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def evaluate_text(reference, prediction):
    metrics = {}

    # BLEU
    metrics["BLEU"] = bleu.compute(predictions=[prediction], references=[reference])["bleu"]

    # METEOR
    metrics["METEOR"] = meteor.compute(predictions=[prediction], references=[reference])["meteor"]

    # ROUGE-L
    metrics["ROUGE-L"] = rouge.compute(predictions=[prediction], references=[reference])["rougeL"]

    # ChrF++
    metrics["ChrF++"] = chrf.compute(predictions=[prediction], references=[reference])["score"]

    # Jaccard Similarity
    metrics["Jaccard Similarity"] = jaccard_similarity(reference, prediction)

    # Cosine Similarity
    metrics["Cosine Similarity"] = embedding_cosine_similarity(reference, prediction)

    # BERTScore
    P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
    metrics["BERTScore Precision"] = float(P.mean())
    metrics["BERTScore Recall"] = float(R.mean())
    metrics["BERTScore F1"] = float(F1.mean())

    return metrics
