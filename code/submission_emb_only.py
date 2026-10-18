"""纯嵌入方法对比"""
import numpy as np

def predict(bank, arrays, checkpoint):
    embeddings = arrays['embeddings']
    predictions = []
    for i in range(len(bank)):
        emb_q = embeddings[i]
        norms = np.linalg.norm(emb_q, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_normalized = emb_q / norms
        cos_sim = np.dot(emb_normalized, emb_normalized.T)
        avg_similarity = cos_sim.sum(axis=1) - 1.0
        predictions.append(int(np.argmax(avg_similarity)))
    return predictions
