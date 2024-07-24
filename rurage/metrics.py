import pandas as pd 

def compute_nli_score(labels: pd.Series, mtype: str) -> float:
    counts = labels.value_counts()
        norm = self.golden_set_cfg.golden_set.shape[0]
        for id, label in self._nli_labels.items():
            if label == mtype:
                return counts[int(id)] / norm

def compute_similarity(context, answer):
    # Implementation of similarity computation
    pass

def compute_overlap(reference, candidate):
    # Implementation of overlap computation
    pass

def calculate_rouge(reference, candidate):
    # Implementation of ROUGE calculation
    pass

def calculate_bleu(reference, candidate):
    # Implementation of BLEU calculation
    pass
