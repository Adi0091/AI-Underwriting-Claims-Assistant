def compute_precision_recall(retrieved_sources, expected_sources):
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)

    true_positive = len(retrieved_set & expected_set)
    false_positive = len(retrieved_set - expected_set)
    false_negative = len(expected_set - retrieved_set)

    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)

    return precision, recall
