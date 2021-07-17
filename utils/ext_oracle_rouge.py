import json
import spacy
import scispacy
from collections import defaultdict, Counter


nlp = spacy.load('en_core_sci_lg')


def load_data(data_path):
    with open(data_path, "r") as f:
        for line in f:
            yield json.loads(line)


def get_preprocessed_sentence(sent):
    return ' '.join([token.lemma_ for token in sent if not token.is_stop and not token.is_punct])


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams


def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def get_oracle_sentence(data, n=1):
    reference = get_preprocessed_sentence(nlp(data["citation_text"])).split()
    original_sentences = data["citedAbstract_segmented"]
    preprocessed_cited_abstract_sents = data["citedAbstract_preprocessed"]
    max_rouge = 0
    max_idx = 0

    for idx, sent in enumerate(preprocessed_cited_abstract_sents):
        sent = sent.split()
        # greedy selection (select sentence with maximum of (ROUGE-1 + ROUGE-2))
        rouge = compute_rouge_n(
            sent, reference, n=1) + compute_rouge_n(sent, reference, n=2)
        if rouge > max_rouge:
            max_rouge = rouge
            max_idx = idx

    return original_sentences[max_idx]


if __name__ == '__main__':
    dataset = load_data("data/ext-oracle/test.jsonl")

    with open("experiments/oracle/oracle_rouge.result", "w") as fout:
        for data in dataset:
            fout.write(get_oracle_sentence(data, n=2))
            fout.write("\n")
