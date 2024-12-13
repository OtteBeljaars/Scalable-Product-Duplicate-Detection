import re
import numpy as np
import json
import random
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from nltk.util import ngrams

def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"(inches|inch)", "inch", text)
    text = re.sub(r"(hz|hertz)", "hz", text)

    return text.strip()

def preprocess_feature(value):
    try:
        value = value.strip().lower()
        value = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", value)
        value = re.sub(r"[^\w\s.]", "", value)
        value = re.sub(r"(inches|inch)", "inch", value)
        value = re.sub(r"(hz|hertz)", "hz", value)
        value = re.sub(r"(lb|lbs|pounds)", "lbs", value)
        value = re.sub(r"(lbs[^)]*)[^\w\s]*", "lbs", value)
        value = re.sub(r"x", " x ", value)

        if value in {"yes", "true", "on"}:
            return "yes"
        elif value in {"no", "false", "off"}:
            return "no"

        return value.strip()
    except (AttributeError, ValueError):
        return hash(value) % 10000

def extract_brand_from_title(title):
    known_brands = [
                "Affinity", "Coby", "Compaq", "Craig", "Elo", "Epson", "HANNspree", "HP", "Haier",
                "Hannspree", "Hisense", "JVC", "LG", "Magnavox", "NEC", "Naxa", "Panasonic",
                "Philips", "Proscan", "Pyle", "RCA", "SIGMAC", "Samsung", "Sansui", "Sanyo",
                "Sceptre", "Seiki", "Sharp", "Sony", "SunBriteTV", "SuperSonic", "TCL", "Toshiba",
                "UpStar", "VIZIO", "ViewSonic", "Westinghouse"
             ]
    if not isinstance(title, str):
         return "unknown"
    for brand in known_brands:
         if brand.lower() in title:
            return brand.lower()
    return "unknown"

def process_data(data):
    processed_data = []
    for modelID in data:
        for incident in data[modelID]:
            title = incident.get('title', '')
            incident['title'] = preprocess_text(title)

            if "featuresMap" in incident:
                for feature, value in incident["featuresMap"].items():
                    incident["featuresMap"][feature] = preprocess_feature(value)

            processed_data.append(incident)
    return processed_data

def is_model_word(token):
    return sum(bool(re.search(modelword, token)) for modelword in [r'[a-zA-Z]', r'[0-9]', r'[^a-zA-Z0-9]']) >= 2

def extract_model_words(title):
    if not isinstance(title, str):
        return []
    return [word for word in re.findall(r'\S+', title) if is_model_word(word)]

def create_binary_vectors(processed_data, extract_model_words):

    prioritized_features = ["Aspect Ratio", "UPC", "Width", "Screen Size", "Screen Size Class", "Vertical Resolution",
                           "Maximum Resolution", "Height", "HDMI Inputs"]

    def filter_features(incident):
        return {
            feature: incident["featuresMap"][feature]
            for feature in prioritized_features or []
            if "featuresMap" in incident and feature in incident["featuresMap"]
        }

    all_words = set()
    product_vectors = []

    for incident in processed_data:
        product_words = set()

        title_model_words = extract_model_words(incident.get("title", ""))
        product_words.update(title_model_words)

        filtered_features = filter_features(incident)
        product_words.update(f"{feature}: {value}" for feature, value in filtered_features.items())

        all_words.update(product_words)
        product_vectors.append(product_words)

    vocabulary = sorted(all_words)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    binary_vectors = np.zeros((len(product_vectors), len(vocabulary)), dtype=int)
    for i, product_words in enumerate(product_vectors):
        for word in product_words:
            binary_vectors[i, word_to_index[word]] = 1

    return binary_vectors, vocabulary

def generate_permutations(num_hashes, num_vectors):
    np.random.seed(12)
    return np.array([np.random.permutation(num_vectors) for _ in range(num_hashes)])


def minhash_signature(binary_matrix):
    num_rows, num_products = binary_matrix.shape
    num_hashes = num_rows // 2

    permutations = generate_permutations(num_hashes, num_rows)

    signature_matrix = np.full((num_hashes, num_products), np.inf)

    for i in range(num_hashes):
        permutation = permutations[i]
        for j in range(num_products):
            permuted_vector = binary_matrix[permutation, j]
            minhash = np.where(permuted_vector == 1)[0]
            if minhash.size > 0:
                signature_matrix[i, j] = minhash[0]

    return signature_matrix

def lsh(signature_matrix):
    num_hashes, num_products = signature_matrix.shape
    num_bands = 600
    num_rows = num_hashes // num_bands
    candidate_pairs = []
    candidate_buckets = defaultdict(list)

    for band in range(num_bands):
        start_row = band * num_rows
        end_row = (band + 1) * num_rows
        band_hashes = signature_matrix[start_row:end_row, :]

        for col in range(band_hashes.shape[1]):
            band_key = tuple(band_hashes[:, col])
            candidate_buckets[band_key].append(col)

    for bucket in candidate_buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket) - 1):
                for j in range(i + 1, len(bucket)):
                    pair = tuple(sorted([bucket[i], bucket[j]]))
                    candidate_pairs.append(pair)

    return list(set(candidate_pairs))

def compute_qgrams(value1, value2, q=3):
    qgrams_1 = list(ngrams(value1, q))
    qgrams_2 = list(ngrams(value2, q))
    intersection = len(set(qgrams_1).intersection(qgrams_2))
    union = len(set(qgrams_1).union(qgrams_2))
    similarity_qgrams = intersection / union if union > 0 else 0

    return similarity_qgrams

def compute_model_words_similarity(value1, value2, extract_model_words):
    model_words1 = set(extract_model_words(value1))
    model_words2 = set(extract_model_words(value2))
    intersection_words = len(model_words1 & model_words2)
    union_words = len(model_words1 | model_words2)
    similarity_model_words = intersection_words / union_words if union_words > 0 else 0

    return similarity_model_words

def compute_title_similarity(title1, title2, extract_model_words):
    model_words1 = set(extract_model_words(title1))
    model_words2 = set(extract_model_words(title2))
    intersection_words = len(set(model_words1).intersection(model_words2))
    union_words = len(set(model_words1).union(model_words2))
    similarity_model_title = intersection_words / union_words if union_words > 0 else 0

    return similarity_model_title


def compute_dissimilarity_matrix(processed_data, signature_matrix, extract_model_words, q=3, model_word_weight=0.3, qgram_weight=0.4):
    num_products = len(processed_data)
    dissimilarity_matrix = np.full((num_products, num_products), 9999.0)
    candidate_pairs = lsh(signature_matrix)

    valid_candidate_pairs = {(i, j) for i, j in candidate_pairs if i < num_products and j < num_products}
    for i, j in valid_candidate_pairs:
        product1 = processed_data[i]
        product2 = processed_data[j]

        brand1 = extract_brand_from_title(product1.get("title", ""))
        brand2 = extract_brand_from_title(product2.get("title", ""))
        shop1 = product1.get("shop", "unknown")
        shop2 = product2.get("shop", "unknown")

        if brand1 == brand2 and shop1 != shop2:
            title1 = product1.get("title", "")
            title2 = product2.get("title", "")
            title_similarity = compute_title_similarity(title1, title2, extract_model_words)

            total_similarity_qgram = 0
            total_similarity_model_word = 0
            keys1 = list(product1.keys())
            keys2 = list(product2.keys())
            for key1 in keys1:
                for key2 in keys2:
                    if key1 == key2:
                        value1 = product1[key1]
                        value2 = product2[key2]
                        key_similarity = compute_qgrams(key1, key2, q)
                        qgram_similarity = compute_qgrams(value1, value2, q)
                        if key_similarity >= 0.7:
                            total_similarity_qgram += qgram_weight * qgram_similarity

            model_word_similarity = compute_model_words_similarity(value1, value2, extract_model_words)
            total_similarity_model_word += model_word_weight * model_word_similarity

            weighted_similarity = (0.15* title_similarity + 0.1*total_similarity_qgram + 0.1*total_similarity_model_word)
            dissimilarity_matrix[i, j] = dissimilarity_matrix[j, i] = 1 - weighted_similarity

    np.fill_diagonal(dissimilarity_matrix, 0)

    return dissimilarity_matrix

def hierarchical_clustering(processed_data, dissimilarity_matrix, threshold):
    condensed_dissimilarity = squareform(dissimilarity_matrix)
    linkage_matrix = linkage(condensed_dissimilarity, method='single')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    grouped_clusters = defaultdict(list)
    for product_idx, cluster_id in enumerate(clusters):
        product = processed_data[product_idx]
        grouped_clusters[cluster_id].append(product)

    return grouped_clusters

def get_bootstrap(processed_data, perc=0.63):
    return random.choices(processed_data, k=int(len(processed_data) * perc))

def evaluate(processed_data, dissimilarity_matrix, threshold=0.8):
    grouped_clusters = hierarchical_clustering(processed_data, dissimilarity_matrix, threshold)

    found_pairs = set()
    for cluster_id, cluster in grouped_clusters.items():
        if len(cluster) > 1:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    found_pairs.add((cluster[i]['modelID'], cluster[j]['modelID']))

    actual_pairs = set()
    for i, item in enumerate(processed_data):
        for j in range(i + 1, len(processed_data)):
            if item["modelID"] == processed_data[j]["modelID"]:
                actual_pairs.add((item["modelID"], processed_data[j]["modelID"]))


    Df = len(found_pairs.intersection(actual_pairs))
    Nc = len(found_pairs)
    Dn = len(actual_pairs)

    pair_quality = Df / Nc if Nc > 0 else 0
    pair_completeness = Df / Dn if Dn > 0 else 0

    TP_set = found_pairs.intersection(actual_pairs)
    FP_set = found_pairs.difference(actual_pairs)
    FN_set = actual_pairs.difference(found_pairs)

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    F1_star = 2 * pair_quality * pair_completeness / (pair_quality + pair_completeness) if (pair_quality + pair_completeness) > 0 else 0

    print(f"Pair Quality: {pair_quality}")
    print(f"Pair Completeness: {pair_completeness}")
    print(f"F1: {F1}")
    print(f"F1* Score: {F1_star}")

    return F1, pair_quality, pair_completeness, F1_star

def main():
    file_path = "Dataset.json"
    data = load_data(file_path)
    processed_data = process_data(data)

    print("\nModel IDs for the cases:")
    for product in processed_data:
        print(f"modelID: {product['modelID']}")

    bootstrap_results = []
    for i in range(5):
        print(f"\nBootstrap {i + 1}:")

        bootstrap_data = get_bootstrap(processed_data, perc=0.63)
        binary_vectors, vocabulary = create_binary_vectors(bootstrap_data, extract_model_words)
        signature_matrix = minhash_signature(binary_vectors)

        dissimilarity_matrix = compute_dissimilarity_matrix(
            bootstrap_data,
            signature_matrix,
            extract_model_words,
            q=3
        )
        threshold = 0.9
        f1, pair_quality, pair_completeness, f1_star = evaluate(bootstrap_data, dissimilarity_matrix, threshold)

        bootstrap_results.append({
            'bootstrap': i + 1,
            'F1': f1,
            'Pair Quality': pair_quality,
            'Pair Completeness': pair_completeness,
            'F1*': f1_star
        })

    avg_f1 = np.mean([result['F1'] for result in bootstrap_results])
    avg_pair_quality = np.mean([result['Pair Quality'] for result in bootstrap_results])
    avg_pair_completeness = np.mean([result['Pair Completeness'] for result in bootstrap_results])
    avg_f1_star = np.mean([result['F1*'] for result in bootstrap_results])

    print("\nEvaluation results from 5 bootstraps (averaged):")
    print(f"Average F1: {avg_f1}")
    print(f"Average Pair Quality: {avg_pair_quality}")
    print(f"Average Pair Completeness: {avg_pair_completeness}")
    print(f"Average F1* Score: {avg_f1_star}")

if __name__ == "__main__":
    main()