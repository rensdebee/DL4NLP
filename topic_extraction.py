# # pip install bertopic
# # pip install bertopic[visualization]

from linguistic_utils import get_dataset

import os
import pickle
import numpy as np
import pandas as pd

from umap import UMAP
from bertopic import BERTopic

from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sentence_transformers import SentenceTransformer



def write_list_to_file(filename, data):
    with open(filename, 'w+') as f:
        for items in data:
            f.write('%s\n' %items)
    f.close()
    print('Done')



def write_list_of_lists_to_file(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print('Done')



def read_list_from_file(filename):
    f = open(filename, "r") 
    data = f.read() 
    data_list = [int(line.strip()) for line in data.split("\n") if line.strip()]
    f.close()
    return data_list



def read_list_of_lists_from_file(filename):
    with open(filename, 'rb') as file:
        data_list = pickle.load(file)

    return data_list

  

def dataset_string(dataset_column):
    return [' '.join(item) if isinstance(item, list) else item for item in dataset_column]



def bertopic_fit(dataset):
    sentence_model = SentenceTransformer("all-mpnet-base-v2") # model with best overall performance: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    embeddings = sentence_model.encode(dataset)

    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(dataset, embeddings)
    
    return topic_model, topics, embeddings



def visualize_document_datamap(
    docs: List[str],
    reduced_embeddings: np.ndarray = None,
    custom_labels: Union[bool, List[str]] = False, 
    doc_set_labels: List[int] = None, 
    title: str = "Documents and Topics",
    legend_title: str = "Domain",
    sub_title: Union[str, None] = None,
    width: int = 1200,
    height: int = 1200,
    point_colors: Union[List[str], None] = None,
    alpha: float = 0.75,  
) -> Figure:
    """Visualize documents by document sets in 2D using DataMapPlot with discrete colors and custom legend names.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        custom_labels: List of custom names for each document set (for the legend).
        doc_set_labels: List of labels indicating the document set each document belongs to.
        title: Title of the plot.
        sub_title: Sub-title of the plot.
        width: Width of the figure.
        height: Height of the figure.
        point_colors: Custom colors for each document set.
        alpha: Default transparency level of the points (0.0 to 1.0).
        last_color_alpha: Transparency level of the points for the last color.
        **datamap_kwds: Further keyword arguments for DataMapPlot.

    Returns:
        figure: A Matplotlib Figure object.
    """

    unique_labels = sorted(set(doc_set_labels))
    reddit_label = unique_labels[-1]
    doc_set_names = custom_labels

    topic_per_doc = doc_set_labels

    df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    df["doc"] = docs
    df["topic"] = topic_per_doc

    # map labels and colors
    label_color_mapping = {label: point_colors[i] for i, label in enumerate(unique_labels)}
    label_name_mapping = {label: doc_set_names[i] for i, label in enumerate(unique_labels)}
    doc_colors = pd.Series(topic_per_doc).map(label_color_mapping).values

    # create figure
    figure, axes = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    axes.set_xticks([])
    axes.set_yticks([])

    # all except reddit
    non_last_mask = np.array(doc_set_labels) != reddit_label
    axes.scatter(
        reduced_embeddings[non_last_mask, 0], reduced_embeddings[non_last_mask, 1],
        c=doc_colors[non_last_mask], 
        s=50, alpha=alpha 
    )
    
    # reddit
    last_mask = np.array(doc_set_labels) == reddit_label
    axes.scatter(
        reduced_embeddings[last_mask, 0], reduced_embeddings[last_mask, 1],
        c=doc_colors[last_mask],
        s=50, alpha=0.5
    )

    axes.set_xlim([-6, 5])

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color_mapping[label], 
                                  markersize=10, label=label_name_mapping[label]) 
                       for label in unique_labels]
    axes.legend(handles=legend_elements, title=legend_title, loc="upper right")

    axes.set_title(title, fontsize=14)
    if sub_title:
        axes.set_title(sub_title, fontsize=10, loc="center")

    return figure



if __name__ == "__main__":
    m_ds = dataset_string(get_dataset(['medicine'])['llama_answers'])
    f_ds = dataset_string(get_dataset(['finance'])['llama_answers'])
    o_ds = dataset_string(get_dataset(['open_qa'])['llama_answers'])
    w_ds = dataset_string(get_dataset(['wiki_csai'])['llama_answers'])
    r_ds = dataset_string(get_dataset(['reddit_train', 'reddit_test'])['llama_answers'])
    full_ds = m_ds + f_ds + o_ds + w_ds + r_ds

    file_path = "topic_ex/reduced_embeddings_finance_medicine.pkl"

    if os.path.exists(file_path):
        reduced_embeddings = read_list_of_lists_from_file("topic_ex/reduced_embeddings_finance_medicine.pkl")
    else:
        _, _, embeddings = bertopic_fit(full_ds)
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine").fit(embeddings)
        reduced_embeddings = umap_model.embedding_
        write_list_of_lists_to_file("topic_ex/reduced_embeddings_finance_medicine.pkl", reduced_embeddings)
    
    labels = [0]*len(m_ds) + [1]*len(f_ds) + [2]*len(o_ds) + [3]*len(w_ds) + [4]*len(r_ds)
    doc_set_names = ["Medicine", "Finance", "Open QA", "Wiki Csai", "Reddit"]

    figure = visualize_document_datamap(
        full_ds,
        reduced_embeddings,
        doc_set_labels=labels,
        title="Domain Overlap Based on Llama Answer Topics",
        custom_labels=doc_set_names,
        width= 1200,
        height = 1200,
        point_colors = ["#2ee5e8", "#e82ee5", "#e6d724", "#2ee84d", "#fe7f7f"]
    )
    figure.savefig("topic_ex/plot_datamap.png", dpi=300, bbox_inches='tight')