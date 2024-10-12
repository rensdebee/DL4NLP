# install BERTopic if necessary
# pip install bertopic
# pip install bertopic[visualization]

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



def write_list_of_lists_to_file(filename, data):
    '''
    Writes a lists of lists (data) to filename
    '''
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print('Done')



def read_list_of_lists_from_file(filename):
    '''
    Retrieve and return list of lists from filename
    '''
    with open(filename, 'rb') as file:
        data_list = pickle.load(file)

    return data_list

  

def dataset_string(dataset_column):
    '''
    Change and return dataset to strings
    '''
    return [' '.join(item) if isinstance(item, list) else item for item in dataset_column]



def bertopic_fit(dataset):
    '''
    Perform BERTopic on a dataset
    '''
    sentence_model = SentenceTransformer("all-mpnet-base-v2") # model with best overall performance: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    embeddings = sentence_model.encode(dataset)

    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(dataset, embeddings)
    
    return topic_model, topics, embeddings



def scatter_plot(
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
    """
    Creates a scatter plot from the reduced embeddings, as rechieved from BERTopic
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

    axes.set_xlim([2, None])

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color_mapping[label], 
                                  markersize=10, label=label_name_mapping[label]) 
                       for label in unique_labels]
    axes.legend(handles=legend_elements, title=legend_title, loc="upper right", title_fontsize=20, prop={'size': 20})

    axes.set_title(title, fontsize=14)
    if sub_title:
        axes.set_title(sub_title, fontsize=10, loc="center")

    return figure


'''
Code from BERTopic but slightly modified
plotting -> _datamap.py
'''
from warnings import warn

try:
    import datamapplot
except ImportError:
    warn("Data map plotting is unavailable unless datamapplot is installed.")

    # Create a dummy figure type for typing
    class Figure(object):
        pass

def visualize_document_datamap(
    topic_model,
    docs: List[str],
    topics: List[int] = None,
    reduced_embeddings: np.ndarray = None,
    custom_labels: Union[bool, str] = False,
    title: str = "Documents and Topics",
    sub_title: Union[str, None] = None,
    width: int = 1200,
    height: int = 1200,
    **datamap_kwds,
) -> Figure:
    """Visualize documents and their topics in 2D as a static plot for publication using
    DataMapPlot.

    Arguments:
        topic_model:  A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`. Documents not in these topics will be shown
                as noise points.
        embeddings:  The embeddings of all documents in `docs`.
        reduced_embeddings:  The 2D reduced embeddings of all documents in `docs`.
        custom_labels:  If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        sub_title: Sub-title of the plot.
        width: The width of the figure.
        height: The height of the figure.
        **datamap_kwds:  All further keyword args will be passed on to DataMapPlot's
                         `create_plot` function. See the DataMapPlot documentation
                         for more details.

    Returns:
        figure: A Matplotlib Figure object.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_document_datamap(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_document_datamap(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    fig.savefig("path/to/file.png", bbox_inches="tight")
    ```
    <img src="../../getting_started/visualization/datamapplot.png",
         alt="DataMapPlot of 20-Newsgroups", width=800, height=800></img>
    """
    topic_per_doc = topic_model.topics_

    df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    df["doc"] = docs
    df["topic"] = topic_per_doc

    unique_topics = set(topic_per_doc)

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = [" ".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = [
            f"Topic-{topic}: " + " ".join([word for word, value in topic_model.get_topic(topic)][:3])
            for topic in unique_topics
        ]

    topic_name_mapping = {topic_num: topic_name for topic_num, topic_name in zip(unique_topics, names)}
    topic_name_mapping[-1] = "Unlabelled"

    # If a set of topics is chosen, set everything else to "Unlabelled"
    if topics is not None:
        selected_topics = set(topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    # Map in topic names and plot
    named_topic_per_doc = pd.Series(topic_per_doc).map(topic_name_mapping).values

    figure, axes = datamapplot.create_plot(
        reduced_embeddings,
        named_topic_per_doc,
        figsize=(width / 100, height / 100),
        dpi=100,
        title=title,
        sub_title=sub_title,
        label_font_size=15,
        **datamap_kwds,
    )
    axes.set_xlim([-4, None])

    return figure



if __name__ == "__main__":
    # retrieve all datasets
    m_ds = dataset_string(get_dataset(['medicine'])['llama_answers'])
    f_ds = dataset_string(get_dataset(['finance'])['llama_answers'])
    o_ds = dataset_string(get_dataset(['open_qa'])['llama_answers'])
    w_ds = dataset_string(get_dataset(['wiki_csai'])['llama_answers'])
    r_ds = dataset_string(get_dataset(['reddit_train', 'reddit_test'])['llama_answers'])
    full_ds = m_ds + f_ds + o_ds + w_ds + r_ds

    file_path = "topic_ex/reduced_embeddings_full.pkl"

    # if BERTopic has already be run retrieve, else run and store
    if os.path.exists(file_path):
        reduced_embeddings = read_list_of_lists_from_file(file_path)
        topic_model = BERTopic.load("topic_ex/bert_topic_model")
    else:
        topic_model, _, embeddings = bertopic_fit(full_ds)
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine").fit(embeddings)
        reduced_embeddings = umap_model.embedding_
        topic_model.save("topic_ex/bert_topic_model")
        write_list_of_lists_to_file(file_path, reduced_embeddings)
        topic_info = topic_model.get_topic_info()
        topic_info.to_csv("topic_ex/topic_info.csv", index=False)

    # create the labels for the scatter plot
    labels = [0]*len(m_ds) + [1]*len(f_ds) + [2]*len(o_ds) + [3]*len(w_ds) + [4]*len(r_ds)
    doc_set_names = ["Medicine", "Finance", "Open QA", "Wiki CS/AI", "Reddit"]

    # create scatter plot
    figure = scatter_plot(
        full_ds,
        reduced_embeddings,
        doc_set_labels=labels,
        title="",
        custom_labels=doc_set_names,
        width= 1200,
        height = 1200,
        point_colors = ["#2ee5e8", "#e82ee5", "#e6d724", "#2ee84d", "#fe7f7f"]
    )

    # create plot with top 30 topics
    figure2 = visualize_document_datamap(
        topic_model,
        docs = full_ds,
        topics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        reduced_embeddings = reduced_embeddings,
        title = "",
        width = 1200,
        height = 1200,
    )

    # store the figures
    figure.savefig("topic_ex/scatter_plot.png", dpi=300, bbox_inches='tight')
    figure2.savefig("topic_ex/datamapplot.png", dpi=300, bbox_inches='tight')