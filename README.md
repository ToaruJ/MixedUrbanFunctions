# Recognizing mixed urban functions from human activities using representation learning methods

The code for the article [**Hu et al. (2023)**](https://dx.doi.org/10.1080/17538947.2023.2170482) entitled "Recognizing mixed urban functions from human activities using representation learning methods" published on *International Journal of Digital Earth*.

The preprocessed data for this article can be obtained from [**Figshare**](https://doi.org/10.6084/m9.figshare.21088669.v2).

## Research framework

The procedure to recognize the mixed urban functions consists of four parts:

1. We utilize representation learning methods to extract three human activity features of a location:

    | Features | Data | Method and model |
    | ---- | ---- | ----|
    | Activity dynamic | Time series of taxi departures/arrivals | Temporal Convolutional network |
    | Mobility interaction | Taxi OD (Origin-Destination) graph | node2vec |
    | Activity semantic | geo-tagged social media tweets | doc2vec |

    The three embedding vectors of the features are concatenated into an integrated vector.

2. Fuzzy C-Means (FCM) clustering algorithm is performed to calculate fuzzy partitions of clusters. We considered the membership value of a cluster as the proportion of an urban function.

3. The mixture index of urban functions was calculated from membership values.

4. We analyze mixed urban functions with auxiliary data.

![framework](https://www.tandfonline.com/na101/home/literatum/publisher/tandf/journals/content/tjde20/2023/tjde20.v016.i01/17538947.2023.2170482/20230201/images/large/tjde_a_2170482_f0002_oc.jpeg)

## Requirements

Experiments were done with the following packages for Python 3.7:

```
gensim==4.1.2
geopandas==0.10.1
networkx==2.6.3
node2vec==0.4.3
numpy==1.19.5
pandas==1.3.3
pkuseg==0.0.25
scikit-fuzzy==0.4.2
scikit-learn==1.0
scipy==1.7.1
tensorflow==2.6.0
tensorflow-addons==0.14.0
tslearn==0.5.2
```

## File descriptions

* `cluster.py`: Calculate the proportion of mixed urban functions (five functions in the article) by soft clustering.

* `dataclean.py`: Clean and preprocess the raw data, including generating the time series and OD graph from taxi trip data, and filtering Weibo tweets. The data after preprocessing can be obtained from [**Figshare**](https://doi.org/10.6084/m9.figshare.21088669.v2).

* `globalval.py`: Some global variables of this repository.

* `odmodel.py`: Generate the embedding vectors of mobility interaction features by node2vec.

* `postanalyze.py`: Analyze the results, including the correlation between the function proportions and land use, the linear regression between the mixture index and taxi trip distance.

* `pyscratch.py`: Scratch file for data visualization and charting.

* `timeseriesmodel.py`: Generate the embedding vectors of activity dynamic features by neural network.

* `weibomodel.py`: Generate the embedding vectors of activity semantic features by doc2vec.

## Citation

* Plain Text

    Hu, J., Gao, Y., Wang, X., & Liu, Y. (2023). Recognizing mixed urban functions from human activities using representation learning methods. *International Journal of Digital Earth*, *16*(1), 289-307. doi: 10.1080/17538947.2023.2170482

* BibTex

    ```
    @article{doi:10.1080/17538947.2023.2170482,
    author = {Junjie Hu and Yong Gao and Xuechen Wang and Yu Liu},
    title = {Recognizing mixed urban functions from human activities using representation learning methods},
    journal = {International Journal of Digital Earth},
    volume = {16},
    number = {1},
    pages = {289-307},
    year  = {2023},
    publisher = {Taylor & Francis},
    doi = {10.1080/17538947.2023.2170482}
    }
    ```