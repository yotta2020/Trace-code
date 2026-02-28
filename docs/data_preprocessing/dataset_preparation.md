# Dataset Preparation

## Dataset for Victim Model

### Overview
For Victim Model, we use Clone Detection Dataset, Defect Detection Dataset and Code Search Dataset.

Detailed dataset information is listed below:

| Task | Dataset Name | Available at|
|------|--------------|-------------|
|Clone Detection (cd) | BigCloneBench | |
|Defect Detection (dd)| Devign||
|Code Search (cs) | CodeSearchNet||

By injecting a set of triggers randomly into the given dataset, we can easily poison a dataset.

### Dataset Preparation
#### Download Dataset
- Please refer to table for downloading the public dataset.
- Once the download process is down, simply copy the dataset to the position listed below.
  - For cd task, place the dataset folder at ``data/raw/cd``
  - For dd task, place the dataset folder at ``data/raw/dd``
  - For cs task, place the dataset folder at ``data/raw/cs``

#### Dataset preprocessing
Since the dataset is in different format and structure, we need to unify them into ``jsonl`` format.
In this case, please head for ``scripts/data_preprocessing`` to preprocess the dataset.

Detailedly, 


## Dataset for Defense Model