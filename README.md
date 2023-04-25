# Blastocyst-Dataset

Repository containing scripts supporting the manuscript "An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization",
submitted to Nature Scientific Data. The dataset associated with this manuscript, uploaded to figshare, is referenced here: https://doi.org/10.6084/m9.figshare.20123153.v3

## Environment
Windows: v10

Python Version: 3.7

Pip-packages: numpy, pandas, scikit-learn, glob, os

## Content: 
* Folder [annotations](annotations): contains all annotations from the international experts on splits of the test set (150 out of 300 images were annotated by each expert such that each image was seen by at least 5 experts), and the consensus vote (gold-standard annotations) of the test set ([test_rev.csv](/annotations/test_rev.csv))
* Folder [model_predictions](model_predictions): contains predictions of multiple CNN- and vision-transformer architectures on the gold-standard test set, trained using the silver-standard training- and validation set. 
* [complete.csv](complete.csv): Silver-standard annotations of the entire image dataset, for each of the three Gardner criteria (expansion, inner cell mass quality and trophectoderm quality), created by the Gardner expert.
* [create_testset.py](create_testset.py): Script used to create a stratified split of the dataset into training-, validation-, and test set.
* [combine_testset_annotations.py](combine_testset_annotations.py): Script applied to calculate the majority vote for each blastocyst image and for each of the three Gardner criteria, based on experts' votes (upon removing annotations from experts whose accuracy was below 0.5 when compared to the Gardner-Expert annotations).
* [convert_model_predictions.py](convert_model_predictions.py): The trained models generate a probability for each class and a confidence interval upon inference, for each of the three Gardner criteria. The script assigns the class with the maximal value and combines the prediction results of each model (one model for each of the three Gardner criteria), for each of the images.
* [testset_statistics_annotators.py](testset_statistics_annotators.py): Script applied to create the mean and standard deviation of experts' accuracy values in comparison to the consensus votes (gold-standard annotations) of the test set, for each of the three Gardner criteria. These values serve as baseline to compare the performance of trained models to experts' performance.
* [calculate_model_metrics.py](calculate_model_metrics.py): Script applied to calculate accuracy, average precision, average recall and weighted F1-score of a model's result compared to the consensus votes of the test set.
* [calculate_inter_annotator_agreement.py](calculate_inter_annotator_agreement.py): Script applied to calculate inter-annotator agreement as well as the agreement of annotators to the consensus vote, based on Cohen's Kappa score.
