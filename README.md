# Blastocyst-Dataset

Repository containing scripts supporting the manuscript "An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization",
submitted to Nature Scientific Data. The dataset associated with this manuscript, uploaded to figshare, is referenced here: https://doi.org/10.6084/m9.figshare.20123153.v3

Content: 
* complete.csv: Silver-standard annotations of the entire image dataset, for each of of the three Gardner criteria (expansion, inner cell mass quality and trophectoderm quality), created by the Gardner expert.
* create_testset.py: Script used to create a stratified split of the dataset into training-, validation-, and test set.
* combine_testset_annotations.py: Script applied to calculate the majority vote for each blastocyst image and for each of the three Gardner criteria, based on experts' votes (upon removing annotations from experts whose accuracy was below 0.5 when compared to the Gardner-Expert annotations).
* convert_model_predictions.csv: The trained models generate a probability for each class and a confidence interval upon inference, for each of the three Gardner criteria. The script assigns the class with the maximal value and combines the prediction results of each model (one model for each of the three Gardner criteria), for each of the images.
* testset_statistics_annotators.py: Script applied to create the mean and standard deviation of expert's accuracy values in comparison to the consensus votes (gold-standard annotations) of the test set, for each of the three Gardner criteria.
* evaluate_prediction_results.py: Script applied to calculate accuracy, average precision, average recall and weighted F1-score of a model's result compared to the consensus votes of the test set.
