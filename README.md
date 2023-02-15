# Debunkathon
## Brief Intro
As we move forward in our modern, digitalized age with the benefits of cutting-edge technology, we must also be aware of the adverse impacts of some things. As with a coin, everything in this world has two sides. The dissemination of false information is one of today's biggest issues. It is clear that we cannot resolve this issue entirely accurately, but we will make an effort to do so. For this reason, ML Strawhats created a model for predicting fake news that will accept both the text and an image from an article as input and check the information.

## Approach
As a result, we have created a path for building the model. It proceeds as follows
i) Text preprocessing / Data cleaning
ii)Text vectorization and Exploratory data analysis
iii)Building a bag of models and selecting the best performing one.
iv)Hyperparameter tuning using automated libraries like hyperopt,optuna
v)Reverse Image search for image verification.
vi) Model deployment

### Text preprocessing / Data cleaning -> preprocess.py
Raw data is tainted, as we are all aware. Not all data components are helpful. Therefore, we must use several methods of filtering to remove the contaminants.
Preprocess.py, a file we created specifically for this purpose, will perform the necessary preprocessing, such as eliminating stopwords and punctuation.

### Text vectorization -> vectorizer.py
 It vectorizes the data which we get after preprocessing using tokenization.

### Building a bag of models and selecting the best performing one -> model.py
It contains many models in which dataset's data will be processed. It contains models like PassiveAgressiveclassifier, SGD classifier, Random Forest classifier etc.

### Reverse Image search for image verification -> image.py , imagedet.py
 In this stage using google reverse image search api we collect different images in image.py . Then using imagedet.py we show the result to the user if the image involved in the news is fake or not.
### Model evaluation and use - eval.py
Using eval.py our user will give input to the model and our model will do the prediction.

So that's it! We finally made a full-fledged model!!
                                                          THANK YOU!
