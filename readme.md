<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Black Friday Regression
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-blackfridaypurchaseregression.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1RWroxgwIpuQxqV2oSiQM1DLl99Th6SUA)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Pre-processing__](#c-pre-processing)
    - [__(D) Exploratory Data Analysis__](#d-exploratory-data-analysis)
    - [__(E) Modelling__](#e-modelling)
    - [__(F) Saving the project__](#f-saving-the-project)
    - [__(G) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __regression__ project that uses the  [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday) to __predict the purchase amount__.
- The __goal__ is build a model that accurately __predicts the purchase amount__  based on the features. 
- The performance of the model is evaluated using several __metrics__, including _MaxError_, _MeanAbsoluteError_, _MeanAbsolutePercentageError_, _MSE_, _RMSE_, _MAE_, _R2_, _ExplainedVariance_ and other imbalanced regression metrics.

#### __Overview__
- This project involves building a machine learning model to predict the purchase amounts based on number of 12 features. 8 features are categorical and 4 features are numerical. The dataset contains 550068 records. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, Scikit-learn.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-blackfridaypurchaseregression.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1fhc2m734FTpzvQTLg3fd01z9JUVc9U20#scrollTo=MoonaHldOh1T"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/BlackFridayPurchaseRegression/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1RWroxgwIpuQxqV2oSiQM1DLl99Th6SUA"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __predict The purchase amount__  based on features.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-blackfridaypurchaseregression.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-blackfridaypurchaseregression.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday) from kaggle platform.
- The dataset contains 12 features, 8 features are categorical and 4 features are numerical.
- The dataset contains the following features:


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>



| Column ID |         Column Name        | Data type |           Description           | Masked |
|:---------:|:--------------------------:|:---------:|:-------------------------------:|--------|
|     0     |           User_ID          |   int64   |      Unique Id of customer      | False  |
|     1     |         Product_ID         |   object  |       Unique Id of product      | False  |
|     2     |           Gender           |   object  |         Sex of customer         | False  |
|     3     |             Age            |   object  |         Age of customer         | False  |
|     4     |         Occupation         |   int64   |   Occupation code of customer   | True   |
|     5     |        City_Category       |   object  |         City of customer        | True   |
|     6     | Stay_In_Current_City_Years |   object  | Number of years of stay in city | False  |
|     7     |       Marital_Status       |   int64   |    Marital status of customer   | False  |
|     8     |     Product_Category_1     |   int64   |       Category of product       | True   |
|     9     |     Product_Category_2     |  float64  |       Category of product       | True   |
|     10    |     Product_Category_3     |  float64  |       Category of product       | True   |
|     11    |          Purchase          |   int64   |         Purchase amount         | False  |


</td></tr> </table>


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

|   # | Column                      | Non-Null Count | Dtype    |
|----:|:----------------------------|---------------:|:--------|
|   0 | User_ID                     |    550068      | int64   |
|   1 | Product_ID                  |    550068      | object  |
|   2 | Gender                      |    550068      | object  |
|   3 | Age                         |    550068      | object  |
|   4 | Occupation                  |    550068      | int64   |
|   5 | City_Category               |    550068      | object  |
|   6 | Stay_In_Current_City_Years  |    550068      | object  |
|   7 | Marital_Status              |    550068      | int64   |
|   8 | Product_Category_1          |    550068      | int64   |
|   9 | Product_Category_2          |    376430      | float64 |
|  10 | Product_Category_3          |    166821      | float64 |
|  11 | Purchase                    |    550068      | int64   |


</td><td>

<div style="flex: 50%; padding-left: 50px;">

| Column              | Count         | Mean            | Std            | Min          | 25%          | 50%          | 75%           | Max           |
|---------------------|---------------|----------------|----------------|--------------|--------------|--------------|---------------|---------------|
| User_ID             | 550068.0      | 1.003029e+06   | 1727.591586    | 1000001.0    | 1001516.0    | 1003077.0    | 1004478.0     | 1006040.0     |
| Occupation          | 550068.0      | 8.076707e+00   | 6.522660       | 0.0          | 2.0          | 7.0          | 14.0          | 20.0          |
| Marital_Status      | 550068.0      | 4.096530e-01   | 0.491770       | 0.0          | 0.0          | 0.0          | 1.0           | 1.0           |
| Product_Category_1  | 550068.0      | 5.404270e+00   | 3.936211       | 1.0          | 1.0          | 5.0          | 8.0           | 20.0          |
| Product_Category_2  | 550068.0      | 9.576434e+00   | 4.226025       | 2.0          | 8.0          | 9.0          | 14.0          | 18.0          |
| Purchase            | 550068.0      | 9.263969e+03   | 5023.065394    | 12.0         | 5823.0       | 8047.0       | 12054.0       | 23961.0       |


</div>

</td></tr> </table>


<div style="text-align: center;">
    <img src="docs/images/target_dist.png" style="max-width: 100%; height: auto;">
</div>


#### Problem, Goal and Solving approach
- This is a __regression__ problem  that uses the a bank dataset [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday)  from kaggle to __predict the purchase amount__ based on 12 features which are 8 categorical and 4 numerical.
- The __goal__ is to build a model that accurately __predict the purchase amount__ based on the features.
- __Solving approach__ is that using the supervised machine learning models (linear, non-linear, ensemly).

#### Study
The project aimed predict the house prices using the features. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset.
- __(C) Pre-processing__: It includes data type casting, feature engineering, missing value handling, outlier handling.
- __(D) Exploratory Data Analysis__: Univariate, Bivariate, Multivariate anaylsises. Correlation and other relations. 
- __(E) Modelling__: Model tuning via GridSearch on Linear, Non-linear, Ensemble Models.  
- __(F) Saving the project__: Saving the project and demo studies.
- __(G) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __lgbm regression__ because of the results and less complexity.
<div style="flex: 50%; padding-left: 80px;">

|            | MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE          | RMSE         | MAE          | R2          | ExplainedVariance |
|----------- |-----------|------------------|-----------------------------|-------------|-------------|-------------|-------------|-------------------|
| lgbm  | 10078.0  | 1701.0           | 36.453917                    | 8.748727e+06 | 2957.824659 | 2223.717391 | 0.654627 | 0.654628 |


</div>


- Model tuning results are below.

<table>
<tr><th>Linear Model</th></tr>
<tc><td>

| models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2          | ExplainedVariance |
|-----------|----------|------------------|------------------------------|-------------|-------------|-------------|-------------|-------------------|
| lin_reg   | 0.129605 | 3598.89804       | 2.204832e+07                 | 17826.0     | 2907.0      | 109.23373  | 0.129601    | 4695.564052       |
| l1_reg    | 17826.0  | 2907.0           | 109.233910                   | 2.204833e+07 | 4695.564874 | 3598.897840 | 0.129600    | 0.129605          |
| l2_reg    | 17826.0  | 2907.0           | 109.233799                   | 2.204832e+07 | 4695.563909 | 3598.897786 | 0.129601    | 0.129606          |
| enet_reg  | 17824.0  | 2906.0           | 109.234604                   | 2.204837e+07 | 4695.569602 | 3598.892341 | 0.129599    | 0.129603          |



</td><td> </table>


<table>
<tr><th>Non-Linear Model</th></tr>
<tc><td>

| model     | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2         | ExplainedVariance |
|-----------|----------|-------------------|------------------------------|-------------|-------------|-------------|------------|-------------------|
| knn_reg   | 20120.0  | 1730.0            | 41.478205                    | 9.998852e+06 | 3162.096173 | 2317.771029 | 0.605276  | 0.605349          |
| dt_params | 14026.0  | 1692.0            | 34.324296                    | 8.968823e+06 | 2994.799331 | 2234.869398 | 0.645939  | 0.645940          |


</td><td> </table>


<table>
<tr><th>Ensemble Model</th></tr>
<tc><td>

| model | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE | RMSE | MAE | R2 | ExplainedVariance |
|-------|----------|------------------|------------------------------|-----|------|-----|----|-------------------|
| rf    | 12503.0  | 1629.0           | 32.883898                    | 8.647943e+06 | 2940.738466 | 2165.866272 | 0.658606 | 0.658606 |
| gbr   | 9591.0   | 1719.0           | 36.007663                    | 8.921979e+06 | 2986.968117 | 2252.078981 | 0.647788 | 0.647788 |
| xgbr  | 9864.0   | 1691.0           | 35.596922                    | 8.741287e+06 | 2956.566832 | 2220.059138 | 0.654921 | 0.654921 |
| lgbm  | 10078.0  | 1701.0           | 36.453917                    | 8.748727e+06 | 2957.824659 | 2223.717391 | 0.654627 | 0.654628 |
| cb    | 10116.0  | 1675.0           | 36.303317                    | 8.570677e+06 | 2927.571887 | 2188.880224 | 0.661656 | 0.661657 |


</td><td> </table>


## Details

### Abstract
- [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday) is used to predict the purchase amount. The dataset has 550068 records, 12 features which are 8 categorical and 4 numerical typed. The problem is supervised learning task as regression. The goal is predicting  a purchase value  correctly through using supervised machine learning algorithms such as non-linear, ensemble and similar model.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through cross validation and Grid Search model tuning approachs. Hyperparameter tuning implemented Greedy Greed Search approach which tunes a hyper param at once a time while iterating the sorted order according the importance of the hyperparams. Models are evaluated with cross validation methods using 5 split. Regression results collected and compared between the models. Selected the basic and more succesful model. Tuned __lgbm regression__ model has __2957.824659 __ RMSE , __2223.717391__ MAE, __0.654627 __ R2, __0.654628__ Explained Variance, the other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── cat_encods.json
│   ├── component_configs.json
│   ├── requirements.txt
│   ├── finalized_model_lgbm.sav
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/component_configs.json :
    - It includes the web components to generate web page.
  - demo_app/finalized_model_lgbm.sav:
    - The trained (Model Tuned) model as pickle (python object saving) format.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    


### Explanation of the Study
#### __(A) Dependencies__:
  -  There is a third-parth installation which is kaggle dataset api, just follow the study codes it will be handled. The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
#### __(B) Dataset__: 
  - Downloading the [__Black Friday Dataset__](https://www.kaggle.com/datasets/sdolezel/black-friday) via kaggle dataset api from kaggle platform. The dataset has 550068 records. There are 12 features which are 8 categorical and 4 numerical typed. For more info such as histograms and etc... you can look the '(D) Exploratory Data Analysis' chapter.
#### __(C) Pre-processing__: 
  - The processes are below:
    - Preparing the dtypes such as casting the object type to categorical type.
    - Missing value processes: Finding the missing values and handled the missing value via dropping or imputation.
    - Outlier analysis processes: uses  both visual and IQR calculation apporachs. According to IQR approach, detected statistically significant outliers are handled using boundary value casting assignment method. (There was no outlier value as statistically significant)

      <div style="text-align: center;">
          <img src="docs/images/outliers.png" style="width: 600px; height: 150px;">
      </div>
 
#### __(D) Exploratory Data Analysis__:
  - Dataset Stats
<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

|   # | Column                      | Non-Null Count | Dtype    |
|----:|:----------------------------|---------------:|:--------|
|   0 | User_ID                     |    550068      | int64   |
|   1 | Product_ID                  |    550068      | object  |
|   2 | Gender                      |    550068      | object  |
|   3 | Age                         |    550068      | object  |
|   4 | Occupation                  |    550068      | int64   |
|   5 | City_Category               |    550068      | object  |
|   6 | Stay_In_Current_City_Years  |    550068      | object  |
|   7 | Marital_Status              |    550068      | int64   |
|   8 | Product_Category_1          |    550068      | int64   |
|   9 | Product_Category_2          |    376430      | float64 |
|  10 | Product_Category_3          |    166821      | float64 |
|  11 | Purchase                    |    550068      | int64   |

</td><td>

<div style="flex: 50%; padding-left: 50px;">

| Column              | Count         | Mean            | Std            | Min          | 25%          | 50%          | 75%           | Max           |
|---------------------|---------------|----------------|----------------|--------------|--------------|--------------|---------------|---------------|
| User_ID             | 550068.0      | 1.003029e+06   | 1727.591586    | 1000001.0    | 1001516.0    | 1003077.0    | 1004478.0     | 1006040.0     |
| Occupation          | 550068.0      | 8.076707e+00   | 6.522660       | 0.0          | 2.0          | 7.0          | 14.0          | 20.0          |
| Marital_Status      | 550068.0      | 4.096530e-01   | 0.491770       | 0.0          | 0.0          | 0.0          | 1.0           | 1.0           |
| Product_Category_1  | 550068.0      | 5.404270e+00   | 3.936211       | 1.0          | 1.0          | 5.0          | 8.0           | 20.0          |
| Product_Category_2  | 550068.0      | 9.576434e+00   | 4.226025       | 2.0          | 8.0          | 9.0          | 14.0          | 18.0          |
| Purchase            | 550068.0      | 9.263969e+03   | 5023.065394    | 12.0         | 5823.0       | 8047.0       | 12054.0       | 23961.0       |

</div>

</td></tr> </table>
  - Variable Analysis
    - Univariate analysis, 
      <div style="text-align: center;">
          <img src="docs/images/feat_dists.png" style="width: 400px; height: 200px;">
          <img src="docs/images/cat_feat_dist.png" style="width: 400px; height: 200px;">
          <img src="docs/images/feat_violin.png" style="width: 400px; height: 200px;">
      </div>
    - Bivariate analysis
      <div style="text-align: center;">
          <img src="docs/images/bi_var_1.png" style="width: 400px; height: 300px;">
          <img src="docs/images/bi_var_2.png" style="width: 400px; height: 300px;">
          <img src="docs/images/bi_var_3.png" style="width: 400px; height: 300px;">
      </div>
    - Multivariate analysis.
      <div style="text-align: center;">
          <img src="docs/images/multi_1.png" style="width: 400px; height: 300px;"> 
      </div>
  - Other relations.
    <div style="display:flex; justify-content: center; align-items:center;">
      <div style="text-align: center;">
      <figure>
      <p>Correlation</p>
      <img src="docs/images/corr_heat_map.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
       <div style="text-align: center;">
      <figure>
      <p>Correlation between target</p>
      <img src="docs/images/feat_corr.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Variance</p>
      <img src="docs/images/feat_var.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Covariance</p>
      <img src="docs/images/feat_covar.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
    </div>

#### __(E) Modelling__: 
  - Data Split
    - Splitting the dataset via  sklearn.model_selection.train_test_split (test_size = 0.2).
  - Util Functions
    - Greedy Step Tune
      - It is a custom tuning approach created by me. It tunes just a hyperparameter per step using through GridSerchCV. It assumes the params ordered by importance so it reduces the computation and time consumption.  
    - Model Tuner
      - It is an abstraction of the whole training process. It aims to reduce the code complexity. It includes the corss validation and GridSerachCV approachs to implement training process.
    - Learning Curve Plotter
      - Plots the learning curve of the already trained models to provide insight.
  - Linear Model Tuning Results _without balanciy process_
    - linear, l1, l2, enet regressions
    - Cross Validation Scores
      | models    | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2          | ExplainedVariance |
      |-----------|----------|------------------|------------------------------|-------------|-------------|-------------|-------------|-------------------|
      | lin_reg   | 0.129605 | 3598.89804       | 2.204832e+07                 | 17826.0     | 2907.0      | 109.23373  | 0.129601    | 4695.564052       |
      | l1_reg    | 17826.0  | 2907.0           | 109.233910                   | 2.204833e+07 | 4695.564874 | 3598.897840 | 0.129600    | 0.129605          |
      | l2_reg    | 17826.0  | 2907.0           | 109.233799                   | 2.204832e+07 | 4695.563909 | 3598.897786 | 0.129601    | 0.129606          |
      | enet_reg  | 17824.0  | 2906.0           | 109.234604                   | 2.204837e+07 | 4695.569602 | 3598.892341 | 0.129599    | 0.129603          |
    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/linear_reg_f_imp.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_reg_f_imp.png" style="width: 450px; height: 200px;">
      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/lin_regression_l_curv.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_reg_l_curv.png" style="width: 450px; height: 200px;">
      </div>
  - Non-Linear Models
    - Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Tree
    - Cross Validation Scores _without balanciy process_
      | model     | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2         | ExplainedVariance |
      |-----------|----------|-------------------|------------------------------|-------------|-------------|-------------|------------|-------------------|
      | knn_reg   | 20120.0  | 1730.0            | 41.478205                    | 9.998852e+06 | 3162.096173 | 2317.771029 | 0.605276  | 0.605349          |
      | dt_params | 14026.0  | 1692.0            | 34.324296                    | 8.968823e+06 | 2994.799331 | 2234.869398 | 0.645939  | 0.645940          |
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/non_lin_feat_imp.png" style="width: 400px; height: 300px;">
      </div>


  - Ensemble Models
    - Random Forest, Gradient Boosting Machines, XGBoost, LightGBoost, CatBoost
    - Cross Validation Scores _without balanciy process_
      | model | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE | RMSE | MAE | R2 | ExplainedVariance |
      |-------|----------|------------------|------------------------------|-----|------|-----|----|-------------------|
      | rf    | 12503.0  | 1629.0           | 32.883898                    | 8.647943e+06 | 2940.738466 | 2165.866272 | 0.658606 | 0.658606 |
      | gbr   | 9591.0   | 1719.0           | 36.007663                    | 8.921979e+06 | 2986.968117 | 2252.078981 | 0.647788 | 0.647788 |
      | xgbr  | 9864.0   | 1691.0           | 35.596922                    | 8.741287e+06 | 2956.566832 | 2220.059138 | 0.654921 | 0.654921 |
      | lgbm  | 10078.0  | 1701.0           | 36.453917                    | 8.748727e+06 | 2957.824659 | 2223.717391 | 0.654627 | 0.654628 |
      | cb    | 10116.0  | 1675.0           | 36.303317                    | 8.570677e+06 | 2927.571887 | 2188.880224 | 0.661656 | 0.661657 |

    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_feat_imp.png" style="width: 800px; height: 200px;">
      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_l_curv.png" style="width: 800px; height: 200px;">
      </div>

#### __(F) Saving the project__: 
  - Saving the project and demo studies.
    - trained model __lgbm_model.sav__ as pickle format.
#### __(G) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is predicting the sales price based on four features.
    - Usage: Set the feature values through sliding the radio buttons and dropdown menu then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-blackfridaypurchaseregression.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

