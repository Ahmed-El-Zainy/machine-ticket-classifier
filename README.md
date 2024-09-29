# Ticketing_system_Task
[![Open in Colab-AraBert-Mini-Meduim](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZRzwdN7C--i8vFADslYDwpkPwQVLUGXb?usp=sharing) [![Open in Colab-CAMeL-Poetry-Arabic](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1POiN7j0JmJDYupA-Vppr_qg_hNhEZ5A3?usp=sharing)[![Open in Colab-CAMeL-Poetry-Arabic](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DO8EyR_zanXakP_T6Llrv8h68oLOpxiB?usp=sharing)



--------------------------------
## Text-Based Machine Learning Classifier for Unlabeled Data

This repository contains a project focused on building a machine learning classifier for text data. The dataset, provided in the form of an email attachment, includes records with no predefined labels. The goal of the project is to identify and assign meaningful labels based on the content of the "Request Detail" ,  "Notes" fields, "Subject", fields, "Subject_Eng" fields and "Request Type" fields which are predominantly in Arabic, though English may also be present.

--------------------------------
### The primary objectives of this project are:

### 1. Data Processing</b></summary>

  Techniques:
  - Drop Unnamed: 0 column and Uneeded columns  - Apply Stemming and Lemmatization
  - Remove Null values - Remove Duplicate values - Apply Arabic normalization - Remove Special Characters
  - Remove Punctuations - Apply Regular Expression Techniques with Arabic Letters
  - Remove Numbers - Apply splitting hashtag to words - Remove Arabic Stop Words - Clean hashtag
  - Remove English Stop Words - (Optional) Remove emoji - Remove Whitespace - Remove URLs
  - Remove HTML Tags - Remove Emails - Remove Phone Numbers - Remove Fax Numbers 
  - (Optional) Remove Tweets - Remove Arabic Numbers - Remove Arabic Diacritics - (Optional) Remove Outliers
  - (Optional) Apply Translation if Columns after cleaning most Values ' ' or empty values 


Before Cleaning:
  df_process["Request Detail"][0]
  ```
  *** This is an external email. Be Vigilant and take precautions.***_x000D_*** Do not click links or open attachments or reply unless you recognize the sender and their email address, and you are expecting the email.***_x000D__x000D__x000D_نفيد سعادتكم بوجود مشكله في جهاز عيادة العظام_x000D_حيث انه لا يمكن الدخول وفتح الجهاز بيوزر موظف وزاره الصحة_x000D__x000D_نوع الجهاز : DEEL OPTIPLEX 3080_x000D_رقم الجهاز : DLH0PJ3_x000D__x000D_برجاء الاطلاع وتوجيه المختصين ليدكم لحل المشكلة_x000D__x000D_ولكم جزيل الشكر ،،،،_x000D__x000D__x000D__x000D_مدير تقنية المعلومات بمستشفى السليل العام_x000D_منصور بن عبدالله الحابي_x000D_0556355578_x000D__x000D__x000D_[MANSOUR ABDULLAH AL-HABI (2)]_x000D__x000D__x000D__x000D__x000D__x000D_تنبيه بإخلاء المسئولية: هذه الرسالة ومرفقاتها معدة لاستخدام المُرسَل إليه المقصود بالرسالة فقط و قد تحتوي على معلومات سرية أو محمية قانونيا. إن لم تكن الشخص المقصود، فإنه يُمنع منعا باتا أي عرض أو نشر أو استخدام غير مصرح به للمحتوى. نرجو إخطار المُرسِل عن طريق الرد على هذا البريد الإلكتروني وإتلاف جميع النسخ الموجودة لديك. تعد التصريحات و الآراء المذكورة في الرسالة خاصة بالمُرسِل و لا تمثل وزارة الصحة. كما لا تتحمل الوزارة مسؤولية الأضرار الناتجة عن أي فيروسات قد تحملها هذه الرسالة._x000D__x000D_CONFIDENTIALITY NOTICE: This e-mail message, including any attachments, is for the sole use of the intended recipient(s) and may contain confidential and privileged information or otherwise protected by law. If you are not the intended recipient, you are notified that any unauthorized review, use, disclosure or distribution is strictly prohibited. please notify the sender by replying to this email and destroy all copies of the original message. Statements and opinions expressed in this Email are those of the sender, and do not necessarily reflect those of Ministry of Health (MOH). Ministry of Health (MOH) accepts no liability for damage caused by any virus transmitted by this Email._x000D__x000D__x000D_MOH Site. <http://www.moh.gov.sa>_x000D_[attachment name=image001.jpg]
  ```


After Cleaning:
  df_process["Request Detail"][0]
  ```
   نفيد سعادتكم بوجود مشكله في جهاز عياده العظام حيث انه لا يمكن الدخول وفتح الجهاز بيوزر موظف وزاره الصحه نوع الجهاز رقم الجهاز برجاء الاطلاع وتوجيه المختصين ليدكم لحل المشكله ولكم جزيل الشكر مدير تقنيه المعلومات بمستشفي السليل العام منصور بن عبدالله الحابي تنبيه باخلاء المسءوليه هذه الرساله ومرفقاتها معده لاستخدام المرسل اليه المقصود بالرساله فقط و قد تحتوي علي معلومات سريه او محميه قانونيا ان لم تكن الشخص المقصود فانه يمنع منعا باتا اي عرض او نشر او استخدام غير مصرح به للمحتوي نرجو اخطار المرسل عن طريق الرد علي هذا البريد الالكتروني واتلاف جميع النسخ الموجوده لديك تعد التصريحات و الاراء المذكوره في الرساله خاصه بالمرسل و لا تمثل وزاره الصحه كما لا تتحمل الوزاره مسءوليه الاضرار الناتجه عن اي فيروسات قد تحملها هذه الرساله 
  ```


## 2. Exploratory Data Analysis (EDA)
<img src="asset/Screenshot 2024-09-29 at 6.52.02 PM.png">

* Word Cloud Mapping for the combine text of Reuest Details and Notes

<img src="asset/Value_C_Subject.png">

* The charts show that the majority of issues arise from computers and printers. It’s essential to classify these as distinct categories. Prioritizing these will improve issue resolution. 
The bar chart displays the frequency of different subjects in the dataset. 
The x-axis represents the unique subjects and the y-axis indicates the number of occurrences of each subject.
The subject with the highest frequency is '{fig.data[0].x[0]}' with a count of {fig.data[0].y[0]}. 
Other subjects with notable counts include '{fig.data[0].x[1]}', '{fig.data[0].x[2]}', and so on.

<img src="asset/Screenshot 2024-09-29 at 7.45.33 PM.png"> 

* Distribution of the word count 


<img src="asset/Screenshot 2024-09-29 at 7.50.00 PM.png">

* Apply PCA to reduce the dimensionality of the data and show the cardinalty of the data

###  3. Dimensionality Reduction

### 4. Feature Engineering

5. Data Labeling</b></summary>
  Implement a strategy to generate labels from the text data. The labels should reflect key discrepancies or noteworthy information derived from the "Request Detail" and "Notes" fields. Utilize techniques such as fuzzy matching, keyword extraction, Zero-shot classification, Few-shot classification and manual annotation where necessary.
</details>

<details>
  <summary><b>6. Text Classification</b></summary>
  Develop a machine learning model to classify the labeled data, ensuring high accuracy and relevance. Experiment with different classification models such as logistic regression, support vector machines, and deep learning models to find the best fit.
</details>


<details>
  <summary><b>7. Model Evaluation</b></summary>
  Assess the performance of the model using standard evaluation metrics such as accuracy, F1-score, precision, and recall. Compare the predictions against a validation set of manually labeled data to measure the effectiveness of the model.
</details>

### Key Features

<details>
  <summary><b>1. Automated Labeling</b></summary>
  Utilizes custom algorithms and techniques such as fuzzy matching, keyword extraction, and semantic analysis to generate labels for the text data. This helps in the efficient categorization of unstructured data and reduces manual effort.
</details>

<details>
  <summary><b>2. NLP Techniques</b></summary>
  Implements preprocessing steps like tokenization, stop-word removal, and word embeddings to handle the complexity of Arabic and English text. These techniques enable the extraction of meaningful features from the text, facilitating better model performance.
</details>

<details>
  <summary><b>3. Model Evaluation</b></summary>
  Employs metrics such as accuracy, F1-score, precision, and recall to validate the model's effectiveness in text classification. Detailed performance reports are generated to guide further improvements and refinements in the model.
</details>


### Project Structure
```
- `data/`: Contains the dataset (pre-labeled and raw).
- `notebooks/`: Jupyter notebooks detailing the exploratory data analysis (EDA) and model development process.
- `src/`: Source code for data preprocessing, model training, and evaluation scripts.
- `results/`: Contains evaluation reports and visualizations of the model's performance.
- `README.md`: Detailed instructions on how to run the project and interpret the results.
```
s
### Getting Started

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/repo-name.git
    cd repo-name
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Contributing

Contributions are welcome! Feel free to submit issues or pull requests if you have ideas for improvement or new features.

### License

This project is licensed under the MIT License.

---

This description should provide a clear and professional overview of the project, its goals, structure, and usage.