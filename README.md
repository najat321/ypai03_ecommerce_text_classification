# E-commerce Text Classification

## Project Description
### i.	What this project does?
This project is to categorize unseen products into 4 categories namely "Electronics”, "Household”, "Books”, and “Clothing & Accessories” by using LSTM neural network.
### ii.	Any challenges that was faced and how I solved them?
- There were some challenges in the data cleaning process where there was 1 missing value in the text column. I solved it by removing the missing values
- The label was converted into integers using LabelEncoder before continuing with the model. The X_text and X_train was tokenized and padded. 
### iii.	Some challenges / features I hope to implement?
I hope to implement feature engineering. Experimenting with different feature engineering approaches can be beneficial. These approaches include using bag-of-words, TF-IDF, word embeddings (e.g., Word2Vec, GloVe, or fastText), character-level n-grams, part-of-speech tags, syntactic parsing, or domain-specific features. Feature engineering can capture important semantic and contextual information from the text and enhance the model's ability to learn meaningful representations.
## How to install and run the project 
Here's a step-by-step guide on how to install and run this project:

1. Install Python: Ensure that Python is installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org/) and follow the installation instructions specific to your operating system.

2. Clone the repository: Go to the GitHub repository where your .py file is located. Click on the "Code" button and select "Download ZIP" to download the project as a ZIP file. Extract the contents of the ZIP file to a location on your computer.

3. Set up a virtual environment (optional): It is recommended to set up a virtual environment to keep the project dependencies isolated. Open a terminal or command prompt, navigate to the project directory, and create a virtual environment by running the following command: ``python -m venv myenv``

   Then, activate the virtual environment:

   If you're using Windows: myenv\Scripts\activate

   If you're using macOS/Linux: source myenv/bin/activate

4. Install dependencies: In the terminal or command prompt, navigate to the project directory (where the requirements.txt file is located). Install the project dependencies by running the following command: ``pip install -r requirements.txt``

   This will install all the necessary libraries and packages required by the project.

5. Run the .py file: Once the dependencies are installed, you can run the .py file from the command line. In the terminal or command prompt, navigate to the project directory and run the following command: ``python your_file.py``
   Replace "your_file.py" with the name of your .py file that was opened in this github repository.

   Now, you're done! The project should now run, and you should see the output or any other specified behavior defined in your .py file.

## Output of this project
#### i. Model Accuracy:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_ecommerce_text_classification/main/Accuracy_model.PNG)

#### ii. Model F1-Score:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_ecommerce_text_classification/main/F1%20Score_model.PNG)

#### iii. Model Performance:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_ecommerce_text_classification/main/Model_performance.png)
 
#### iv. Model Architecture:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_ecommerce_text_classification/main/Model%20architecture.PNG)

## Source of datasets : 
https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification


