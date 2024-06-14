# Render-App-NLP
## Description

This project Deploys a dialect classification algorithm Dashboard using Plotly as a dash viualization. 

## Demo Video
## Usage


# Model Training Script


To run the script with default values, simply execute:

```sh
python train.py
```

This will use the following defaults:
- Data directory: `/kaggle/input/arabic-dialect-db`
- Models: `logistic`, `naive_bayes`, `deep_learning`
- Save paths: `logistic_model.pkl`, `nb_model.pkl`, `dl_model.h5`, `tokenizer.pkl`, `label_encoder.pkl`

### Custom Usage

To provide custom arguments, use the following format:

```sh
python script.py <data_directory> <model_types> <save_path1> <save_path2> <save_path3> <tokenizer_path> <label_encoder_path>
```

#### Arguments:
- `<data_directory>`: Path to the directory containing the dataset (default: `/kaggle/input/arabic-dialect-db`)
- `<model_types>`: Comma-separated list of models to train (options: `logistic`, `naive_bayes`, `deep_learning`)
- `<save_path1>`: Path to save the logistic regression model (default: `logistic_model.pkl`)
- `<save_path2>`: Path to save the Naive Bayes model (default: `nb_model.pkl`)
- `<save_path3>`: Path to save the deep learning model (default: `dl_model.h5`)
- `<tokenizer_path>`: Path to save the tokenizer (default: `tokenizer.pkl`)
- `<label_encoder_path>`: Path to save the label encoder (default: `label_encoder.pkl`)

#### Example:

```sh
python train.py /training/ logistic,naive_bayes,deep_learning logistic_model.pkl nb_model.pkl dl_model.h5 tokenizer.pkl label_encoder.pkl
```

This command will:
1. Use `/training/data` as the data directory.
2. Train logistic regression, Naive Bayes, and deep learning models.
3. Save the logistic regression model to `logistic_model.pkl`.
4. Save the Naive Bayes model to `nb_model.pkl`.
5. Save the deep learning model to `dl_model.h5`, the tokenizer to `tokenizer.pkl`, and the label encoder to `label_encoder.pkl`.

### train Script Details

The script will:
1. Load the dataset from the specified data directory.
2. Split the data into training and testing sets.
3. Train and evaluate the specified models.
4. Save the trained models and necessary components to the specified paths.

### Example Output

Upon successful execution, you will see output similar to the following:

```
Logistic Regression Accuracy: 0.85
Naive Bayes Accuracy: 0.82
Deep Learning Model Accuracy: 0.88
```

This indicates the accuracy of each model on the test set.

## Dependencies

Ensure you have the required Python packages installed:

```sh
pip install -r requirements.txt
```


## Usage of the trained model

### Step 1: Install Packages & Run the Application

 1. Ensure the virtual environment is set up and dependencies are installed:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

 2. Run the `src/app.py` script:
    
    ```bash
    python src/app.py
    ```
### Step 2: Open the Browser
Open your web browser and navigate to the `http://localhost:8050` to interact with the application.
