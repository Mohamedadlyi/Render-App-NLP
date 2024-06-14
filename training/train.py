import sys
from ModelTraining import DeepLearningModel, TextClassificationModel
from DataFetching import DataLoader
from sklearn.model_selection import train_test_split

def main(argc, argv):
    # Default values
    default_main_dir = '/kaggle/input/arabic-dialect-db'
    default_model_types = ['logistic', 'naive_bayes', 'deep_learning']
    default_save_paths = ['logistic_model.pkl', 'nb_model.pkl', 'dl_model.h5', 'tokenizer.pkl', 'label_encoder.pkl']

    # Assign values based on input arguments or use default values
    main_dir = argv[1] if argc > 1 else default_main_dir
    model_types = argv[2].split(',') if argc > 2 else default_model_types
    save_paths = argv[3:] if argc > 3 else default_save_paths

    dbfile = f'{main_dir}/dialects_database.db'
    data_loader = DataLoader(dbfile)
    data_df, type_df = data_loader.load_data()

    X_train, X_test, y_train, y_test = train_test_split(data_df['text'], type_df['dialect'], test_size=0.2, random_state=42)

    # Logistic Regression Model
    if 'logistic' in model_types:
        logistic_model = TextClassificationModel(model_type='logistic')
        logistic_model.train(X_train, y_train)
        print("Logistic Regression Accuracy:", logistic_model.evaluate(X_test, y_test))
        logistic_model.save(save_paths[0] if len(save_paths) > 0 else 'logistic_model.pkl')

    # Naive Bayes Model
    if 'naive_bayes' in model_types:
        nb_model = TextClassificationModel(model_type='naive_bayes')
        nb_model.train(X_train, y_train)
        print("Naive Bayes Accuracy:", nb_model.evaluate(X_test, y_test))
        nb_model.save(save_paths[1] if len(save_paths) > 1 else 'nb_model.pkl')

    # Deep Learning Model
    if 'deep_learning' in model_types:
        dl_model = DeepLearningModel(max_len=100)
        dl_model.train(X_train, y_train, epochs=10, patience=3)
        print("Deep Learning Model Accuracy:", dl_model.evaluate(X_test, y_test))
        dl_model.save(save_paths[2] if len(save_paths) > 2 else 'dl_model.h5',
                      save_paths[3] if len(save_paths) > 3 else 'tokenizer.pkl',
                      save_paths[4] if len(save_paths) > 4 else 'label_encoder.pkl')

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
