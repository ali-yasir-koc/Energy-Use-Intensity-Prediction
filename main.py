import helpers as h

train_path = "datasets/train.csv"
test_path = "datasets/test.csv"
sub_path = "datasets/sample_solution.csv"

def main():
    train, test, data, submission = h.read_data(train_path, test_path, sub_path)
    data = h.outlier_update(data)
    data = h.missing_update(data)
    new_data = h.generate_features(data)
    X, y = h.split_data(new_data)
    model = h.create_model(X, y)
    h.prediction(model, new_data, submission)


main()


