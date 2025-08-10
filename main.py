from sklearn.model_selection import train_test_split
from data.load_data import get_iris_data
from models.classifier import train_model
from utils.metrics import evaluate_model

def main():
    X, y = get_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()