import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import tensorflow as tf

def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess it for training
    """
    print("Loading and preprocessing MNIST data...")
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Convert labels to categorical format
    y_train_categorical = to_categorical(y_train, num_classes=10)
    y_test_categorical = to_categorical(y_test, num_classes=10)
    
    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape data (flatten images)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return x_train, y_train_categorical, x_test, y_test_categorical, y_train, y_test

def build_model():
    """
    Create and compile the neural network model
    """
    print("Building model...")
    model = Sequential()
    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model and evaluate on test data
    """
    print("Training model...")
    batch_size = 512
    epochs = 10
    
    # Add callback for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        x=x_train, 
        y=y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history, test_acc

def save_model(model, test_accuracy, output_dir='./models'):
    """
    Save the trained model and its metrics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model in Keras format
    model_path = os.path.join(output_dir, "mnist_model.h5")
    model.save(model_path)
    
    # # Save model in TensorFlow SavedModel format for potential serving
    # tf_model_path = os.path.join(output_dir, "mnist_model_tf")
    # tf.saved_model.save(model, tf_model_path)
    
    # Save accuracy metrics
    metrics = {
        'test_accuracy': test_accuracy
    }
    metrics_path = os.path.join(output_dir, "mnist_model_metrics.joblib")
    joblib.dump(metrics, metrics_path)
    
    print(f"Model saved to {model_path}")
    # print(f"TensorFlow model saved to {tf_model_path}")
    print(f"Model metrics saved to {metrics_path}")

def visualize_results(model, x_test, y_test, y_test_original, output_dir='./models'):
    """
    Generate and save visualizations for model evaluation
    """
    print("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get predictions
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_original, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # Sample predictions visualization
    n_samples = 5
    plt.figure(figsize=(15, 3))
    for i in range(n_samples):
        idx = np.random.randint(0, len(x_test))
        plt.subplot(1, n_samples, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        pred_label = np.argmax(y_pred_proba[idx])
        true_label = y_test_original[idx]
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    
    samples_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(samples_path)
    plt.close()
    
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Sample predictions saved to {samples_path}")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test, y_train_original, y_test_original = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    trained_model, history, test_accuracy = train_model(model, x_train, y_train, x_test, y_test)
    
    # Save model
    save_model(trained_model, test_accuracy)
    
    # Visualize results
    visualize_results(trained_model, x_test, y_test, y_test_original)
    
    print("\nModel training, evaluation, and saving complete!")

if __name__ == "__main__":
    main()