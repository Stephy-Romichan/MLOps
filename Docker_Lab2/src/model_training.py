import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    # Load Breast Cancer dataset
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target   # 30 features

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(30,), activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_test, y_test)
    )

    model.save('breast_cancer_model.keras')
    print("Model was trained and saved")
