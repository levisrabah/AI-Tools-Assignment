# debug_tensorflow_script.py
"""
Debugging Common TensorFlow Errors
This script demonstrates common bugs and their fixes.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

print("=== TENSORFLOW DEBUGGING CHALLENGE ===")

# BUGGY CODE (with common errors):
def buggy_model_creation():
    """Demonstrates common TensorFlow bugs and their fixes."""
    
    print("\n1. FIXING DIMENSION MISMATCH ERRORS")
    
    # BUG: Incorrect input shape
    # ORIGINAL BUGGY CODE:
    # model = keras.Sequential([
    #     layers.Dense(128, activation='relu', input_shape=(784)),  # Missing comma!
    #     layers.Dense(10, activation='softmax')
    # ])
    
    # FIXED CODE:
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),  # Added comma for tuple
        layers.Dense(10, activation='softmax')
    ])
    print("✓ Fixed input_shape tuple syntax")
    
    print("\n2. FIXING LOSS FUNCTION ERRORS")
    
    # BUG: Wrong loss function for multi-class classification
    # ORIGINAL BUGGY CODE:
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',  # Wrong for multi-class!
    #     metrics=['accuracy']
    # )
    
    # FIXED CODE:
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Correct for integer labels
        metrics=['accuracy']
    )
    print("✓ Fixed loss function for multi-class classification")
    
    print("\n3. FIXING DATA PREPROCESSING ERRORS")
    
    # Generate sample data
    X_train = np.random.random((1000, 784))
    y_train = np.random.randint(0, 10, (1000,))
    
    # BUG: Not normalizing data
    # ORIGINAL BUGGY CODE:
    # X_train_processed = X_train  # No normalization!
    
    # FIXED CODE:
    X_train_processed = X_train / 255.0  # Normalize to [0, 1]
    print("✓ Fixed data normalization")
    
    print("\n4. FIXING TRAINING LOOP ERRORS")
    
    # BUG: Inconsistent data shapes
    try:
        # This would cause an error if shapes don't match
        history = model.fit(
            X_train_processed, y_train,
            epochs=2,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        print("✓ Model training successful")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None
    
    print("\n5. FIXING PREDICTION ERRORS")
    
    # Generate test data
    X_test = np.random.random((100, 784)) / 255.0
    
    # BUG: Wrong prediction interpretation
    # ORIGINAL BUGGY CODE:
    # predictions = model.predict(X_test)
    # predicted_classes = predictions  # Wrong! These are probabilities
    
    # FIXED CODE:
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to classes
    print("✓ Fixed prediction interpretation")
    
    return model

# Common debugging strategies
def debugging_strategies():
    """Demonstrate debugging strategies for TensorFlow."""
    
    print("\n=== DEBUGGING STRATEGIES ===")
    
    strategies = [
        "1. Check tensor shapes at each step using .shape",
        "2. Use tf.print() for debugging tensor values",
        "3. Validate data types (float32 vs int32)",
        "4. Test with small datasets first",
        "5. Use tf.debugging.assert_* functions",
        "6. Enable eager execution for easier debugging",
        "7. Use TensorBoard for visualization",
        "8. Check for NaN/Inf values in loss"
    ]
    
    for strategy in strategies:
        print(f"   {strategy}")

# Advanced debugging example
def advanced_debugging_example():
    """Show advanced debugging techniques."""
    
    print("\n=== ADVANCED DEBUGGING EXAMPLE ===")
    
    # Create a model with potential issues
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Add debugging callbacks
    class DebugCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if batch % 100 == 0:
                print(f"Batch {batch}: loss = {logs['loss']:.4f}")
    
    # Compile with debugging
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate sample data
    X_sample = np.random.random((1000, 10))
    y_sample = np.random.randint(0, 2, (1000,))
    
    print("Training with debug callback...")
    history = model.fit(
        X_sample, y_sample,
        epochs=1,
        batch_size=100,
        callbacks=[DebugCallback()],
        verbose=0
    )
    
    print("✓ Advanced debugging implemented successfully")

# Run debugging examples
buggy_model_creation()
debugging_strategies()
advanced_debugging_example()

print("\n=== DEBUGGING CHALLENGE COMPLETED ===")