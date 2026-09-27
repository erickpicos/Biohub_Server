import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrize import remove_parametrizations

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from braindecode.models import EEGNet

from config import *
from dataset_loader import load_dataset

import matplotlib.pyplot as plt

#Train
def train_and_export_onnx(user_id: int):

    X, y = load_dataset(user_id, DATASET_TYPE)
    print(f"Dataset Loaded -> X:{X.shape} Y:{y.shape}")

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= TEST_SIZE, random_state=42, stratify=y)

    print( f"\nDataset Split" )
    print( f"Total Samples: {len(X)} Train Samples: {len(X_train)} Test Samples: {len(X_test)}" )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    model = EEGNet( n_chans=N_CHANNELS, n_outputs=N_CLASSES, n_times=N_SAMPLES, final_conv_length="auto" ).to(device)
    print(model)

    user_folder = ONNX_FOLDER / f"User_{user_id}"
    user_folder.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE )

    best_acc = -1
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        train_correct = 0
        train_total = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            inputs = X_train[i:i+BATCH_SIZE]
            labels = y_train[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
       
        train_loss = running_loss / (len(X_train) / BATCH_SIZE)
        train_acc = train_correct / train_total
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            val_loss = criterion(outputs, y_test).item()
            predicted = torch.argmax(outputs, dim=1)
            predicted_np = predicted.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            acc = accuracy_score(y_test_np, predicted_np)
            precision = precision_score(y_test_np, predicted_np, average="macro", zero_division=0)
            recall = recall_score(y_test_np, predicted_np, average="macro", zero_division=0)
            f1 = f1_score(y_test_np, predicted_np, average="macro", zero_division=0)
            cm = confusion_matrix(y_test_np, predicted_np)

            correct = np.sum(predicted_np == y_test_np)
            total = len(y_test_np)
            wrong = total - correct

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(acc)
        
        print(
            f"\n========== Epoch {epoch+1}/{EPOCHS} =========="
            f"\nLoss: {running_loss:.4f}"
            f"\nAccuracy: {acc:.4f}"
            f"\nPrecision: {precision:.4f}"
            f"\nRecall: {recall:.4f}"
            f"\nF1: {f1:.4f}"
            f"\nCorrect: {correct}/{total}"
            f"\nWrong: {wrong}"
        )

        print(f"\nConfusion Matrix:\n{cm}")

        if acc >= best_acc:
            best_acc = acc
            torch.save( model.state_dict(), user_folder / "model.pth" )

    print(f"\nBest Accuracy: {best_acc:.4f}")
    # GRAFICAS
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(12,5))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, train_accs, marker='o', label='Train')
    plt.plot(epochs, val_accs, marker='s', label='Validation')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, train_losses, marker='o', label='Train')
    plt.plot(epochs, val_losses, marker='s', label='Validation')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(user_folder / "training_curve.png", dpi=300)

    plt.show()
   
    #Export ONNX
    onnx_path = user_folder / "model.onnx"

    model.load_state_dict(torch.load(user_folder / "model.pth", map_location=device))
    model.eval()

    dummy_input = torch.randn(1, N_CHANNELS, N_SAMPLES).to(device)
    print(f"Dummy Input Shape: {dummy_input.shape}")

    remove_parametrizations( model.conv_spatial, "weight", leave_parametrized=True )

    torch.onnx.export( model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], 
        opset_version=15, dynamo=False )

   
    print(f"\nONNX Exported -> {onnx_path}")

    return str(onnx_path)