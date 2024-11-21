import gc
import torch
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=2, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = 0.0

    def __call__(self, val_loss):
        if self.best_loss == 0.0:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping")
                return True
        return False
    
def train(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict):
    early_stopping = EarlyStopping()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for videos, audio, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            videos, audio, labels = videos.to(device), audio.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        if val_loss > early_stopping.best_loss: 
            torch.save(model.state_dict(), f'{save_dict}/{model_name}.pth')

        if early_stopping(val_loss):
            break
        
            
    print("Training complete.")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for videos, audio, labels in tqdm(val_loader, desc='Validating'):
            videos, audio, labels = videos.to(device), audio.to(device), labels.to(device)

            outputs = model(videos, audio)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / (len(val_loader) * val_loader.batch_size)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return avg_val_loss, val_accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for videos, audio, labels in tqdm(test_loader, desc='Testing'):
            videos, audio, labels = videos.to(device), audio.to(device), labels.to(device)

            outputs = model(videos, audio)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / (len(test_loader) * test_loader.batch_size)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

def reset_GPU():
    gc.collect()
    torch.cuda.empty_cache()
