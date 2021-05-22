import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Keep track of the plot values
train_loss_values, val_loss_values, train_acc_values, val_acc_values= [], [], [], []

#   ---------- FUNCTIONS ----------

# Use to show a couple of transformed images
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels: ', labels)

# Compute the mean and std of images
def get_mean_std(loader):
    mean = 0.
    std = 0.
    tot_images = 0
    for images, _ in loader:
        images_per_batch = images.size(0)
        images = images.view(images_per_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        tot_images += images_per_batch
    
    mean /= tot_images
    std /= tot_images

    print('mean: ' + mean)
    print('std: ' + std)

    return mean, std

# Use to set CUDA or CPU
def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

# Train the NN
def train_NN(model, train_loader, val_loader, criterion, optimizer, n_epochs):
    # set CUDA (if available)
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch+1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        print("   - Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" %(running_correct, total, epoch_acc, epoch_loss))

        # Append data for plot
        train_loss_values.append(epoch_loss)
        train_acc_values.append(epoch_acc/100)
        
        # log
        f = open('log.txt','a')
        f.write("%.3f \t %.3f \n" %(epoch_loss, (epoch_acc/100)))
        f.close()


        validation_set_acc = evaluate_model_on_validation_set(model, val_loader)

        if (validation_set_acc > best_acc):
            best_acc = validation_set_acc
            save_checkpoint(model, epoch, optimizer, best_acc)

    print("Finished")
    return model

# Eval model
def evaluate_model_on_validation_set(model, val_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("   - Validation dataset. Got %d out of %d images correctly (%.3f%%)" %(predicted_correctly_on_epoch, total, epoch_acc))

    # Append data for plot
    # val_loss_values.append(epoch_loss)
    val_acc_values.append(epoch_acc/100)

    return epoch_acc

# Save best weights
def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_accuracy': best_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')

# Plot charts
def plot(train_values, val_values, n_epochs, title, train_label, val_label, y_label, img_name):
    epochs = range(1, n_epochs+1)
    plt.plot(epochs, train_values, 'g', label=train_label)
    if (val_values):
        plt.plot(epochs, val_values, 'b', label=val_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()

    # save the png
    figure = plt.gcf()
    figure.set_size_inches(8, 8)
    plt.savefig(img_name, dpi = 100)

    plt.show()
    

#   ---------------------------------


def main():
    # PATHS
    train_dataset_path = "C://Users//tommy//Desktop//Exam//Galaxies//train"
    test_dataset_path = "C://Users//tommy//Desktop//Exam//Galaxies//test"

    # Computed normalization values
    mean = [0.1675, 0.1626, 0.1589]
    std = [0.1231, 0.1118, 0.1047]
    # Pytorch mean and std vals:
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]

    # Transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.CenterCrop(175),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])


    # Datasets import
    full_train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

    # Split of the train set to get a validation set
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [9315, 3100])

    # Dataloaders Datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 10 Galaxies' classes
    num_of_classes = 10

    # get_mean_std(train_loader)
    # show_transformed_images(train_dataset)
    
    # Import the model
    resnet50 = models.resnet50(pretrained=True)
    num_ftrs = resnet50.fc.in_features

    # Adjusting the last layer to match classes' number
    resnet50.fc = nn.Linear(num_ftrs, num_of_classes)
    
    # pass the model to GPU
    resnet50 = resnet50.to(set_device())

    # set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003)

    epochs = 50

    train_NN(resnet50, train_loader, val_loader, loss_fn, optimizer, epochs)
    plot(train_loss_values, val_loss_values, epochs, 'Training and Validation Loss', 'Training loss', 'Validation loss', 'Loss', 'Loss.png')
    plot(train_acc_values, val_acc_values, epochs, 'learning rate = 0.001', 'Training accuracy', 'Validation accuracy', 'Accuracy', 'Accuracy.png')
    
    # Used to save the best model as .pth
    best_saved = torch.load('model_best_checkpoint.pth.tar')
    print(best_saved['epoch'])
    print(best_saved['best_accuracy'])
    saved_model = models.resnet50()
    num_ftrs_saved = saved_model.fc.in_features
    saved_model.fc = nn.Linear(num_ftrs_saved, num_of_classes)
    saved_model.load_state_dict(best_saved['model'])
    
    torch.save(saved_model, 'best_model.pth')


if __name__ == "__main__":
    main()
