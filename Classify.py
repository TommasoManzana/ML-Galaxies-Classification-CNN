import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import csv
import os
import re

#   ---------- FUNCTIONS ----------

# Used to sort the paths in the images folder
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# def classify(model, image, classes):
#     model = model.eval()
#     output = model(image)
#     _, predicted = torch.max(output.data, 1)
    
#     print(classes[predicted.item()])
#     return classes[predicted.item()]

# Classify each image
def classify(model, transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    # print(classes[predicted.item()])
    return classes[predicted.item()]


#   ---------------------------------

def main():

    # Galaxies' classes
    classes = [
        "Barred Spiral",
        "Cigar Shaped Smooth",
        "Disturbed",
        "Edge-on with Bulge",
        "Edge-on without Bulge",
        "In-between Round Smooth",
        "Merging",
        "Round Smooth",
        "Unbarred Loose Spiral",
        "Unbarred Tight Spiral"
    ]

    # Import saved model
    model = torch.load('best_model.pth')

    # Computed normalization values
    mean = [0.1675, 0.1626, 0.1589]
    std = [0.1231, 0.1118, 0.1047]

    # Transform
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Dataset import & load
    # test_dataset_path = "C://Users//tommy//Desktop//Exam//Galaxies//test"
    # test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    # with open('results.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for i, (images, labels) in enumerate(test_loader):
    #         predicted_class = classify(model, images, classes)
    #         writer.writerow([i, predicted_class])
    
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        folder_path = "C://Users//tommy//Desktop//Exam//Galaxies//test//1"
        images_paths = sorted_alphanumeric(os.listdir(folder_path))

        index = 0
        for image_path in images_paths:
            input_path = os.path.join(folder_path, image_path)
            predicted_class = classify(model, test_transforms, input_path, classes)
            writer.writerow([index, predicted_class])
            index += 1

        
if __name__ == "__main__":
    main()
