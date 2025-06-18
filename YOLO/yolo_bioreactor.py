from ultralytics import YOLO
import numpy as np
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
import argparse
import matplotlib.pyplot as plt
import time


def predicition_yolo(yolo_path: str, path_data_dir: str = 'val'):
    torch.manual_seed(42)

    model_yolo = YOLO(yolo_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to a specific size
        transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    ])

    testset_images = ImageFolder(root=path_data_dir, transform=transform_test)

    testloader_images = DataLoader(testset_images, batch_size=16, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        labels_complete = []
        predicted_complete = []
        for data in testloader_images:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_yolo.predict(images)
            for i, orig_label in zip(outputs, labels):
                predicted = i.probs.top1
                total += 1
                correct += (predicted == orig_label).sum().item()
                labels_complete.append(orig_label.cpu().numpy())
                predicted_complete.append(predicted)

    # Get metrics
    accuracy = 100 * correct / total
    print(accuracy)

    # calc confusion matrix
    y_true = labels_complete
    y_pred = predicted_complete
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # calc balanced accuracy
    bacc = 0
    for i in range(3):
        bacc += cm[i][i] / np.sum(cm[i])
    bacc /= 3

    return [path_data_dir, bacc*100, accuracy, cm]


def train_yolo(yolo_path: str, data_path: str, epochs: int = 70):
    model = YOLO(yolo_path)
    model.train(data=data_path, epochs=epochs, plots=True)


def main(data_path_predict, save_results=False, results_path='None', train=False, data_path_train='None', epochs=70):
    start_time = time.strftime("%d%m-%H%M")

    YOLO_PATH = r'F:\KI in den Life Sciences\hautkrebserkennung\YOLO\models\yolov8m-cls.pt'

    if train:
        train_yolo(YOLO_PATH, data_path_train, epochs)
    else:
        metric_best = predicition_yolo(YOLO_PATH, data_path_predict)
        print('------------------- Metrics best -------------------')
        print(metric_best)

        if save_results:
            os.makedirs(results_path, exist_ok=True)
            # write results to csv file
            with open(results_path+'/results_validation_'+str(start_time)+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["data", "bacc", "acc"])
                writer.writerow(metric_best[:3])

            # plot confusion matrix
            cm = metric_best[3]
            cm_display = ConfusionMatrixDisplay(cm, display_labels=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
            cm_display.plot(cmap='Blues')
            cm_display.ax_.set_title('Confusion Matrix')
            plt.savefig(results_path+'/confusion_matrix_'+str(start_time)+'.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser arguments for prediction
    parser.add_argument('--image_path', type=str, default=r'F:\KI in den Life Sciences\dataset3\val')
    parser.add_argument('--save_results', type=bool, default=False)
    parser.add_argument('--results_path', type=str, default='./results')

    # parser arguments for training
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default=r'F:\KI in den Life Sciences\dataset3')
    parser.add_argument('--epochs', type=int, default=70)
    

    args = parser.parse_args()
    main(args.image_path, args.save_results, args.results_path, args.train, args.data_path, args.epochs)
