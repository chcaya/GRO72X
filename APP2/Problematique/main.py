#! usr/bin/python3
import argparse
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from dataset import ConveyorSimulator
from metrics import AccuracyMetric, MeanAveragePrecisionMetric, SegmentationIntersectionOverUnionMetric
from models.classification_network import ClassificationNetwork
from models.detection_network import DetectionNetwork, SimpleDetLoss
from models.segmentation_network import SegmentationNetwork
from visualizer import Visualizer

TRAIN_VALIDATION_SPLIT = 0.9
CLASS_PROBABILITY_THRESHOLD = 0.5
INTERSECTION_OVER_UNION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
SEGMENTATION_BACKGROUND_CLASS = 3


class ConveyorCnnTrainer():
    def __init__(self, args):
        self._args = args
        # Initialisation de pytorch
        use_cuda = args.use_gpu and torch.cuda.is_available()
        self._device = torch.device('cuda' if use_cuda else 'cpu')
        seed = np.random.rand()
        torch.manual_seed(seed)
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Generation des 'path'
        self._dir_path = os.path.dirname(__file__)
        self._train_data_path = os.path.join(self._dir_path, 'data', 'training')
        self._test_data_path = os.path.join(self._dir_path, 'data', 'test')
        self._weights_path = os.path.join(self._dir_path, 'weights', 'task_' + self._args.task + '_best.pt')

        # Initialisation du model et des classes pour l'entraînement
        self._model = self._create_model(self._args.task).to(self._device)
        self._criterion = self._create_criterion(self._args.task)

        print('Model : ')
        print(self._model)
        print('\nNumber of parameters in the model : ', sum(p.numel() for p in self._model.parameters()))

    def _create_model(self, task):
        if task == 'classification':
            return ClassificationNetwork()
        elif task == 'detection':
            return DetectionNetwork()
        elif task == 'segmentation':
            return SegmentationNetwork(n_channels=1, n_classes=4)
        else:
            raise ValueError('Not supported task')

    def _create_criterion(self, task):
        if task == 'classification':
            return nn.BCEWithLogitsLoss()
        elif task == 'detection':
            # def detection_loss(prediction, target):
            #     """
            #     A standard object detection loss function that combines confidence, bounding
            #     box, and classification losses.

            #     :param prediction: The (N, 3, 7) output from the model.
            #                        Format: [conf, x, y, size, score_c0, score_c1, score_c2]
            #     :param target: The (N, 3, 5) ground truth tensor.
            #                    Format: [objectness, x, y, size, class_index]
            #     """
            #     # --- Hyperparameters for balancing the loss components ---
            #     lambda_coord = 5.0
            #     lambda_noobj = 0.5
                
            #     # --- Create a mask to find which prediction slots contain an object ---
            #     obj_mask = target[..., 0] == 1
            #     noobj_mask = target[..., 0] == 0

            #     # --- 1. Confidence (Objectness) Loss ---
            #     loss_conf_obj = F.binary_cross_entropy_with_logits(
            #         prediction[..., 0][obj_mask],
            #         target[..., 0][obj_mask]
            #     )
            #     loss_conf_noobj = F.binary_cross_entropy_with_logits(
            #         prediction[..., 0][noobj_mask],
            #         target[..., 0][noobj_mask]
            #     )
            #     loss_confidence = loss_conf_obj + (lambda_noobj * loss_conf_noobj)
                
            #     # --- 2. Bounding Box Loss (Localization) ---
            #     loss_bbox = torch.tensor(0.0, device=prediction.device)
            #     if obj_mask.sum() > 0:
            #         bbox_pred = prediction[..., 1:4][obj_mask]
            #         bbox_target = target[..., 1:4][obj_mask]
            #         loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_target, reduction='mean')
                    
            #     # --- 3. Classification Loss ---
            #     loss_class = torch.tensor(0.0, device=prediction.device)
            #     if obj_mask.sum() > 0:
            #         class_pred_logits = prediction[..., 4:][obj_mask]
            #         target_class_indices = target[..., 4][obj_mask].long()
            #         loss_class = F.cross_entropy(class_pred_logits, target_class_indices, reduction='mean')

            #     # --- Final Combined Loss ---
            #     print(f"BBox Loss: {(lambda_coord * loss_bbox).item():.4f}, Conf Loss: {loss_confidence.item():.4f}, Class Loss: {loss_class.item():.4f}")
            #     total_loss = loss_confidence + (lambda_coord * loss_bbox) + loss_class
                
            #     return total_loss

            # return detection_loss

            return SimpleDetLoss()
        elif task == 'segmentation':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError('Not supported task')

    def _create_metric(self, task):
        if task == 'classification':
            return AccuracyMetric(CLASS_PROBABILITY_THRESHOLD)
        elif task == 'detection':
            return MeanAveragePrecisionMetric(3, INTERSECTION_OVER_UNION_THRESHOLD)
        elif task == 'segmentation':
            return SegmentationIntersectionOverUnionMetric(SEGMENTATION_BACKGROUND_CLASS)
        else:
            raise ValueError('Not supported task')

    def test(self):
        params_test = {'batch_size': self._args.batch_size, 'shuffle': False, 'num_workers': 4}

        dataset_test = ConveyorSimulator(self._test_data_path, self.transform)
        test_loader = torch.utils.data.DataLoader(dataset_test, **params_test)

        test_metric = self._create_metric(self._args.task)
        visualizer = Visualizer('test', self._args.task, CLASS_PROBABILITY_THRESHOLD, CONFIDENCE_THRESHOLD,
                                SEGMENTATION_BACKGROUND_CLASS)

        print('Test data : ', len(dataset_test))
        self._model.load_state_dict(torch.load(self._weights_path))
        self._model.eval()

        test_loss = 0
        with torch.no_grad():
            for image, segmentation_target, boxes, class_labels in test_loader:
                image = image.to(self._device)
                segmentation_target = segmentation_target.to(self._device)
                boxes = boxes.to(self._device)
                class_labels = class_labels.to(self._device)

                loss = self._test_batch(self._args.task, self._model, self._criterion, test_metric,
                                        image, segmentation_target, boxes, class_labels)
                test_loss += loss.item()

        test_loss /= len(dataset_test)
        print('Test - Average loss: {:.6f}, {}: {:.6f}'.format(
            test_loss, test_metric.get_name(), test_metric.get_value()))

        prediction = self._model(image)

        # Get the number of images in the current batch
        batch_size = image.shape[0]
        # Pick a random index from 0 to (batch_size - 1)
        random_idx = random.randint(0, batch_size - 1)

        # Show the prediction for the randomly selected image
        visualizer.show_prediction(
            image[random_idx], 
            prediction[random_idx], 
            segmentation_target[random_idx], 
            boxes[random_idx], 
            class_labels[random_idx]
        )

    def train(self):
        epochs_train_losses = []
        epochs_validation_losses = []
        epochs_train_metrics = []
        epochs_validation_metrics = []
        best_validation = 0
        nb_worse_validation = 0

        params_train = {'batch_size': self._args.batch_size, 'shuffle': True, 'num_workers': 4}
        params_validation = {'batch_size': self._args.batch_size, 'shuffle': False, 'num_workers': 4}

        dataset_trainval = ConveyorSimulator(self._train_data_path, self.transform)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_trainval,
                                                                          [int(len(
                                                                              dataset_trainval) * TRAIN_VALIDATION_SPLIT),
                                                                           int(len(dataset_trainval) - int(len(
                                                                               dataset_trainval) * TRAIN_VALIDATION_SPLIT))])
        train_loader = torch.utils.data.DataLoader(dataset_train, **params_train)
        validation_loader = torch.utils.data.DataLoader(dataset_validation, **params_validation)

        print('Number of epochs : ', self._args.epochs)
        print('Training data : ', len(dataset_train))
        print('Validation data : ', len(dataset_validation))

        optimizer = optim.Adam(self._model.parameters(), lr=self._args.lr)
        train_metric = self._create_metric(self._args.task)
        validation_metric = self._create_metric(self._args.task)

        visualizer = Visualizer('train', self._args.task, CLASS_PROBABILITY_THRESHOLD, CONFIDENCE_THRESHOLD,
                                SEGMENTATION_BACKGROUND_CLASS)

        for epoch in range(1, self._args.epochs + 1):
            print('\nEpoch: {}'.format(epoch))
            # Entraînement
            self._model.train()
            train_loss = 0
            train_metric.clear()

            # Boucle pour chaque batch
            for image, segmentation_target, boxes, class_labels in train_loader:
                image = image.to(self._device)
                segmentation_target = segmentation_target.to(self._device)
                boxes = boxes.to(self._device)
                class_labels = class_labels.to(self._device)

                loss = self._train_batch(self._args.task, self._model, self._criterion, train_metric, optimizer,
                                         image, segmentation_target, boxes, class_labels)

                train_loss += loss.item()

            # Affichage après la batch
            train_loss = train_loss / len(dataset_train)
            epochs_train_losses.append(train_loss)
            epochs_train_metrics.append(train_metric.get_value())
            print('Train - Average Loss: {:.6f}, {}: {:.6f}'.format(
                train_loss, train_metric.get_name(), train_metric.get_value()))

            # Validation
            self._model.eval()
            validation_loss = 0
            validation_metric.clear()
            with torch.no_grad():
                for image, masks, boxes, labels in validation_loader:
                    image = image.to(self._device)
                    masks = masks.to(self._device)
                    boxes = boxes.to(self._device)
                    labels = labels.to(self._device)

                    loss = self._test_batch(self._args.task, self._model, self._criterion, validation_metric,
                                            image, masks, boxes, labels)
                    validation_loss += loss.item()

            validation_metric_value = validation_metric.get_value()
            validation_loss /= len(dataset_validation)
            if validation_metric_value > best_validation:
                best_validation = validation_metric_value
                nb_worse_validation = 0
                print('Saving new best model')
                torch.save(self._model.state_dict(), self._weights_path)
            else:
                nb_worse_validation += 1

            epochs_validation_losses.append(validation_loss)
            epochs_validation_metrics.append(validation_metric.get_value())
            print('Validation - Average loss: {:.6f}, {}: {:.6f}'.format(
                validation_loss, validation_metric.get_name(), validation_metric_value))

            prediction = self._model(image)

            # Get the number of images in the current batch
            batch_size = image.shape[0]
            # Pick a random index from 0 to (batch_size - 1)
            random_idx = random.randint(0, batch_size - 1)

            # Show the prediction for the randomly selected image
            visualizer.show_prediction(
                image[random_idx], 
                prediction[random_idx], 
                masks[random_idx], 
                boxes[random_idx], 
                labels[random_idx]
            )
            
            visualizer.show_learning_curves(epochs_train_losses, epochs_validation_losses,
                                            epochs_train_metrics, epochs_validation_metrics,
                                            train_metric.get_name())

        ans = input('Do you want ot test? (y/n):')
        if ans == 'y':
            self.test()

    def _train_batch(self, task, model, criterion, metric, optimizer, image, segmentation_target, boxes, class_labels):
        """
        Méthode qui effectue une passe d'entraînement sur un lot de données.
        Vous devez appeler la méthode "accumulate" de l'objet "metric" pour que le calcul de la métrique se fasse.
        La définition des paramètres de cette méthode se trouve dans le fichier "metrics.py"
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param task: La tâche à effectuer ('classification', 'detection' ou 'segmentation')
        :param model: Le modèle créé dans create_model
        :param criterion: La fonction de coût créée dans create_criterion
        :param metric: La métrique créée dans create_metric
        :param optimizer: L'optimisateur pour entraîner le modèle
        :param image: Le tenseur contenant les images du lot à passer au modèle
            Dimensions : (N, 1, H, W)
        :param segmentation_target: Le tenseur cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        :param boxes: Le tenseur cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.

        :param class_labels: Le tenseur cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        :return: La valeur de la fonction de coût pour le lot
        """

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted output by passing inputs to the model
        prediction = model(image)

        # Calculate the loss and update the metric based on the task
        if task == 'classification':
            target = class_labels.float()
            loss = criterion(prediction, target)
            metric.accumulate(torch.sigmoid(prediction), target)

        elif task == 'detection':
            target = boxes
            loss = criterion(prediction, target)
            # The metric expects confidence scores as probabilities, so we apply sigmoid
            pred_for_metric = prediction.detach().clone()
            pred_for_metric[:, :, 0] = torch.sigmoid(pred_for_metric[:, :, 0])
            metric.accumulate(pred_for_metric, target)

        elif task == 'segmentation':
            target = segmentation_target.long()
            loss = criterion(prediction, target)
            metric.accumulate(prediction, target)

        else:
            raise ValueError(f"Task '{task}' is not supported.")
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        return loss

    def _test_batch(self, task, model, criterion, metric, image, segmentation_target, boxes, class_labels):
        """
        Méthode qui effectue une passe de validation ou de test sur un lot de données.
        Vous devez appeler la méthode "accumulate" de l'objet "metric" pour que le calcul de la métrique se fasse.
        La définition des paramètres de cette méthode se trouve dans le fichier "metrics.py"
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param task: La tâche à effectuer ('classification', 'detection' ou 'segmentation')
        :param model: Le modèle créé dans create_model
        :param criterion: La fonction de coût créée dans create_criterion
        :param metric: La métrique créée dans create_metric
        :param image: Le tenseur PyTorch contenant les images du lot à passer au modèle
            Dimensions : (N, 1, H, W)
        :param segmentation_target: Le tenseur PyTorch cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        :param boxes: Le tenseur PyTorch cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.

        :param class_labels: Le tenseur PyTorch cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        :return: La valeur de la fonction de coût pour le lot
        """

        # Forward pass: compute predicted output by passing inputs to the model
        prediction = model(image)

        # Calculate the loss and update the metric based on the task
        if task == 'classification':
            target = class_labels.float()
            loss = criterion(prediction, target)
            metric.accumulate(torch.sigmoid(prediction), target)

        elif task == 'detection':
            target = boxes
            loss = criterion(prediction, target)
            pred_for_metric = prediction.detach().clone()
            pred_for_metric[:, :, 0] = torch.sigmoid(pred_for_metric[:, :, 0])
            metric.accumulate(pred_for_metric, boxes)

        elif task == 'segmentation':
            target = segmentation_target.long()
            loss = criterion(prediction, target)
            metric.accumulate(prediction, target)
            
        else:
            raise ValueError(f"Task '{task}' is not supported.")

        return loss


if __name__ == '__main__':
    #  Settings
    parser = argparse.ArgumentParser(description='Conveyor CNN')
    parser.add_argument('--mode', choices=['train', 'test'], help='The script mode', required=True)
    parser.add_argument('--task', choices=['classification', 'detection', 'segmentation'],
                        help='The CNN task', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training (default: 20)')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate used for training (default: 4e-4)')
    parser.add_argument('--use_gpu', action='store_true', help='use the gpu instead of the cpu')
    parser.add_argument('--early_stop', type=int, default=25,
                        help='number of worse validation loss before quitting training (default: 25)')

    args = parser.parse_args()

    conv = ConveyorCnnTrainer(args)

    if args.mode == 'train':
        print('\n--- Training mode ---\n')
        conv.train()
    elif args.mode == 'test':
        print('\n--- Testing mode ---\n')
        conv.test()
