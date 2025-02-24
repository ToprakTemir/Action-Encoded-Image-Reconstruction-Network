import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


from homework1 import load_dataset


# takes in the image of scene and executed action as inputs, and predicts the final object position as (x, y)
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(3 * 128 * 128 + 1, 256),  # 3 channels for RGB, 128x128 image, + 1 for action
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)         # 2 output for x, y
        )

    def forward(self, x):
        return self.nn(x)


def train(dataLoader_training, dataLoader_validation, epochs=100):
    model = DNN()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train(True)

    training_loss_graph = []
    validation_loss_graph = []
    validation_min_loss = np.inf

    for epoch in range(epochs):

        # training
        training_loss = 0
        for i, (positions, actions, images, _) in enumerate(dataLoader_training):
            optimizer.zero_grad()

            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)
            loss = mse_loss(outputs, positions)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss_graph.append(training_loss / len(dataLoader_training))

        # validation
        validation_loss = 0

        for i, (positions, actions, images, _) in enumerate(dataLoader_validation):
            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)
            loss = mse_loss(outputs, positions)
            validation_loss += loss.item()

        validation_loss_graph.append(validation_loss / len(dataLoader_validation))
        if validation_loss < validation_min_loss:
            validation_min_loss = validation_loss
            torch.save(model, "best_DNN.pth")

        print(f"Epoch: {epoch}, Training Loss: {training_loss / len(dataLoader_training)}, Validation Loss: {validation_loss / len(dataLoader_validation)}")


    # draw the loss graph
    plt.figure()
    plt.plot(training_loss_graph, label="Training Loss")
    plt.plot(validation_loss_graph, label="Validation Loss")
    plt.legend()
    plt.show()

    model.train(False)


def test(dataLoader_testing, model=torch.load("best_DNN.pth")):
    with torch.no_grad():
        for i, (positions, actions, images, _) in enumerate(dataLoader_testing):
            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)

            for i in range(4):
                print(f"Predicted: {outputs[i]}, Real: {positions[i]}, difference: {outputs[i] - positions[i]}")

            # graph the predicted and real x,y positions as pairs for every single predicted-real pair
            plt.figure()
            for i in range(len(outputs)):
                plt.xlim(0, 1)
                plt.ylim(-1, 1)
                plt.plot(outputs[i][0], outputs[i][1], 'ro')
                plt.plot(positions[i][0], positions[i][1], 'bo')

                # make a legend saying that red is the predicted position and blue is the real position
                plt.legend(["Predicted", "Real"])

                plt.show()


if __name__ == "__main__":

    dataloader_training, dataloader_validation, dataloader_testing = load_dataset()

    train(dataloader_training, dataloader_validation, epochs=100)
    # test(model=torch.load("best_DNN.pth"), dataLoader_testing=dataloader_testing)

