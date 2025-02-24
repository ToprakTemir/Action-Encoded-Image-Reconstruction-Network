import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from homework1 import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this one just predicts 2d object position, not the image
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 128 x 128 -> 64 x 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 64 -> 32 x 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 x 32 -> 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 16 x 16 -> 8 x 8
        )

        self.fc = nn.Linear(256 * 8 * 8 + 1, 2) # + 1 for action

    def forward(self, image, action):
        x = self.cnn(image)
        x = x.view(x.shape[0], -1)
        action = action.unsqueeze(1)
        x = torch.cat((x, action), dim=-1)
        x = self.fc(x)
        return x



def train(dataLoader_training, dataLoader_validation, save_path="best_cnn.pth", epochs=100):
    model = CNN()
    model.to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train(True)

    training_loss_graph = []
    validation_loss_graph = []
    validation_min = np.inf

    for epoch in range(epochs):

        # training
        training_loss = 0
        for positions, actions, images, _ in dataLoader_training:
            optimizer.zero_grad()

            images = images.to(device)
            actions = actions.to(device)
            positions = positions.to(device)

            outputs = model(images, actions)
            loss = mse_loss(outputs, positions).mean()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss_graph.append(training_loss / len(dataLoader_training))

        # validation
        validation_loss = 0

        for positions, actions, images, _ in dataLoader_validation:
            images = images.to(device)
            actions = actions.to(device)
            positions = positions.to(device)

            outputs = model(images, actions)
            loss = mse_loss(outputs, positions).mean()
            validation_loss += loss.item()

        validation_loss_graph.append(validation_loss / len(dataLoader_validation))
        if validation_loss < validation_min:
            validation_min = validation_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch: {epoch}, Training Loss: {training_loss / len(dataLoader_training)}, Validation Loss: {validation_loss / len(dataLoader_validation)}")

    # draw the loss graph
    plt.figure()
    plt.plot(training_loss_graph, label="Training Loss")
    plt.plot(validation_loss_graph, label="Validation Loss")
    plt.legend()
    plt.show()

    model.train(False)


def test(dataloader_testing, weights=torch.load("best_cnn.pth")):
    model = CNN()
    model.load_state_dict(weights)
    with torch.no_grad():
        for positions, actions, images, _ in dataloader_testing:
            outputs = model(images, actions)

            for i in range(4):
                output = outputs[i][0].item(), outputs[i][1].item()
                position = positions[i][0].item(), positions[i][1].item()
                # distance = np.sqrt((output[0] - position[0]) ** 2 + (output[1] - position[1]) ** 2)
                # print(f"predicted: {output}, real: {position}, distance: {distance}")

            # graph the predicted and real x,y positions as pairs for every single predicted-real pair
            plt.figure()
            for i in range(len(outputs)):
                plt.xlim(0, 1)
                plt.ylim(-1, 1)
                plt.plot(outputs[i][0], outputs[i][1], 'ro')
                plt.plot(positions[i][0], positions[i][1], 'bo')

                plt.legend(["Predicted", "Real"])
                plt.show()


if __name__ == "__main__":
    dataloader_training, dataloader_validation, dataloader_testing = load_dataset()

    train(dataloader_training, dataloader_validation)
    # test(dataloader_testing, weights=torch.load("best_cnn.pth"))