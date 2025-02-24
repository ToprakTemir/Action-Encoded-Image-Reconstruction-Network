import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from homework1 import load_dataset


# takes in the image and action, outputs the final image
class CNNWithReconstruction(nn.Module):
    def __init__(self):
        super(CNNWithReconstruction, self).__init__()

        # encode image features using CNN
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU()
        )

        # combine image and action features
        self.encode_action_into_image = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 1, 1024), # 256 * 8 * 8 + 1 for action
            nn.ReLU(),
            nn.Linear(1024, 256 * 8 * 8)
        )

        self.predict_xy_from_encoded_image = nn.Linear(256 * 8 * 8, 2)

        # decode image features after being combined with action features
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64 -> 128x128
            nn.Sigmoid()
        )

    def forward(self, image, action):
        image_features = self.image_encoder(image)

        image_features = image_features.view(image_features.size(0), -1) # flatten along batch dimension
        action = action.view(-1, 1)
        image_features_and_action = torch.cat((image_features, action), dim=-1)

        combined_features = self.encode_action_into_image(image_features_and_action)

        decoded_image = self.image_decoder(combined_features.view(-1, 256, 8, 8))
        # decoded_image = decoded_image * 255
        return decoded_image


    # this loss was before I realized we were allowed to use target images, and is now outdated
    # def loss(self, input_images, actions, final_positions):
    #
    #     images_encoded = self.image_encoder(input_images)
    #     images_encoded = images_encoded.view(images_encoded.size(0), -1) # flatten along batch dimension
    #     predicted_xy = self.predict_xy_from_encoded_image(images_encoded)
    #
    #     mse_loss = torch.nn.MSELoss()
    #     real_xy = final_positions
    #     xy_diff_loss = mse_loss(predicted_xy, real_xy).mean()
    #
    #     predicted_images = self.forward(input_images, actions)
    #     predicted_images_encoded = self.image_encoder(predicted_images)
    #     predicted_images_encoded = predicted_images_encoded.view(predicted_images_encoded.size(0), -1) # flatten along batch dimension
    #     xy_in_predicted_images = self.predict_xy_from_encoded_image(predicted_images_encoded)
    #     image_loss = mse_loss(xy_in_predicted_images, real_xy).mean()
    #
    #     return xy_diff_loss + image_loss


    def loss(self, input_images, actions, target_images, final_positions):

        latent_image = self.image_encoder(input_images)
        latent_image = latent_image.view(latent_image.size(0), -1) # flatten along batch dimension

        predicted_xy = self.predict_xy_from_encoded_image(latent_image)
        position_prediction_loss = torch.nn.MSELoss()(predicted_xy, final_positions).mean()

        latent_image_and_action = torch.cat((latent_image, actions.view(-1, 1)), dim=-1)
        combined_features = self.encode_action_into_image(latent_image_and_action)

        decoded_image = self.image_decoder(combined_features.view(-1, 256, 8, 8))
        target_images = target_images.view(-1, 3, 128, 128)
        image_reconstruction_loss = torch.nn.MSELoss()(decoded_image, target_images).mean()

        reconstructed_latent_image = self.image_encoder(decoded_image)
        reconstructed_latent_image = reconstructed_latent_image.view(reconstructed_latent_image.size(0), -1) # flatten along batch dimension
        predicted_xy_from_reconstruction = self.predict_xy_from_encoded_image(reconstructed_latent_image)
        reconstructed_position_loss = torch.nn.MSELoss()(predicted_xy_from_reconstruction, final_positions).mean()

        return position_prediction_loss + image_reconstruction_loss + reconstructed_position_loss



def train(dataloader_training, dataLoader_validation, epochs=100):
    model = CNNWithReconstruction()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train(True)

    training_loss_graph = []
    validation_loss_graph = []

    best_validation = np.inf

    for epoch in range(epochs):

        # training
        training_loss = 0
        for positions, actions, images, target_images in dataloader_training:
            loss = model.loss(images, actions, target_images, positions)
            loss = loss.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.float().item()

        training_loss_graph.append(training_loss / len(dataloader_training))

        # validation
        validation_loss = 0

        for positions, actions, images, target_images in dataLoader_validation:
            loss = model.loss(images, actions, target_images, positions)
            validation_loss += loss.item()

        validation_loss_graph.append(validation_loss / len(dataLoader_validation))

        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(model.state_dict(), "best_cnn_reconstruction.pth")

        print(f"Epoch: {epoch}, Training Loss: {training_loss / len(dataloader_training)}, Validation Loss: {validation_loss / len(dataLoader_validation)}")


    # draw the loss graph
    plt.figure()
    plt.plot(training_loss_graph, label="Training Loss")
    plt.plot(validation_loss_graph, label="Validation Loss")
    plt.legend()
    plt.show()


    model.train(False)

def test(test_dataset, weights=torch.load("best_cnn_reconstruction.pth")):
    """
    Reconstructs image using the trained CNN model and displays the reconstructed image
    """

    model = CNNWithReconstruction()
    model.load_state_dict(weights)

    with torch.no_grad():
        for _, actions, images, target_images in test_dataset:
            output_imgs = model(images, actions).squeeze(0)

            batch_size = output_imgs.shape[0]

            plt.figure(figsize=(10, 10))
            plt.imshow(target_images[0].permute(1, 2, 0))
            plt.axis("off")
            plt.show()
            for i in range(batch_size // 8):
                img = output_imgs[i].permute(1, 2, 0)  # Convert CHW to HWC

                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.axis("off")
                plt.show()

            break  # only display one batch



if __name__ == "__main__":

    dataloader_training, dataloader_validation, dataloader_testing = load_dataset()

    # train(dataloader_training, dataloader_validation)
    test(weights=torch.load("best_cnn_reconstruction.pth"), test_dataset=dataloader_testing)