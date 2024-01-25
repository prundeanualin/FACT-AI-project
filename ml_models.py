import torch
import torch.nn as nn


class FilterModel(nn.Module):
    def __init__(self, embedding_dim, layer_sizes):
        super(FilterModel, self).__init__()
        layers = [nn.Linear(input_size, output_size) for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.dense_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(layer_sizes[-1], embedding_dim)

    def forward(self, x):
        for layer in self.dense_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class Discriminator(nn.Module):
    def __init__(self, embed_dim, latent_size, num_classes, device):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.embed_dim = embed_dim

        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, num_classes),
        ).to(self.device)

    def forward(self, filtered_embeddings, labels):
        predictions = self.predict(filtered_embeddings)
        loss = self.criterion(predictions, labels)
        return loss

    def hforward(self, filtered_embeddings):
        output = self.predict(filtered_embeddings)
        loss = torch.sum(
            -output * torch.log(output + 1e-8)
            - (1 - output) * torch.log(1 - output + 1e-8)
        ) / len(output)
        return loss

    def predict(self, filtered_embeddings):
        output = torch.softmax(self.network(filtered_embeddings), dim=-1)
        return output

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))