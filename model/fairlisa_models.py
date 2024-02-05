import torch
import torch.nn as nn


class Filter(nn.Module):
    def __init__(self, embedding_dim, dense_layer_dim, device):
        super(Filter, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, dense_layer_dim),
            nn.ReLU(),
            nn.Linear(dense_layer_dim, dense_layer_dim),
            nn.ReLU(),
            nn.Linear(dense_layer_dim, embedding_dim)
        ).to(self.device)

    def forward(self, x):
        return self.network(x)

    # Possible to extend to the 'combine' scenario as well #################
    # if self.num_features != 0:
    #     if self.filter_mode == "separate":
    #         return torch.stack(
    #             [filter_dict["1"](vectors) for _ in range(self.num_features)], 0
    #         )
    #     elif self.filter_mode == "combine":
    #         result = []
    #         if mask == None:
    #             for i in range(self.num_features):
    #                 result.append(filter_dict[str(i + 1)](vectors))
    #         else:
    #             result.append(filter_dict[str(mask + 1)](vectors))
    #         return torch.stack(result, 0)
    #     else:
    #         assert "error!"
    # else:
    #     return vectors

class Discriminator(nn.Module):
    def __init__(self, embed_dim, latent_size, num_classes, device, binary=False):
        super().__init__()
        if binary:
            self.criterion = nn.BCELoss()
        else:
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
