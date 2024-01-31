from torch.functional import F
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss


class NCF(torch.nn.Module):
    def __init__(self, n_users, n_movies, latent_dim, layers,
                 min_rating, max_rating):
        super(NCF, self).__init__()
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_users = n_users
        self.num_items = n_movies
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, scale_back_to_ratings=False):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)

        logits = self.affine_output(vector)
        predictions = self.logistic(logits)
        if scale_back_to_ratings:
            predictions = self.scale_back_to_ratings(predictions)
        return predictions

    def scale_back_to_ratings(self, ratings_zero_one):
        ratings_min_max = (ratings_zero_one * (self.max_rating - self.min_rating) + self.min_rating).round()
        return ratings_min_max

    def get_embeddings(self, user_indices, item_indices):
        # Retrieve embeddings
        user_embeddings = self.embedding_user(user_indices)
        item_embeddings = None
        if item_indices is not None:
            item_embeddings = self.embedding_item(item_indices)
        return user_embeddings, item_embeddings

    def predict_batch_with_embeddings(self, user_embeddings, movie_embeddings, scale_back_to_ratings=False):
        with torch.no_grad():
            vector = torch.cat([user_embeddings, movie_embeddings], dim=-1)  # the concat latent vector
            for idx, _ in enumerate(range(len(self.fc_layers))):
                vector = self.fc_layers[idx](vector)
                vector = torch.nn.ReLU()(vector)
                # vector = torch.nn.BatchNorm1d()(vector)
                # vector = torch.nn.Dropout(p=0.5)(vector)

            logits = self.affine_output(vector)
            pred_ratings = self.logistic(logits)  # This is a score between 0 and 1
            if scale_back_to_ratings:
                pred_ratings = self.scale_back_to_ratings(pred_ratings)
        return pred_ratings

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def evaluate(self, valid_loader, denormalize_predictions=False, denormalize_labels=False):
        with torch.no_grad():
            mse_total, count = 0, 0
            for user_indices_val, movie_indices_val, ratings_val in valid_loader:
                if denormalize_labels:
                    ratings_val = self.scale_back_to_ratings(ratings_val)
                predictions = self.forward(user_indices_val, movie_indices_val, scale_back_to_ratings=denormalize_predictions)
                mse_total += F.mse_loss(predictions.squeeze(), ratings_val) * len(user_indices_val)
                count += len(user_indices_val)
            avg_mse = mse_total / count
        return avg_mse.item(), np.sqrt(avg_mse).item()


class PMF(torch.nn.Module):
    def __init__(self, n_users, n_movies, latent_vectors=5, lam_u=0.3, lam_v=0.3, explicit=True, min_rating=1, max_rating=5):
        super(PMF, self).__init__()
        self.user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
        self.user_features.data.mul_(0.01)
        self.movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
        self.movie_features.data.mul_(0.01)
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.explicit = explicit
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, matrix):
        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        scores = torch.mm(self.user_features, self.movie_features.t())

        if self.explicit:
            predicted = torch.sigmoid(scores)
            diff = (matrix - predicted) ** 2
            prediction_error = torch.sum(diff * non_zero_mask)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(scores, matrix, reduction='none')
            prediction_error = torch.sum(bce_loss * non_zero_mask)  # Negative sign for BCE

        u_regularization = self.lam_u * torch.sum(self.user_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(self.movie_features.norm(dim=1))
        total_loss = prediction_error + u_regularization + v_regularization
        return total_loss

    def predict(self, user_idx, rating_matrix):
        # Predict all movie ratings that user_idx watched (i.e. the observed/labeled data only)
        # Only used for training/evaluating the model
        user_ratings = rating_matrix[user_idx, :]
        true_ratings = user_ratings != -1
        predictions = torch.sigmoid(torch.mm(self.user_features[user_idx, :].view(1, -1), self.movie_features.t()))

        if self.explicit:
            predicted_ratings = self.scale_for_explicit(predictions.squeeze()[true_ratings])
            actual_ratings = self.scale_for_explicit(user_ratings[true_ratings])
        else:
            predicted_ratings = predictions.squeeze()[true_ratings]
            actual_ratings = user_ratings[true_ratings]
        return predicted_ratings, actual_ratings

    def predict_batch(self, user_indices, rating_matrix):
        # Predict all movie ratings that user i watched for all users in user_indices
        # (i.e. the observed/labeled data only)
        # Only used for training/evaluating the model
        user_ratings = rating_matrix[user_indices]
        valid_ratings_mask = (user_ratings != -1)  # Mask to identify valid ratings

        all_predictions = torch.sigmoid(torch.mm(self.user_features[user_indices], self.movie_features.t()))

        predicted_ratings = []
        actual_ratings = []
        for i, user_idx in enumerate(user_indices):
            # Filter predictions and actual ratings for valid ratings only
            valid_predictions = all_predictions[i][valid_ratings_mask[i]]
            valid_actual_ratings = user_ratings[i][valid_ratings_mask[i]]

            if self.explicit:
                valid_predictions = self.scale_for_explicit(valid_predictions)
                valid_actual_ratings = self.scale_for_explicit(valid_actual_ratings)

            predicted_ratings.append(valid_predictions)
            actual_ratings.append(valid_actual_ratings)

        return predicted_ratings, actual_ratings

    def save_model(self, path):
        torch.save({
            'user_features': self.user_features,
            'movie_features': self.movie_features
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.user_features = checkpoint['user_features']
        self.movie_features = checkpoint['movie_features']

    def evaluate(self, valid_loader, denormalize_predictions=False, denormalize_labels=False):
        with torch.no_grad():
            if self.explicit:
                mse_total, count = 0, 0
                for user_indices_val, movie_indices_val, ratings_val in valid_loader:
                    user_embeddings, movie_embeddings = self.get_embeddings(user_indices_val, movie_indices_val)
                    if denormalize_labels:
                        ratings_val = self.scale_for_explicit(ratings_val)
                    predictions = self.predict_batch_with_embeddings(user_embeddings, movie_embeddings,
                                                                     scale_back_to_ratings=denormalize_predictions)
                    mse_total += F.mse_loss(predictions.squeeze(), ratings_val) * len(user_indices_val)
                    count += len(user_indices_val)
                avg_mse = mse_total / count
                return avg_mse.item(), np.sqrt(avg_mse).item()
            else:
                all_predictions = []
                all_labels = []

                for user_indices_val, movie_indices_val, ratings_val in valid_loader:
                    user_embeddings, movie_embeddings = self.get_embeddings(user_indices_val, movie_indices_val)
                    predictions = self.predict_batch_with_embeddings(user_embeddings, movie_embeddings,
                                                                     scale_back_to_ratings=False)
                    # Store predictions and labels
                    all_predictions.extend(predictions.squeeze().cpu().numpy())
                    all_labels.extend(ratings_val.cpu().numpy())

                epsilon = 1e-15
                all_predictions = [max(min(p, 1 - epsilon), epsilon) for p in all_predictions]

                # Calculate the BCE loss using log_loss
                bce_loss = log_loss(all_labels, all_predictions)

                # Calculate the AUC-ROC score
                auc_roc_score = roc_auc_score(all_labels, all_predictions)

                return bce_loss, auc_roc_score

    def scale_for_explicit(self, predictions):
        return (predictions * (self.max_rating - self.min_rating) + self.min_rating).round()

    def get_embeddings(self, user_indices, movie_indices):
        # Retrieve embeddings
        user_embeddings = self.user_features[user_indices]
        movie_embeddings = None
        if movie_indices is not None:
            movie_embeddings = self.movie_features[movie_indices]
        return user_embeddings, movie_embeddings

    def predict_single_with_embeddings(self, user_embedding, movie_embedding,  scale_back_to_ratings=True):
        with torch.no_grad():
            prediction = torch.dot(user_embedding, movie_embedding.t())
            prediction = torch.sigmoid(prediction)
            if scale_back_to_ratings:
                prediction = self.scale_for_explicit(prediction)
        return prediction

    def predict_batch_with_embeddings(self, user_embeddings, movie_embeddings, scale_back_to_ratings=True):
        with torch.no_grad():
            # Ensure user_embeddings and movie_embeddings are tensors
            if not isinstance(user_embeddings, torch.Tensor):
                user_embeddings = torch.stack(user_embeddings)
            if not isinstance(movie_embeddings, torch.Tensor):
                movie_embeddings = torch.stack(movie_embeddings)

            # Check if the lengths of user_embeddings and movie_embeddings are the same
            if user_embeddings.shape[0] != movie_embeddings.shape[0]:
                raise ValueError("The number of user embeddings and movie embeddings must be equal.")

            # Perform batch matrix multiplication (dot product for each pair)
            predictions = torch.sum(user_embeddings * movie_embeddings, dim=1)
            predictions = torch.sigmoid(predictions)
            if scale_back_to_ratings:
                predictions = self.scale_for_explicit(predictions)
        return predictions


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, n_layers, min_rating, max_rating):
        super(LightGCN, self).__init__()
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_size)
        self.item_embeddings = nn.Embedding(n_items, embedding_size)

        # Initialization
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, user_indices, item_indices, adj_matrix):
        user_emb = self.user_embeddings(user_indices)
        item_emb = self.item_embeddings(item_indices)

        all_user_embeddings = [user_emb]
        all_item_embeddings = [item_emb]

        # Graph convolution layers
        for _ in range(self.n_layers):
            user_emb = torch.sparse.mm(adj_matrix, user_emb)
            item_emb = torch.sparse.mm(adj_matrix.t(), item_emb)
            all_user_embeddings.append(user_emb)
            all_item_embeddings.append(item_emb)

        # Aggregate embeddings from all layers
        final_user_embeddings = torch.mean(torch.stack(all_user_embeddings, dim=1), dim=1)
        final_item_embeddings = torch.mean(torch.stack(all_item_embeddings, dim=1), dim=1)

        # Compute prediction
        scores = torch.sum(final_user_embeddings * final_item_embeddings, dim=1)
        predictions = torch.sigmoid(scores)
        return predictions

    def scale_back_to_ratings(self, predictions):
        ratings_min_max = (predictions * (self.max_rating - self.min_rating) + self.min_rating).round()
        return ratings_min_max

    @staticmethod
    def compute_adjacency_matrix(interactions, n_users, n_items):
        """
        Create a sparse adjacency matrix.

        Args:
            interactions (DataFrame): DataFrame with columns ['UserID', 'MovieID', 'Rating'].
            n_users (int): Number of users.
            n_items (int): Number of items.

        Returns:
            sp.coo_matrix: A sparse adjacency matrix.
        """
        # Create user-item interaction matrix
        user_idx = interactions['UserID'].values
        item_idx = interactions['MovieID'].values + n_users  # Shift item indices

        # Create the adjacency matrix
        ratings = interactions['Rating'].values
        adj_matrix = sp.coo_matrix((ratings, (user_idx, item_idx)),
                                   shape=(n_users + n_items, n_users + n_items),
                                   dtype=np.float32)

        # Create the symmetric adjacency matrix
        adj_matrix_symmetric = adj_matrix + adj_matrix.T

        # Normalize the matrix
        rowsum = np.array(adj_matrix_symmetric.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_adj_matrix = d_mat_inv_sqrt @ adj_matrix_symmetric @ d_mat_inv_sqrt

        return normalized_adj_matrix.tocoo()

