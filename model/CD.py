import torch
import torch.nn as nn


class BaseCD(nn.Module):
    def __init__(self, args, device):
        super(BaseCD, self).__init__()
        self.user_num = args.USER_NUM
        self.item_num = args.ITEM_NUM
        self.knowledge_num = args.KNOWLEDGE_NUM
        self.criterion = nn.BCELoss()
        self.device = device

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))


class IRT(BaseCD):
    def __init__(self, args, device):
        super(IRT, self).__init__(args, device)
        self.user_embedding_layer = nn.Embedding(self.user_num, 1).to(self.device)
        self.a = nn.Embedding(self.item_num, 1).to(self.device)
        self.b = nn.Embedding(self.item_num, 1).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding_layer.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)

    def predict(self, user_id, item_id, user_embeddings=None):
        if user_embeddings is None:
            user_embeddings = self.user_embedding_layer(user_id)
        alpha = self.a(item_id)
        beta = self.b(item_id)
        pred = alpha * (user_embeddings - beta)
        pred = torch.squeeze(torch.sigmoid(pred), 1)
        out = {"prediction": pred}
        return out

    def forward(self, user_id, item_id, score, user_embeddings=None):
        out = self.predict(user_id, item_id, user_embeddings)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out

    def get_user_embeddings(self, user_id):
        return self.user_embedding_layer(user_id)

class MIRT(BaseCD):
    def __init__(self, args, device):
        super(MIRT, self).__init__(args, device)
        self.user_embedding_layer = nn.Embedding(self.user_num, args.LATENT_NUM).to(self.device)
        self.a = nn.Embedding(self.item_num, args.LATENT_NUM).to(self.device)
        self.b = nn.Embedding(self.item_num, 1).to(self.device)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)

    def predict(self, user_id, item_id, user_embeddings=None):
        if user_embeddings is None:
            user_embeddings = self.user_embedding_layer(user_id)
        alpha = self.a(item_id)
        beta = self.b(item_id)
        pred = torch.sum(alpha * user_embeddings, dim=1).unsqueeze(1) - beta
        pred = torch.squeeze(torch.sigmoid(pred), 1)
        out = {"prediction": pred}
        return out

    def forward(self, user_id, item_id, score, user_embeddings=None):
        out = self.predict(user_id, item_id, user_embeddings)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out

    def get_user_embeddings(self, user_id):
        return self.user_embedding_layer(user_id)

class NCDM(BaseCD):
    def __init__(self, args, device):
        super(NCDM, self).__init__(args, device)
        self.knowledge_dim = args.KNOWLEDGE_NUM
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        self.user_embedding_layer = nn.Embedding(self.user_num, self.knowledge_dim).to(self.device)
        self.k_difficulty = nn.Embedding(self.item_num, self.knowledge_dim).to(self.device)
        self.e_difficulty = nn.Embedding(self.item_num, 1).to(self.device)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1).to(
            self.device
        )
        self.drop_1 = nn.Dropout(p=0.5).to(self.device)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2).to(self.device)
        self.drop_2 = nn.Dropout(p=0.5).to(self.device)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding_layer.weight)
        nn.init.xavier_uniform_(self.k_difficulty.weight)
        nn.init.xavier_uniform_(self.e_difficulty.weight)
        nn.init.xavier_uniform_(self.prednet_full1.weight)
        nn.init.xavier_uniform_(self.prednet_full2.weight)
        nn.init.xavier_uniform_(self.prednet_full3.weight)

    def predict(self, user_id, item_id, input_knowledge_point, user_embeddings=None):
        if user_embeddings is None:
            user_embeddings = self.user_embedding_layer(user_id)
        stat_emb = torch.sigmoid(user_embeddings)
        k_vector = self.k_difficulty(item_id)
        e_vector = self.e_difficulty(item_id)
        k_difficulty = torch.sigmoid(k_vector)
        e_difficulty = torch.sigmoid(e_vector)
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x)).view(-1)
        out = {"prediction": output_1}
        return out

    def forward(self, user_id, item_id, input_knowledge_point, score, user_embeddings=None):
        out = self.predict(user_id, item_id, input_knowledge_point, user_embeddings)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out

    def get_user_embeddings(self, user_id):
        return self.user_embedding_layer(user_id)