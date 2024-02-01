from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

from model.CD import BaseCD
from utils import *



def train_model(model: BaseCD, args, train_data, validation_data, test_data, device, save_base_path):
    """
    args is a dict that must contain the following parameters:
    - lr: learning rate
    - batch_size: batch size for training
    - epochs: number of epochs
    - evaluate_every_epoch: how often to evaluate the model on the validation data
    - eval_batch_size: batch size for evaluating the model on the validation data

    """
    print("Training model with parameters: ", args)
    model_optimizer = torch.optim.Adam(model.parameters(), args['lr'])
    best_acc = 0
    for epoch in range(args['epochs']):
        model.train()

        for batch_data in tqdm(train_data, "Epoch %s " % epoch):
            if model_has_knowledge_dimension(args['model']):
                user_id, item_id, knowledge, response, _ = batch_data
                knowledge = knowledge.to(device)
            else:
                user_id, item_id, response, _ = batch_data
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            response = response.to(device)

            # Reset the gradient
            model_optimizer.zero_grad()

            if model_has_knowledge_dimension(args['model']):
                out = model(user_id, item_id, knowledge, response)
            else:
                out = model(user_id, item_id, response)
            loss = out["loss"]
            loss.backward()
            model_optimizer.step()
        print(f"Evaluation at epoch {epoch + 1}/{args['epochs']}")
        acc, _, _, _ = evaluate_model(model, args, validation_data, device)
        if acc > best_acc:
            print("New best accuracy found on validation set!")
            best_acc = acc
            print("Saving best model...")
            model.save_model(save_base_path + f'{args["model"]}.pt')


def evaluate_model(model: BaseCD, args, eval_data, device, filter_model=None):
    model.eval()
    y_pred = []
    y_true = []
    for batch_data in tqdm(eval_data, "Test"):
        if model_has_knowledge_dimension(args['model']):
            user_id, item_id, knowledge, response, _ = batch_data
            knowledge = knowledge.to(device)
        else:
            user_id, item_id, response, _ = batch_data
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        response = response.to(device)

        # If the filter model is present, use it to get the filtered user embeddings
        if filter_model is not None:
            filtered_user_embeddings = filter_model(model.get_user_embeddings(user_id))
        else:
            filtered_user_embeddings = None

        if model_has_knowledge_dimension(args['model']):
            out = model.predict(user_id, item_id, knowledge, filtered_user_embeddings)
        else:
            out = model.predict(user_id, item_id, filtered_user_embeddings)
        y_pred.extend(out["prediction"].tolist())
        y_true.extend(response.tolist())
    acc = accuracy_score(y_true, np.array(y_pred) > 0.5)
    roc_auc = roc_auc_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print("acc:{:.4f}".format(acc))
    print("auc:{:.4f}".format(roc_auc))
    print("mae:{:.4f}".format(mae))
    print("mse:{:.4f}".format(mse))
    return acc, roc_auc, mae, mse
