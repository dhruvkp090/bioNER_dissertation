import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
import os

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
BIO_LABELS = ["PAD", "O", "B-GENOTYPE", "I-GENOTYPE"]

BASE_PATH = "/home/msc23dhruv/msc23dhruvvol1claim/models/"

# config = {"lr": tune.loguniform(5e-5, 1e-4), "batch_size": tune.choice([2, 4, 8])}
config = {"lr": tune.choice([3e-5]), "batch_size": tune.choice([8])}


scheduler = ASHAScheduler(
    metric="loss", mode="min", max_t=10, grace_period=1, reduction_factor=2
)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "f1_span", "training_iteration"]
)


normal_train_df = pd.read_pickle("data/bert/normal/train.df")
normal_test_df = pd.read_pickle("data/bert/normal/test.df")
normal_val_df = pd.read_pickle("data/bert/normal/val.df")

normal_train_df_tokens = normal_train_df["tokens"].tolist()
normal_train_df_labels = normal_train_df["labels"].tolist()

normal_val_df_tokens = normal_val_df["tokens"].tolist()
normal_val_df_labels = normal_val_df["labels"].tolist()


def calculate_span_f1(true_labels, predicted_labels):
    true_spans = get_spans(true_labels)
    predicted_spans = get_spans(predicted_labels)
    true_entities = set(true_spans)
    predicted_entities = set(predicted_spans)

    true_positive = len(true_entities.intersection(predicted_entities))
    false_positive = len(predicted_entities - true_entities)
    false_negative = len(true_entities - predicted_entities)

    precision, recall, f1_score = calculate_precision_recall_f1(
        true_positive, false_positive, false_negative
    )

    return f1_score


def get_spans(labels):
    spans = []
    start = None
    for i, label in enumerate(labels):
        if label == 2:
            if start is not None:
                spans.append((start, i))
            start = i
        elif label == 3:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


# Convert the tokens and labels into the format required by the BERT model
def tokens_to_ids(tokens, labels):
    input_ids = []
    attention_masks = []
    label_ids = []
    for token_list, label_list in zip(tokens, labels):
        input_ids.append(tokenizer.convert_tokens_to_ids(token_list))
        attention_mask = [1] * len(token_list)
        attention_masks.append(attention_mask)
        label_id = [BIO_LABELS.index(label) for label in label_list]
        label_ids.append(label_id)

    input_ids = pad_sequence(
        [torch.tensor(seq) for seq in input_ids], batch_first=True, padding_value=0.0
    ).tolist()
    attention_masks = pad_sequence(
        [torch.tensor(seq) for seq in attention_masks],
        batch_first=True,
        padding_value=0.0,
    ).tolist()
    label_ids = pad_sequence(
        [torch.tensor(seq) for seq in label_ids], batch_first=True, padding_value=0.0
    ).tolist()

    # Convert the data into PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    return input_ids, attention_masks, label_ids


# Train the model
def train_model(
    train_dataloader,
    val_dataloader,
    val_labels,
    model,
    optimizer,
    epochs,
    checkpoint_dir,
):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            batch_labels = batch_labels.view(
                -1, batch_labels.size(1)
            )  # Reshape labels tensor
            optimizer.zero_grad()
            model.to(device)
            outputs = model(
                batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_masks,
                labels=batch_labels,
            )
            total_train_loss += outputs[0].item()
            outputs[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss.append(avg_train_loss)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        model.eval()
        total_val_loss = 0
        val_preds = []
        for batch in val_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_masks,
                    labels=batch_labels,
                )
            total_val_loss += outputs[0].item()
            logits = outputs[1].detach().cpu().numpy()
            val_preds.extend([list(p) for p in np.argmax(logits, axis=2)])
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_loss.append(avg_val_loss)
        val_preds = [p for pred in val_preds for p in pred]
        val_labels_1 = [l for label in val_labels for l in label]
        macro_f1 = f1_score(val_labels_1, val_preds, average="macro")
        weight_f1 = f1_score(val_labels_1, val_preds, average="weighted")
        span_f1 = calculate_span_f1(val_labels_1, val_preds)
        tune.report(loss=avg_val_loss, f1_span=span_f1)
        print(
            "Epoch:",
            epoch + 1,
            "Train Loss:",
            avg_train_loss,
            "Val Loss:",
            avg_val_loss,
            "F1 Score (Macro):",
            macro_f1,
            "F1 Score (Weighted):",
            weight_f1,
            "F1 Score (Span):",
            span_f1,
        )


(
    normal_train_df_input_ids,
    normal_train_df_attention_masks,
    normal_train_df_label_ids,
) = tokens_to_ids(normal_train_df_tokens, normal_train_df_labels)
(
    normal_val_df_input_ids,
    normal_val_df_attention_masks,
    normal_val_df_label_ids,
) = tokens_to_ids(normal_val_df_tokens, normal_val_df_labels)

normal_train_data = TensorDataset(
    normal_train_df_input_ids,
    normal_train_df_attention_masks,
    normal_train_df_label_ids,
)
normal_train_sampler = RandomSampler(normal_train_data)


normal_val_data = TensorDataset(
    normal_val_df_input_ids, normal_val_df_attention_masks, normal_val_df_label_ids
)
normal_val_sampler = SequentialSampler(normal_val_data)


def findHyperParameters(config, checkpoint_dir=None):
    train_dataloader = DataLoader(
        normal_train_data, sampler=normal_train_sampler, batch_size=config["batch_size"]
    )
    val_dataloader = DataLoader(
        normal_val_data, sampler=normal_val_sampler, batch_size=config["batch_size"]
    )

    model_normal = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(BIO_LABELS)
    )
    optimizer_normal = torch.optim.Adam(model_normal.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model_normal.load_state_dict(model_state)
        optimizer_normal.load_state_dict(optimizer_state)

    train_model(
        train_dataloader,
        val_dataloader,
        normal_val_df_label_ids,
        model_normal,
        optimizer_normal,
        epochs=5,
        checkpoint_dir=checkpoint_dir,
    )


result = tune.run(
    partial(findHyperParameters),
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
print(
    "Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]
    )
)

best_trained_model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(BIO_LABELS)
)
best_trained_model.to(device)

best_checkpoint_dir = best_trial.checkpoint.dir_or_data
model_state, optimizer_state = torch.load(
    os.path.join(best_checkpoint_dir, "checkpoint")
)
best_trained_model.load_state_dict(model_state)


train_dataloader = DataLoader(
    normal_train_data,
    sampler=normal_train_sampler,
    batch_size=best_trial.config["batch_size"],
)
val_dataloader = DataLoader(
    normal_val_data,
    sampler=normal_val_sampler,
    batch_size=best_trial.config["batch_size"],
)

optimizer = torch.optim.Adam(
    best_trained_model.parameters(), lr=best_trial.config["lr"]
)
optimizer.load_state_dict(optimizer_state)

print("*************** NORMAL ********************************")
train_model(
    train_dataloader,
    val_dataloader,
    normal_val_df_label_ids,
    best_trained_model,
    optimizer,
    epochs=5,
    checkpoint_dir="test",
)
