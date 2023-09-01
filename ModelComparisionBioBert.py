import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

from sklearn.model_selection import ParameterGrid


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL = "dmis-lab/biobert-base-cased-v1.2"

tokenizer = BertTokenizer.from_pretrained(MODEL)
BIO_LABELS = ["PAD", "O", "B-GENOTYPE", "I-GENOTYPE"]

# config = {"lr": tune.loguniform(5e-5, 1e-4), "batch_size": tune.choice([2, 4, 8])}
config = {"lr": [3e-5, 1e-4, 5e-5], "batch_size": [2, 4, 8]}


normal_train_df = pd.read_pickle("../data/bioBert/offset/train.df")
normal_test_df = pd.read_pickle("../data/bioBert/offset/test.df")
normal_val_df = pd.read_pickle("../data/bioBert/offset/val.df")

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

    return f1_score, precision, recall


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
def train_model(train_dataloader, val_dataloader, val_labels, model, optimizer, epochs):
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
        span_f1, pre, recall = calculate_span_f1(val_labels_1, val_preds)
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
            "Precision (Span):",
            pre,
            "Recall (Span):",
            recall,
        )
    return span_f1, pre, recall


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


def findHyperParameters(config):
    train_dataloader = DataLoader(
        normal_train_data, sampler=normal_train_sampler, batch_size=config["batch_size"]
    )
    val_dataloader = DataLoader(
        normal_val_data, sampler=normal_val_sampler, batch_size=config["batch_size"]
    )

    model_normal = BertForTokenClassification.from_pretrained(
        MODEL, num_labels=len(BIO_LABELS)
    )
    optimizer_normal = torch.optim.Adam(model_normal.parameters(), lr=config["lr"])

    f1, pre, recall = train_model(
        train_dataloader,
        val_dataloader,
        normal_val_df_label_ids,
        model_normal,
        optimizer_normal,
        epochs=5,
    )
    return f1, pre, recall


best_params, best_f1 = None, 0
for params in ParameterGrid(config):
    f1, pre, recall = findHyperParameters(params)
    if f1 > best_f1:
        best_params = params
        best_f1 = f1
print(f"{best_params=}")
print(f"{best_f1=:.3f}")


best_trained_model = BertForTokenClassification.from_pretrained(
    MODEL, num_labels=len(BIO_LABELS)
)
best_trained_model.to(device)

train_dataloader = DataLoader(
    normal_train_data,
    sampler=normal_train_sampler,
    batch_size=best_params["batch_size"],
)
val_dataloader = DataLoader(
    normal_val_data,
    sampler=normal_val_sampler,
    batch_size=best_params["batch_size"],
)

optimizer = torch.optim.Adam(best_trained_model.parameters(), lr=best_params["lr"])
print("*************** NORMAL ********************************")
train_model(
    train_dataloader,
    val_dataloader,
    normal_val_df_label_ids,
    best_trained_model,
    optimizer,
    epochs=25,
)

normal_test_df_tokens = normal_test_df["tokens"].tolist()
normal_test_df_labels = normal_test_df["labels"].tolist()

test_loss = []

(
    normal_test_df_input_ids,
    normal_test_df_attention_masks,
    normal_test_df_label_ids,
) = tokens_to_ids(normal_test_df_tokens, normal_test_df_labels)

normal_test_data = TensorDataset(
    normal_test_df_input_ids, normal_test_df_attention_masks, normal_test_df_label_ids
)
normal_test_sampler = SequentialSampler(normal_test_data)


test_dataloader = DataLoader(
    normal_test_data,
    sampler=normal_test_sampler,
    batch_size=best_params["batch_size"],
)

total_test_loss = 0
test_preds = []
for batch in test_dataloader:
    batch_input_ids = batch[0].to(device)
    batch_attention_masks = batch[1].to(device)
    batch_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = best_trained_model(
            batch_input_ids,
            token_type_ids=None,
            attention_mask=batch_attention_masks,
            labels=batch_labels,
        )
    total_test_loss += outputs[0].item()
    logits = outputs[1].detach().cpu().numpy()
    test_preds.extend([list(p) for p in np.argmax(logits, axis=2)])
avg_test_loss = total_test_loss / len(test_dataloader)
test_loss.append(avg_test_loss)
test_preds = [p for pred in test_preds for p in pred]
test_labels_1 = [l for label in normal_test_df_label_ids for l in label]
macro_f1 = f1_score(test_labels_1, test_preds, average="macro")
weight_f1 = f1_score(test_labels_1, test_preds, average="weighted")
span_f1, pre, recall = calculate_span_f1(test_labels_1, test_preds)
print(
    "F1 Score (Span):", span_f1, "Precission (Span):", pre, "Precission (Span):", recall
)
