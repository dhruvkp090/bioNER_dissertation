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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
BIO_LABELS = ["PAD", "O", "B-GENOTYPE", "I-GENOTYPE"]
# # Initialize the BERT model and optimizer

# model_offset = BertForTokenClassification.from_pretrained(
#     "bert-base-uncased", num_labels=len(BIO_LABELS)
# )
# optimizer_offset = torch.optim.Adam(model_offset.parameters(), lr=3e-5)
# epochs = 10
# BATCH_SIZE = 8

param_grid = {"batch_size": [2, 4, 8], "lr": [1e-5, 3e-5, 5e-5]}

offset_train_df = pd.read_pickle("data/bert/offset/train.df")
offset_test_df = pd.read_pickle("data/bert/offset/test.df")
offset_val_df = pd.read_pickle("data/bert/offset/val.df")


offset_train_df_tokens = offset_train_df["tokens"].tolist()
offset_train_df_labels = offset_train_df["labels"].tolist()

offset_val_df_tokens = offset_val_df["tokens"].tolist()
offset_val_df_labels = offset_val_df["labels"].tolist()


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

    # Pad the sequences to the same length
    max_length = 512

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
def train_model(train_dataloader, val_dataloader, val_labels, model, optimizer):
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
            label_ids = batch_labels.to("cpu").numpy()
            val_preds.extend([list(p) for p in np.argmax(logits, axis=2)])
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_loss.append(avg_val_loss)
        val_preds = [p for pred in val_preds for p in pred]
        val_labels_1 = [l for label in val_labels for l in label]
        macro_f1 = f1_score(val_labels_1, val_preds, average="macro")
        weight_f1 = f1_score(val_labels_1, val_preds, average="weighted")
        span_f1 = calculate_span_f1(val_labels_1, val_preds)
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
    offset_train_df_input_ids,
    offset_train_df_attention_masks,
    offset_train_df_label_ids,
) = tokens_to_ids(offset_train_df_tokens, offset_train_df_labels)
(
    offset_val_df_input_ids,
    offset_val_df_attention_masks,
    offset_val_df_label_ids,
) = tokens_to_ids(offset_val_df_tokens, offset_val_df_labels)

offset_train_data = TensorDataset(
    offset_train_df_input_ids,
    offset_train_df_attention_masks,
    offset_train_df_label_ids,
)
offset_train_sampler = RandomSampler(offset_train_data)
offset_train_dataloader = DataLoader(
    offset_train_data, sampler=offset_train_sampler, batch_size=BATCH_SIZE
)

offset_val_data = TensorDataset(
    offset_val_df_input_ids, offset_val_df_attention_masks, offset_val_df_label_ids
)
offset_val_sampler = SequentialSampler(offset_val_data)
offset_val_dataloader = DataLoader(
    offset_val_data, sampler=offset_val_sampler, batch_size=BATCH_SIZE
)

print("*************** OFFSET ********************************")
train_model(
    offset_train_dataloader,
    offset_val_dataloader,
    offset_val_df_label_ids,
    model_offset,
    optimizer_offset,
)
