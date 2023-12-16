from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np


excel_file_path = 'tape.xlsx'
model_name = "microsoft/deberta-base"
tokenizer = DebertaTokenizer.from_pretrained(model_name)
config = DebertaConfig.from_pretrained("microsoft/deberta-base")
config.output_hidden_states = True
model = DebertaForSequenceClassification.from_pretrained(model_name, config=config).to("cuda")

df = pd.read_excel(excel_file_path)
sentence = df.iloc[:, 0].tolist()
reasoning = df.iloc[:, 1].tolist()
pseudo_labels = df.iloc[:, 2].tolist()


a = [1]*100
b= [0]*88
labels = [] 
labels.extend(a)
labels.extend(b)

train_index = []
for i in range(187):
    train_index.append(i)

test_index = random.sample(range(187),50)
train_index = [x for x in train_index if x not in test_index]

train_sentence = []
train_pseudo = []
test_pseudo = []
train_reasoning = []
test_sentence = []
test_reasoning = []
test_label = []
train_label = []
for i in train_index:
    train_sentence.append(sentence[i])
    train_reasoning.append(reasoning[i])
    train_pseudo.append(pseudo_labels[i])
    train_label.append(labels[i])
for i in test_index:
    test_sentence.append(sentence[i])
    test_reasoning.append(reasoning[i])
    test_pseudo.append(pseudo_labels[i])
    test_label.append(pseudo_labels[i])

inputs_train_sent = tokenizer(train_sentence, return_tensors="pt", padding=True, truncation=True)
inputs_train_reason = tokenizer(train_reasoning, return_tensors="pt", padding=True, truncation=True)
inputs_test_sent = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True)
inputs_test_reason = tokenizer(test_reasoning, return_tensors="pt", padding=True, truncation=True)

train_dataset = TensorDataset(inputs_train_sent["input_ids"], inputs_train_sent["attention_mask"],inputs_train_reason["input_ids"], inputs_train_reason["attention_mask"], torch.tensor(train_pseudo), torch.tensor(train_index))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = TensorDataset(inputs_test_sent["input_ids"], inputs_test_sent["attention_mask"],inputs_test_reason["input_ids"], inputs_test_reason["attention_mask"], torch.tensor(test_pseudo), torch.tensor(test_index))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

train_embeddings_sent = {}
train_embeddings_reason = {}

test_embeddings_sent = {}
test_embeddings_reason = {}

# Fine-tune loop
for epoch in range(10):
    print(epoch)
    for batch in train_loader:
        optimizer.zero_grad()
        sent_input_ids, sent_attention_mask, reason_input_ids,reason_attention_mask , label, index = batch
        outputs = model(sent_input_ids.to("cuda"), attention_mask=sent_attention_mask.to("cuda"), labels=label.to("cuda"))
        loss = outputs.loss
        print(loss)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        outputs = model(reason_input_ids.to("cuda"), attention_mask=reason_attention_mask.to("cuda"), labels=label.to("cuda"))
        loss = outputs.loss
        print(loss)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for epoch in range(1):
        print(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            sent_input_ids, sent_attention_mask, reason_input_ids,reason_attention_mask , label, index = batch
            outputs = model(sent_input_ids.to("cuda"), attention_mask=sent_attention_mask.to("cuda"), labels=label.to("cuda"))
            loss = outputs.loss
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = sent_attention_mask.to("cuda")
            pooled_output = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
            j=0
            for i in index:
                train_embeddings_sent[i.item()] = pooled_output[j].squeeze(0)
                j+=1 
            print(loss)
            outputs = model(reason_input_ids.to("cuda"), attention_mask=reason_attention_mask.to("cuda"), labels=label.to("cuda"))
            loss = outputs.loss
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = reason_attention_mask.to("cuda")
            pooled_output = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
            j=0
            for i in index:
                train_embeddings_reason[i.item()] = pooled_output[j].squeeze(0)
                j+=1
            print(loss)

    for epoch in range(1):
        print(epoch)
        for batch in test_loader:
            optimizer.zero_grad()
            sent_input_ids, sent_attention_mask, reason_input_ids,reason_attention_mask , label, index = batch
            outputs = model(sent_input_ids.to("cuda"), attention_mask=sent_attention_mask.to("cuda"), labels=label.to("cuda"))
            loss = outputs.loss
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = sent_attention_mask.to("cuda")
            pooled_output = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
            j=0
            for i in index:
                test_embeddings_sent[i.item()] = pooled_output[j].squeeze(0)
                j+=1
            print(loss)
            outputs = model(reason_input_ids.to("cuda"), attention_mask=reason_attention_mask.to("cuda"), labels=label.to("cuda"))
            loss = outputs.loss
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = reason_attention_mask.to("cuda")
            pooled_output = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
            j=0
            for i in index:
                test_embeddings_reason[i.item()] = pooled_output[j].squeeze(0)
                j+=1
            print(loss)

X_train = []
y_train = []
X_test = []
y_test = []
for i in train_index:
    feat = train_embeddings_sent[i]
    feat = torch.cat((feat, train_embeddings_reason[i]))
    if pseudo_labels[i] == 0:
        feat = torch.cat((feat, torch.zeros(8).to("cuda")))
    else:
        feat = torch.cat((feat, torch.ones(8).to("cuda")))
    X_train.append(feat)
    y_train.append(labels[i])

X_train = torch.stack(X_train).cpu().numpy()
y_train = np.array(y_train)

for i in test_index:
    feat = test_embeddings_sent[i]
    feat = torch.cat((feat, test_embeddings_reason[i]))
    if pseudo_labels[i] == 0:
        feat = torch.cat((feat, torch.zeros(8).to("cuda")))
    else:
        feat = torch.cat((feat, torch.ones(8).to("cuda")))
    X_test.append(feat)
    y_test.append(labels[i])

X_test = torch.stack(X_test).cpu().numpy()
y_test = np.array(y_test)


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train the MLP model
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

"""
sent = ""
reson = ""
predict = 0

sent = tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
outputs = model(reason_input_ids.to("cuda"), attention_mask=reason_attention_mask.to("cuda"), labels=label.to("cuda"))
last_hidden_states = outputs.hidden_states[-1]
attention_mask = reason_attention_mask.to("cuda")
pooled_output = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
"""