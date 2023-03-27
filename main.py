import torch
import transformers


def predict(input_text):
    device = torch.device("cpu")  # or cuda if you really want... not too necessary unless processing a ton of text

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertForSequenceClassification.from_pretrained("their_there_they're")

    model.eval()

    tokens = tokenizer.encode_plus(input_text, max_length=128, truncation=True, padding='max_length',
                                   return_tensors='pt')

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits[0], dim=-1)

    predicted_label_index = torch.argmax(probs, dim=-1).item()

    if predicted_label_index == 0:
        return 'their', probs
    elif predicted_label_index == 1:
        return 'there', probs
    elif predicted_label_index == 2:
        return "they're", probs


# Print the predicted label
print(f"Predicted label: {predict(input_text=input('> '))}")
