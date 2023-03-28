import torch
import transformers


def predict(input_text):
    device = torch.device("cpu")  

    tokenizer = transformers.BertTokenizer.from_pretrained('final_model')
    model = transformers.BertForSequenceClassification.from_pretrained("final_model")

    model.eval()

    chunks = []
    words = input_text.split()
    for i, word in enumerate(words):
        if word == "there":
            start = max(0, i-1)
            end = min(len(words), i+2)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

    predictions = []
    for chunk in chunks:
        tokens = tokenizer.encode_plus(chunk, max_length=128, truncation=True, padding='max_length',
                                       return_tensors='pt')

        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        probs = torch.softmax(logits[0], dim=-1)
        predicted_label_index = torch.argmax(probs, dim=-1).item()

        if predicted_label_index == 0:
            predictions.append(('their', probs.tolist()[0][predicted_label_index]))
        elif predicted_label_index == 1:
            predictions.append(('there', probs.tolist()[0][predicted_label_index]))
        elif predicted_label_index == 2:
            predictions.append(("they're", probs.tolist()[0][predicted_label_index]))

    return predictions


input_text = input('> ')

predictions = predict(input_text)

for label, prob in predictions:
    print(f"Predicted label: {label}, Probability: {prob:.5f}")
