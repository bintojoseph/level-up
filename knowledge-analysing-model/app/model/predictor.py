from .model_loader import load_model_and_tokenizer
import torch

id2topic = {
    0: "AI/ML",
    1: "Blockchain",
    2: "Cybersecurity",
    3: "Cloud Computing",
    4: "IoT",
    5: "Web Development",
    6: "Mobile App Development",
    7: "Data Science",
    8: "AR/VR",
    9: "Quantum Computing"
}

model, tokenizer = load_model_and_tokenizer()

def predict_topic_and_level(query: str):
    device = model.device
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

    with torch.no_grad():
        topic_logits, level_logits = model(inputs['input_ids'], inputs['attention_mask'])
        topic_id = torch.argmax(topic_logits, dim=1).item()
        knowledge_level = torch.argmax(level_logits, dim=1).item()

    return id2topic[topic_id], knowledge_level
