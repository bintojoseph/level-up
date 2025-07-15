import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig

class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.topic_classifier = torch.nn.Linear(config.hidden_size, 10)  # 10 topics
        self.level_classifier = torch.nn.Linear(config.hidden_size, 6)   # Levels 0 to 5

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        topic_logits = self.topic_classifier(pooled_output)
        level_logits = self.level_classifier(pooled_output)
        return topic_logits, level_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = MultiTaskBERT(config)
    model.load_state_dict(torch.load("multi_task_bert.pth", map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer
