from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # This model is tiny, lightning-fast, and fits easily inside Render's 512MB limit!

class RobertaEmbedder:
    def __init__(self, model_name="roberta-base"):
        print(f"Loading {model_name} from Hugging Face...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding(self, text: str):
        # Tokenize input text
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Perform pooling to get sentence-level embedding
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.squeeze().tolist()
