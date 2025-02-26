from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

class ProcessText4Classification:
    def __init__(self, max_length=128, device="mps", model="tfid", model_ckpt="bert-base-uncased"):
        self.max_length = max_length
        self.device = device
        self.model = model
        self.model_ckpt = model_ckpt
        
        if self.model == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        else:
            self.tokenizer = None

    def _tokenizer_and_vectorization(self, text):
        if self.model == "tfid":
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
            return vectorizer.fit_transform(text)
        
        elif self.model == "bert":
            if not self.tokenizer:
                raise ValueError("The BERT tokenizer is not initiliazed")
            
            encoded_inputs = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            return encoded_inputs
        
        else:
            raise ValueError("Wrong model choose TFIDF or BERT")