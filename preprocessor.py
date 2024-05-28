import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 

class GraphPreprocessor:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.embeddings = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).eval().bert.embeddings
        self.dep_tree_creator = spacy.load("en_core_web_sm")
        
    def preprocess(self, text):
        document = self.dep_tree_creator(text)
        context = {}
        jt = 0
        features = []
        for token in document:
            inputs = self.tokenizer(token.text, return_tensors='pt', truncation=False)
            with torch.no_grad():
                embedding = self.embeddings(inputs['input_ids']).sum(axis=1)
            if token.text not in context:
                context[token.text] = (jt, embedding)
                features.append(embedding.squeeze(0))
                jt += 1
                
        edges = []
        for token in document:
            for child in token.children:
                edges.append(torch.tensor([context[token.text][0], context[child.text][0]]))
        return {'x': torch.stack(features), 'edges': torch.stack(edges).T}