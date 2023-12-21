import transformers

class ModelClass:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model_name = 'bert-base-uncased'


        if self.checkpoint is not None:
            self.model = transformers.BertForSequenceClassification.from_pretrained(self.checkpoint, num_labels=4, problem_type='multi_label_classification').to('cuda')
        else:
            self.model = transformers.BertForSequenceClassification.from_pretrained(self.model_name, num_labels=4, problem_type='multi_label_classification').to('cuda')

        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name, problem_type='multi_label_classification')
        
    def load_tokenizer(self):
        return self.tokenizer
    
    def load_model(self):
        return self.model