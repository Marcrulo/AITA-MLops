
class LoadModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = self.load_model()