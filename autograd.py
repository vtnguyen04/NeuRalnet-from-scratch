class AutogradContext:

    def __init__(self):
        self.current_model = None

    def set_model(self, model):
        self.current_model = model
    
    def clear(self):
        self.current_model = None 

    def get_model(self):
        if self.current_model is None:
            raise RuntimeError("No models")
        return self.current_model

AutogradContext = AutogradContext()