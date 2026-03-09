from models.cnn_rnn_model import CNNRNNModel
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel


def get_model(cfg, num_classes):
    model_type = str(getattr(cfg, "model_type", "rnn")).lower()

    if model_type == "rnn":
        return RNNModel(cfg, num_classes=num_classes)
    if model_type == "cnn_rnn":
        return CNNRNNModel(cfg, num_classes=num_classes)
    if model_type == "transformer":
        return TransformerModel(cfg, num_classes=num_classes)

    raise ValueError(
        f"Unsupported model_type={model_type}. "
        "Choose from: rnn, cnn_rnn, transformer."
    )
