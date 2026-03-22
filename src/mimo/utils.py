from dataclasses import dataclass


@dataclass
class ChannelConfig:
    cdl_model: str = "C"
    num_rx: int = 16
    num_tx: int = 64
