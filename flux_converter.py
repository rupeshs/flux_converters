from flux_helper import convert_flux

# set HF_HOME=F:\om\2024\LLM\converter\flux16model

model_dir = convert_flux("black-forest-labs/FLUX.1-schnell")

from flux_helper import (
    TRANSFORMER_PATH,
    VAE_DECODER_PATH,
    TEXT_ENCODER_PATH,
    TEXT_ENCODER_2_PATH,
)

model_dict = {
    "transformer": model_dir / TRANSFORMER_PATH,
    "text_encoder": model_dir / TEXT_ENCODER_PATH,
    "text_encoder_2": model_dir / TEXT_ENCODER_2_PATH,
    "vae": model_dir / VAE_DECODER_PATH,
}
