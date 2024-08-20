import nncf
import openvino as ov
import gc

from flux_helper import (
    TRANSFORMER_PATH,
    VAE_DECODER_PATH,
    TEXT_ENCODER_PATH,
    TEXT_ENCODER_2_PATH,
)

model_dir = r"F:\om\2024\LLM\converter\FLUX.1-schnell"
model_dict = {
    "transformer": model_dir / TRANSFORMER_PATH,
    "text_encoder": model_dir / TEXT_ENCODER_PATH,
    "text_encoder_2": model_dir / TEXT_ENCODER_2_PATH,
    "vae": model_dir / VAE_DECODER_PATH,
}
compression_args = {
    "mode": nncf.CompressWeightsMode.INT4_SYM,
    "group_size": 64,
    "ratio": 1.0,
}

int4_model_dict = {}

if True:
    core = ov.Core()

    for model_name, model_path in model_dict.items():
        int4_path = model_path.parent / (model_path.stem + "_int4.xml")
        if not int4_path.exists():
            print(f"⌛ {model_path.stem} compression started")
            print(
                f"Compression parameters:\n\tmode = {compression_args['mode']}\n\tratio = {compression_args['ratio']}\n\tgroup_size = {compression_args['group_size']}"
            )
            model = core.read_model(model_path)
            compressed_model = nncf.compress_weights(model, **compression_args)
            ov.save_model(compressed_model, int4_path)
            print(f"✅ {model_path.stem} compression finished")
            del compressed_model
            del model
            gc.collect()
        print(f"Compressed {model_path.stem} can be found in {int4_path}")
        int4_model_dict[model_name] = int4_path
