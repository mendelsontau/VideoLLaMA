# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:

# Need to call this before importing transformers.



import sys
sys.path.insert(0, "/home/gamir/DER-Roei/alon/LLaVA")

#from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

#replace_llama_attn_with_flash_attn()

from llava.train.train_video import train
from slowfast.slowfast.config.defaults import assert_and_infer_cfg
from slowfast.slowfast.utils.misc import launch_job
from slowfast.slowfast.utils.parser import load_config, parse_args
from argparse import Namespace


if __name__ == "__main__":
    mvit_args = Namespace(
        shard_id = 0,
        num_shards = 1,
        init_method = "tcp://localhost:9999",
        cfg = ["slowfast/configs/SSv2/MVITv2_S_16x4.yaml"],
        opts = None
    )
    path_to_mvit_config = "slowfast/configs/SSv2/MVITv2_S_16x4.yaml"
    mvit_cfg = load_config(mvit_args,mvit_args.cfg[0])
    mvit_cfg = assert_and_infer_cfg(mvit_cfg)
    train(mvit_cfg)
