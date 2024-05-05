import os
import torch
import dotenv

from sae_lens import LanguageModelSAERunnerConfig, language_model_sae_runner

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"
dotenv.load_dotenv(".env")

def get_toy_lm_sae_runner_config(
    seed: int = 0,
    sae_class_name: str = "SparseAutoencoder",
    l1_coefficient: float = 1e-3
):
    return LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="tiny-stories-1L-21M",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_point="blocks.0.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        sae_class_name=sae_class_name,
        hook_point_layer=0,  # Only one layer in the model.
        d_in=1024,  # the width of the mlp output.
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
        is_dataset_tokenized=True,
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method="geometric_median",  # The geometric median can be used to initialize the decoder weights.
        # Training Parameters
        lr=0.0008,  # lower the better, we'll go fairly high to speed up the tutorial.
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=10000,  # this can help avoid too many dead features initially.
        l1_coefficient=l1_coefficient,  # will control how sparse the feature activations are
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size=4096,
        context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower.
        # Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=1_000_000
        * 50,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size=16,
        # Resampling protocol
        use_ghost_grads=False,
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="sae-experiments",
        wandb_log_frequency=10,
        # Misc
        device="cuda",
        seed=seed,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype=torch.float32,
    )

# def get_lm_sae_runner_config():
#     return LanguageModelSAERunnerConfig(
#         # Data Generating Function (Model + Training Distibuion)
#         model_name="gpt2-small",
#         sae_class_name="SparseAutoencoder",
#         hook_point="blocks.2.hook_resid_pre",
#         hook_point_layer=2,
#         d_in=768,
#         dataset_path="Skylion007/openwebtext",
#         is_dataset_tokenized=False,
#         # SAE Parameters
#         expansion_factor=64,
#         b_dec_init_method="geometric_median",
#         # Training Parameters
#         lr=0.0004,
#         l1_coefficient=0.00008,
#         lr_scheduler_name="constant",
#         train_batch_size=4096,
#         context_size=128,
#         lr_warm_up_steps=5000,
#         # Activation Store Parameters
#         n_batches_in_buffer=128,
#         training_tokens=1_000_000 * 300,
#         store_batch_size=32,
#         # Dead Neurons and Sparsity
#         use_ghost_grads=True,
#         feature_sampling_window=1000,
#         dead_feature_window=5000,
#         dead_feature_threshold=1e-6,
#         # WANDB
#         log_to_wandb=True,
#         wandb_project="gpt2",
#         wandb_entity=None,
#         wandb_log_frequency=100,
#         # Misc
#         device="cuda",
#         seed=42,
#         n_checkpoints=10,
#         checkpoint_path="checkpoints",
#         dtype=torch.float32,
#     )


if __name__ == "__main__":

    # from sae_experiments.slack import send_slack_notification

    l1_coefficients = [1e-1, 1, 10]
    # l1_coefficients = [1e-4, 1e-3, 1e-2]
    for seed in range(3):
        for sae_class_name in ["GatedSparseAutoencoder", "SparseAutoencoder"]:
            for l1_coefficient in l1_coefficients:
                cfg = get_toy_lm_sae_runner_config(
                    seed=seed, 
                    sae_class_name=sae_class_name,
                    l1_coefficient = l1_coefficient
                )
                language_model_sae_runner(cfg)

    # parser = simple_parsing.ArgumentParser()
    # parser.add_arguments(
    #     LanguageModelSAERunnerConfig, dest="cfg", default=get_lm_sae_runner_config()
    # )

    # args = parser.parse_args()
    # cfg = args.cfg
    # pprint(cfg)
    
    # TODO: send slack notifications
    # language_model_sae_runner(cfg)