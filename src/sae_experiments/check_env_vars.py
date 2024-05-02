import os 

def check_env_vars():
    # wandb variables
    print("WANDB_API_KEY:", os.getenv("WANDB_API_KEY"))
    print("WANDB_NAME:", os.getenv("WANDB_NAME"))
    print("WANDB_PROJECT:", os.getenv("WANDB_PROJECT"))
    print("WANDB_ENTITY:", os.getenv("WANDB_ENTITY"))

if __name__ == "__main__":
    check_env_vars()