# %%
# embedding matice má v řádcích embeddingy (1 řádek - 1 token)
# vizualizovat umapem a olabelovat páry

# %%
checkpoint_path = "/mnt/raid/data/Hyner_Petr/rl/rl_basic_transformer/data/cc/0wnzhmdt/checkpoints/epoch=17-step=26100.ckpt" # #2l
#checkpoint_path = "/mnt/raid/data/Hyner_Petr/rl/rl_basic_transformer/data/cc/aqmcef34/checkpoints/epoch=4-step=7500.ckpt" # 2l

# %%
import wandb
from tqdm import tqdm

# %%
import torch
from lightning import LightningModule
from models import GPT, GPTConfig, CausalSelfAttention
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class NanoGPT(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = None,
        n_head: int = None,
        n_embd: int = None,
        dropout: float = 0.0,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: tuple = (0.9, 0.95)
    ):
        super().__init__()
        self.betas = betas
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.gpt = GPT(self.config)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        return self.gpt(idx, targets)

    def get_attention_map(self, idx: torch.Tensor):
        """
        Get the attention map for a single example.
        """
        self.attention_map = None

        def hook_fn(module, input, output):
            self.attention_map = output[1].detach() 

        first_attn_layer = self.gpt.transformer.h[1].attn
        handle = first_attn_layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            self(idx)

        # Remove the hook
        handle.remove()

        return self.attention_map
## FCE:

def load_trained_model(checkpoint_path, vocab_size, block_size, n_layer, n_head, n_embd):
    model = NanoGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model
    
def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

# %%
loaded_model = load_trained_model(
    checkpoint_path,
    vocab_size=92,
    block_size=32,
    n_layer=2,
    n_head=1,
    n_embd=16,
)
print("Model loaded from local checkpoint")

input_ids = torch.randint(0, 80, (1, 8))  # Random input
with torch.no_grad():
    output, _ = loaded_model(input_ids)
print("Output shape:", output.shape)

# %%
data = load_data("new_data.pkl")
test_data = data["test"]

# %%
def inference_on_test_set(model, test_data):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = len(test_data)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for sample in tqdm(test_data):
            input_ids = torch.tensor(sample['input']).unsqueeze(0)  # Add batch dimension
            true_output = sample['target_idx']
            
            input_ids = input_ids.to(next(model.parameters()).device)
            
            logits, _ = model(input_ids)
            predicted_output = logits.argmax(dim=-1).item()
            
            correct_predictions += (predicted_output == true_output)
            
            predictions.append(predicted_output)
            true_labels.append(true_output)
    
    accuracy = correct_predictions / total_samples
    return accuracy, predictions, true_labels

# %%
accuracy,_,_= inference_on_test_set(loaded_model, test_data)
accuracy

# %%
def inference_on_single_sample(model, sample):
    model.eval()  # Set the model to evaluation mode
    
    input_ids = torch.tensor(sample['input']).unsqueeze(0)  # Add batch dimension
    true_output = sample['target_idx']
    
    input_ids = input_ids.to(next(model.parameters()).device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        predicted_output = logits.argmax(dim=-1).item()
    
    is_correct = (predicted_output == true_output)
    
    return is_correct, f"{predicted_output=}", f"{true_output=}"


# %%
loaded_model.gpt.transformer.wte.weight.shape

# %%
loaded_model.gpt.transformer.wte.weight

# %%
loaded_model.gpt.transformer.h[0].attn

# %%
loaded_model

# %%
index = 10

# %%
test_data[index]["input"]

# %%
test_data[index]["orig_idx"]

# %%
inference_on_single_sample(loaded_model, test_data[index])

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def visualize_attention_map(model, input_ids):
    model.eval()
    with torch.no_grad():
        attention = model.get_attention_map(input_ids)
    print(attention.shape)
    attention = attention.squeeze()
    print(attention.shape)
    print(attention)

    plt.figure(figsize=(8, 8))
    sns.heatmap(attention, cmap='viridis')
    plt.title("Attention Map")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

input_ids = torch.tensor(test_data[index]["input"]).unsqueeze(0)
visualize_attention_map(loaded_model, input_ids)

# %%
embeddings = loaded_model.gpt.transformer.wte.weight
embeddings.shape

# %%
import umap
from sklearn.decomposition import PCA
embeddings = loaded_model.gpt.transformer.wte.weight
embeddings = embeddings.detach().numpy()[:73]

def get_embedding_umap(embeddings):
    reducer = umap.UMAP()
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def get_embedding_pca(embeddings):
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings

# Call the functions to get UMAP and PCA embeddings
umap_embeddings = get_embedding_umap(embeddings)
pca_embeddings = get_embedding_pca(embeddings)

# Print the shapes of the embeddings
print(f"UMAP embeddings shape: {umap_embeddings.shape}")
print(f"PCA embeddings shape: {pca_embeddings.shape}")


# %%
embeddings

# %%
embeddings.shape

# %%
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, title):
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.title(title)
    labels = list(range(len(embeddings)))
    for i in range(len(embeddings)):
        plt.annotate(labels[i], (embeddings[i, 0], embeddings[i, 1]), textcoords="offset points", xytext=(5,5), ha='center')
    plt.show()

# Call the function to visualize the PCA embeddings
visualize_embeddings(pca_embeddings, "PCA Embeddings")
visualize_embeddings(umap_embeddings, "UMAP Embeddings")

# %%


# %%


# %%



