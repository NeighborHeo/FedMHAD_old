# %%
import unittest
import random
import torch
import numpy as np

def set_seed(seed):
    """
    Set the random seed for reproducibility
    Args:
        seed (int): the value of the random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_random_client_logits(n_clients, batch_size, n_classes):
    # client_logits = torch.sigmoid(torch.randn(n_clients, batch_size, n_classes))
    return torch.rand(n_clients, batch_size, n_classes)

def make_random_class_counts(n_clients, n_classes, min=0, max=100):
    return torch.randint(min, max, (n_clients, n_classes))

def make_random_client_attentions(n_clients, batch_size, n_heads, img_h, img_w):
    return torch.rand(n_clients, batch_size, n_heads, img_h, img_w)


def compute_class_weights(class_counts):
    """
    Args:
        class_counts (torch.Tensor): (num_samples, num_classes)
    Returns:
        class_weights (torch.Tensor): (num_samples, num_classes)
    """
    # Normalize the class counts per sample
    class_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
    return class_weights

def compute_ensemble_logits(client_logits, class_weights):
    """
    Args:
        client_logits (torch.Tensor): (num_samples, batch_size, num_classes)
        class_weights (torch.Tensor): (num_samples, num_classes)
    Returns:
        ensemble_logits (torch.Tensor): (batch_size, num_classes)
    """
    weighted_logits = client_logits * class_weights.unsqueeze(1)  # (num_samples, batch_size, num_classes)
    sum_weighted_logits = torch.sum(weighted_logits, dim=0)  # (batch_size, num_classes)
    sum_weights = torch.sum(class_weights, dim=0)  # (num_classes)
    ensemble_logits = sum_weighted_logits / sum_weights
    return ensemble_logits

def compute_euclidean_distance(vector_a, vector_b):
    return 1 - torch.sqrt(torch.sum((vector_a - vector_b) ** 2, dim=-1))

def compute_cosine_similarity(vector_a, vector_b):
    return 1 - torch.sum(vector_a * vector_b, dim=-1) / (torch.norm(vector_a, dim=-1) * torch.norm(vector_b, dim=-1))

def calculate_normalized_similarity_weights(target_vectors, client_vectors, similarity_method='euclidean'):
    if similarity_method == 'euclidean':
        similarity_function = compute_euclidean_distance
    elif similarity_method == 'cosine':
        similarity_function = compute_cosine_similarity
    else:
        raise ValueError("Invalid similarity method. Choose 'euclidean' or 'cosine'.")

    target_vectors_expanded = target_vectors.unsqueeze(0)  # Shape: (1, batch_size, n_class)
    similarities = similarity_function(target_vectors_expanded, client_vectors)  # Shape: (n_client, batch_size)
    mean_similarities = torch.mean(similarities, dim=1)  # Shape: (n_client)
    normalized_similarity_weights = mean_similarities / torch.sum(mean_similarities)  # Shape: (n_client)
    return normalized_similarity_weights

def example_of_calculate_normalized_similarity_weights():
    # Example usage
    target = torch.tensor([[1, 1, 1], [2, 3, 4]])
    clients = torch.tensor([[[1, 1, 1],[2, 3, 4]],[[0, 2, 4],[1, 3, 5]],[[2, 1, 0],[1, 2, 3]]])
    print("Target.shape:", target.shape, "Clients.shape:", clients.shape)
    # Using Euclidean distance
    euclidean_normalized_weights = calculate_normalized_similarity_weights(target, clients, similarity_method='euclidean')
    print("Normalized Weights with Euclidean Similarities:", euclidean_normalized_weights)

    # Using Cosine similarity
    cosine_normalized_weights = calculate_normalized_similarity_weights(target, clients, similarity_method='cosine')
    print("Normalized Weights with Cosine Similarities:", cosine_normalized_weights)

def compute_weighted_loss(client_losses, client_weights):
    """
    Args:
        client_losses (torch.Tensor): (num_clients)
        client_weights (torch.Tensor): (num_clients)
    Returns:
        client_weighted_loss (torch.Tensor): (1)
    """
    print('Client Losses shape : ', client_losses.shape)
    print('Client Weights shape : ', client_weights.shape)
    if client_weights is None:
        client_weighted_loss = torch.mean(client_losses)
    else:    
        weighted_losses = client_losses * client_weights
        client_weighted_loss = torch.sum(weighted_losses)
        
    return client_weighted_loss

def compute_attention_similarity_losses(client_attentions, central_attention):
    """
    Args:
        client_attentions (torch.Tensor): (num_clients, batch_size, num_heads, img_height, img_width)
        central_attention (torch.Tensor): (batch_size, num_heads, img_height, img_width)
    Returns:
        similarity_losses (torch.Tensor): (num_clients)
    """
    similarity_losses = []
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for i in range(client_attentions.shape[0]):
        attention_loss = 1 - cosine_similarity(client_attentions[i].view(-1), central_attention.view(-1))
        print('Attention Loss : ', attention_loss)
        similarity_losses.append(attention_loss)

    return torch.stack(similarity_losses)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    set_seed(0)
    
    # define parameters
    n_clients = 2 # 5
    n_classes = 5 # 20
    batch_size = 10 # 32
    n_heads = 3
    img_h = 224
    img_w = 224
    
    class_counts = make_random_class_counts(n_clients, n_classes)
    client_logits = make_random_client_logits(n_clients, batch_size, n_classes)
    client_attentions = make_random_client_attentions(n_clients, batch_size, n_heads, img_h, img_w)
    central_attentions = torch.randn(batch_size, n_heads, img_h, img_w)
    
    # example_of_calculate_normalized_similarity_weights()

    sum_counts = class_counts.sum()
    localweight = 1.0*class_counts/sum_counts
    localweight = localweight.unsqueeze(dim=1).unsqueeze(dim=2)#nlocal*1*1

    class_weights = compute_class_weights(class_counts)
    # print('class_weights[:,0] : ', class_weights[:,0])
    # print('client_logits[:,0,0] : ', client_logits[:,0,0])
    
    # (n, b, c) * (n, c) -> (b, c)
    ensemble_logits = compute_ensemble_logits(client_logits, class_weights)
    print('ensemble_logits : ', ensemble_logits.shape)
    print('ensemble_logits[0,0] : ', ensemble_logits[0,0])

    # (b, c) * (n, b, c) -> (n, b) -> (n)
    sim_weights = calculate_normalized_similarity_weights(ensemble_logits, client_logits, similarity_method='cosine')
    print('sim_weights : ', sim_weights.shape)
    print('sim_weights : ', sim_weights)
    
    # print('client_attentions : ', client_attentions.shape)
    # print('client_attentions[:,0,0,0,0] : ', client_attentions[:,0,0,0,0])
   
    losses = compute_attention_similarity_losses(client_attentions, central_attentions) 
    client_loss = compute_weighted_loss(losses, sim_weights)
    print('client_loss : ', client_loss)
    
# %%
