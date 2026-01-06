import numpy as np

class MultiHeadSelfAttention():    

    def __init__(self, d_model, num_heads):
        # d_model --> Embedding model size
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model needs to be divisible by num_heads"
        self.d_k = d_model // num_heads

    def forward(self, X):
        B = len(X) # --> Batch size: Number of sequences
        T = len(X[0]) # --> Sequence length: Number of tokens in a sequence
        D = len(X[0][0]) # --> Embedding dimension: Size of each token vector
        
        for b in range(B):
            # Traversing each sequence in a batch of sequences.
            Q = self.W_q(X[b])
            K = self.W_k(X[b])
            V = self.W_v(X[b])

            Qh, Kh, Vh = [], [], []
            for h in range(self.num_heads):
                start = b * self.d_k
                end = start + self.d_k
                Qh.append([q[start:end] for q in Q])
                Kh.append([k[start:end] for k in K])
                Vh.append([v[start:end] for v in V])


    # Step 1: Compute Query, Key, Value
    def compute_Q_K_V(self):
        # X is the output of the embedding layer (plus positional encoding) for a sequence.
        # Shape of x --> n * d_model, where n is number of tokens in the sequence and d_model is the embedding model size 
        # W's are learned weight matrices that linearly project token embeddings into query, key, value
        # Shape of W's are d_model * d_k, where d_model is the embedding model size and d_k = d_model for single head for multi head i.e for n heads d_k becomes d_model / n
        # For example embedding model size is 512 and 8 heads, then d_k will be 512/8 = 64
        # Shape of Q, K, V is n * d_k
        # Q is Query, what the token is looking for in other tokens. Role: Used to compare against all keys
        # K is Key, what this token offers to other tokens. Role: Compared withe queries to determine relevance
        # V is Value, what information this token actually contributes. Role: Weighted and summed to form the output
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        return Q, K, V

# Step 2; Compute Similarity score, it's for finding relevant tokens. Higher score means more relevant
    def simiilarity(self):
        # Since the shape of Q and K is n * d_k, we are going to perform transpose on K, so we can perfrom matrix multiplication on Q and K
        # Shape of score will be n * n
        score = Q @ K.T
        return score

# Step 3: Scaling for similarity score
    def scale(self):
        # Since the dot products grow as the d_k i.e emdedding model size increases, and it grows by sqrt of d_k
        score = score / np.sqrt(d_k)

# Step 4: Computing Softmax from scaled similarity score
    def softmax(self):
        # softmax turns the scaled similarity scores into probablities. Takes a vector of real numbers i.e Scaled similarity scores ans tuens them into all positive numbers
        # All outputs will be between 0 and 1
        softmax = np.exp(S)/np.sum(np.exp(S), axis = 1)
        return softmax

    def compute_attention(self):
        output = softmax @ V 
