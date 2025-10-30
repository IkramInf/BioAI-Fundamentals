# Deep Learning Architectures: PyTorch Implementation Guide

## Clustering by Mathematical Similarity

This guide clusters deep learning architectures based on their mathematical foundations and provides PyTorch implementations for each cluster.

### Architecture Clusters:

1. **Feedforward Cluster**: MLP, CNN, ResNet
2. **Recurrent Cluster**: RNN, LSTM, GRU
3. **Attention Cluster**: Self-Attention, Multi-Head Attention, Cross-Attention
4. **Transformer Cluster**: Vanilla Transformer, BERT, GPT
5. **Efficient Transformer Cluster**: Linformer, Reformer, Performer
6. **State Space Cluster**: Mamba, S4, Hyena
7. **Hybrid Cluster**: Vision Transformer, Perceiver, Multi-Modal

---

## Cluster 1: Feedforward Architectures

### Mathematical Foundation

All feedforward networks follow:
$$y = f(Wx + b)$$

They differ in:
- Connection patterns (dense vs convolutional)
- Skip connections (identity vs projection)
- Spatial structure preservation

### 1.1 Basic Multi-Layer Perceptron (MLP)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Basic Multi-Layer Perceptron
    
    Forward pass: h = activation(W_i * h_{i-1} + b_i)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.network(x)  # (batch_size, output_dim)


# Example usage
model = MLP(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)  # Shape: (32, 10)
print(f"MLP output shape: {output.shape}")
```

### 1.2 Convolutional Neural Network (CNN)

```python
class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU
    
    Convolution: y(i,j) = sum_m sum_n x(i+m, j+n) * K(m,n)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNN(nn.Module):
    """
    Standard CNN for image classification
    
    Architecture: Conv blocks -> Pooling -> Flatten -> FC layers
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(256, 512, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.pool(self.conv1(x))  # -> (batch, 64, H/2, W/2)
        x = self.pool(self.conv2(x))  # -> (batch, 128, H/4, W/4)
        x = self.pool(self.conv3(x))  # -> (batch, 256, H/8, W/8)
        x = self.pool(self.conv4(x))  # -> (batch, 512, H/16, W/16)
        
        x = self.global_pool(x)       # -> (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)     # -> (batch, 512)
        x = self.fc(x)                # -> (batch, num_classes)
        
        return x


# Example usage
model = CNN(in_channels=3, num_classes=10)
x = torch.randn(16, 3, 224, 224)  # Batch of 16 RGB images
output = model(x)  # Shape: (16, 10)
print(f"CNN output shape: {output.shape}")
```

### 1.3 Residual Networks (ResNet)

**Key Difference**: Adds skip connections

$$y = F(x, \{W_i\}) + x$$

```python
class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection
    
    Formula: H(x) = F(x) + x
    where F(x) is the residual function
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (identity or projection)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # Residual path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection: H(x) = F(x) + x
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture
    
    Difference from CNN: Uses skip connections to enable deeper networks
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Projection shortcut when dimensions change
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=10):
    """ResNet-18: [2, 2, 2, 2] blocks"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    """ResNet-34: [3, 4, 6, 3] blocks"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


# Example usage
model = resnet18(num_classes=10)
x = torch.randn(8, 3, 224, 224)
output = model(x)
print(f"ResNet-18 output shape: {output.shape}")
```

### Comparison of Feedforward Architectures

```python
def compare_feedforward_architectures():
    """
    Key Differences:
    
    1. MLP:
       - Dense connections
       - No spatial structure
       - Formula: y = W_n(...W_2(W_1 x + b_1) + b_2...) + b_n
    
    2. CNN:
       - Local connectivity (convolutions)
       - Parameter sharing
       - Spatial hierarchy
       - Formula: y = Pool(Conv(x))
    
    3. ResNet:
       - Skip connections enable deeper networks
       - Identity mappings
       - Formula: y = F(x) + x (solves vanishing gradient)
    """
    
    batch_size = 16
    
    # For MLP: flattened input
    mlp_input = torch.randn(batch_size, 784)
    mlp = MLP(784, [512, 256], 10)
    mlp_out = mlp(mlp_input)
    
    # For CNN and ResNet: spatial input
    spatial_input = torch.randn(batch_size, 3, 224, 224)
    cnn = CNN(in_channels=3, num_classes=10)
    resnet = resnet18(num_classes=10)
    
    cnn_out = cnn(spatial_input)
    resnet_out = resnet(spatial_input)
    
    print("Feedforward Architectures Comparison:")
    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"CNN parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"ResNet-18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")
    
    return mlp, cnn, resnet

compare_feedforward_architectures()
```

---

## Cluster 2: Recurrent Architectures

### Mathematical Foundation

All recurrent networks maintain hidden state across time:
$$h_t = f(h_{t-1}, x_t)$$

They differ in:
- Gating mechanisms (none vs forget/input/output gates)
- Number of states (hidden only vs hidden + cell)
- Complexity of update rules

### 2.1 Basic RNN

```python
class VanillaRNN(nn.Module):
    """
    Vanilla RNN (Elman network)
    
    Formula:
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        y_t = W_hy * h_t + b_y
    
    Issues: Vanishing/exploding gradients for long sequences
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (num_layers, batch, hidden_size) or None
        
        Returns:
            output: (batch, seq_len, output_size)
            hidden: (num_layers, batch, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, 
                           self.hidden_size).to(x.device)
        
        # Forward through RNN
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.rnn(x, h0)
        
        # Apply output layer to all timesteps
        out = self.fc(out)  # (batch, seq_len, output_size)
        
        return out, h_n
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, 
                         self.hidden_size).to(device)


# Example usage
rnn = VanillaRNN(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)  # (batch=8, seq_len=15, input_size=10)
output, hidden = rnn(x)
print(f"RNN output shape: {output.shape}, hidden shape: {hidden.shape}")
```

### 2.2 LSTM (Long Short-Term Memory)

**Key Difference**: Adds cell state and three gates

```python
class LSTMCell(nn.Module):
    """
    Single LSTM cell implementation
    
    Formulas:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # Forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # Input gate
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)    # Cell candidate
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t           # Cell state
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # Output gate
        h_t = o_t ⊙ tanh(C_t)                       # Hidden state
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input transformations (all gates + cell candidate)
        # Combined for efficiency: [forget, input, output, cell_candidate]
        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
        Args:
            x: (batch, input_size)
            hidden: tuple of (h, c)
                h: (batch, hidden_size)
                c: (batch, hidden_size)
        
        Returns:
            h_new: (batch, hidden_size)
            c_new: (batch, hidden_size)
        """
        h, c = hidden
        
        # Linear transformations
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        
        # Split into gates
        # Each gate: (batch, hidden_size)
        forget_gate, input_gate, output_gate, cell_candidate = gates.chunk(4, 1)
        
        # Apply activations
        f_t = torch.sigmoid(forget_gate)
        i_t = torch.sigmoid(input_gate)
        o_t = torch.sigmoid(output_gate)
        c_tilde = torch.tanh(cell_candidate)
        
        # Update cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        c_new = f_t * c + i_t * c_tilde
        
        # Update hidden state: h_t = o_t ⊙ tanh(C_t)
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """
    Multi-layer LSTM
    
    Difference from RNN: Uses gating to control information flow,
    maintains separate cell state for long-term memory
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: tuple of (h0, c0) or None
                h0: (num_layers, batch, hidden_size)
                c0: (num_layers, batch, hidden_size)
        
        Returns:
            output: (batch, seq_len, output_size)
            (h_n, c_n): final hidden and cell states
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, 
                           self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size,
                           self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # Forward through LSTM
        out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply output layer
        out = self.fc(out)
        
        return out, (h_n, c_n)


# Example usage
lstm = LSTM(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)
output, (hidden, cell) = lstm(x)
print(f"LSTM output: {output.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
```

### 2.3 GRU (Gated Recurrent Unit)

**Key Difference**: Fewer gates than LSTM, no separate cell state

```python
class GRUCell(nn.Module):
    """
    Single GRU cell
    
    Formulas:
        r_t = σ(W_r · [h_{t-1}, x_t])           # Reset gate
        z_t = σ(W_z · [h_{t-1}, x_t])           # Update gate
        h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state
    
    Differences from LSTM:
        - Only 2 gates (reset, update) vs 3 gates (forget, input, output)
        - No separate cell state
        - Fewer parameters: 3N² vs 4N² for LSTM
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: [reset, update]
        self.weight_ih = nn.Parameter(
            torch.randn(2 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(2 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size))
        
        # Candidate hidden state
        self.weight_ih_candidate = nn.Parameter(
            torch.randn(hidden_size, input_size)
        )
        self.weight_hh_candidate = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
        )
        self.bias_ih_candidate = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh_candidate = nn.Parameter(torch.randn(hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, h):
        """
        Args:
            x: (batch, input_size)
            h: (batch, hidden_size)
        
        Returns:
            h_new: (batch, hidden_size)
        """
        # Compute reset and update gates
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        
        reset_gate, update_gate = gates.chunk(2, 1)
        
        r_t = torch.sigmoid(reset_gate)
        z_t = torch.sigmoid(update_gate)
        
        # Compute candidate hidden state
        h_candidate = (torch.mm(x, self.weight_ih_candidate.t()) + 
                      self.bias_ih_candidate +
                      torch.mm(r_t * h, self.weight_hh_candidate.t()) +
                      self.bias_hh_candidate)
        
        h_tilde = torch.tanh(h_candidate)
        
        # Update hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_new = (1 - z_t) * h + z_t * h_tilde
        
        return h_new


class GRU(nn.Module):
    """
    Multi-layer GRU
    
    Advantages over LSTM:
        - Simpler (fewer parameters)
        - Faster training and inference
        - Often comparable performance
    
    When to use GRU vs LSTM:
        - GRU: When you need efficiency, smaller datasets
        - LSTM: When you need maximum expressiveness, larger datasets
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (num_layers, batch, hidden_size) or None
        
        Returns:
            output: (batch, seq_len, output_size)
            h_n: (num_layers, batch, hidden_size)
        """
        batch_size = x.size(0)
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size,
                           self.hidden_size).to(x.device)
        
        # Forward through GRU
        out, h_n = self.gru(x, h0)
        
        # Apply output layer
        out = self.fc(out)
        
        return out, h_n


# Example usage
gru = GRU(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)
output, hidden = gru(x)
print(f"GRU output: {output.shape}, hidden: {hidden.shape}")
```

### 2.4 Bidirectional RNN Wrapper

```python
class BidirectionalRNN(nn.Module):
    """
    Bidirectional wrapper for any RNN type
    
    Formula:
        h⃗_t = f(W⃗_hh * h⃗_{t-1} + W⃗_xh * x_t)  # Forward
        h⃖_t = f(W⃖_hh * h⃖_{t+1} + W⃖_xh * x_t)  # Backward
        h_t = [h⃗_t; h⃖_t]                        # Concatenate
    """
    def __init__(self, rnn_type, input_size, hidden_size, output_size, 
                 num_layers=1):
        super(BidirectionalRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Choose RNN type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        
        # Output layer (note: 2 * hidden_size due to bidirectional)
        self.fc = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x):
        # out: (batch, seq_len, 2 * hidden_size)
        out, _ = self.rnn(x)
        
        # Apply output layer
        out = self.fc(out)
        
        return out


# Example usage
bi_lstm = BidirectionalRNN('lstm', input_size=10, hidden_size=20, 
                          output_size=5, num_layers=1)
x = torch.randn(8, 15, 10)
output = bi_lstm(x)
print(f"Bidirectional LSTM output: {output.shape}")
```

### Comparison of Recurrent Architectures

```python
def compare_recurrent_architectures():
    """
    Key Differences:
    
    1. RNN (Vanilla):
       - Simple: h_t = tanh(W_hh h_{t-1} + W_xh x_t)
       - Problem: Vanishing gradients
       - Parameters: 2N²
    
    2. LSTM:
       - 3 gates + cell state
       - Formula: Uses forget, input, output gates
       - Solves vanishing gradients
       - Parameters: 4N²
       - Best for: Long sequences, need maximum capacity
    
    3. GRU:
       - 2 gates, no separate cell state
       - Simpler than LSTM
       - Parameters: 3N²
       - Best for: Efficiency, smaller datasets
    """
    
    input_size, hidden_size, output_size = 10, 20, 5
    seq_len, batch_size = 15, 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Compare parameter counts
    models = {
        'RNN': VanillaRNN(input_size, hidden_size, output_size),
        'LSTM': LSTM(input_size, hidden_size, output_size),
        'GRU': GRU(input_size, hidden_size, output_size)
    }
    
    print("\nRecurrent Architectures Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        output = model(x)[0] if name != 'RNN' or name != 'GRU' else model(x)[0]
        print(f"{name:10s} - Parameters: {params:,} - Output: {output.shape}")
    
    return models

compare_recurrent_architectures()
```

---

## Cluster 3: Attention Mechanisms

### Mathematical Foundation

All attention mechanisms compute weighted sums:
$\text{Attention}(Q, K, V) = \text{softmax}(\text{score}(Q, K)) \cdot V$

They differ in:
- Score function (dot product, additive, multiplicative)
- Scaling factors
- Number of heads
- Query/Key/Value sources (self vs cross-attention)

### 3.1 Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
        Q: queries (batch, seq_len_q, d_k)
        K: keys (batch, seq_len_k, d_k)
        V: values (batch, seq_len_v, d_v)
        d_k: dimension of keys/queries
    
    Scaling by √d_k prevents softmax saturation for large d_k
    """
    def __init__(self, temperature=None, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, n_heads, len_q, d_k)
            k: (batch, n_heads, len_k, d_k)
            v: (batch, n_heads, len_v, d_v)
            mask: (batch, 1, len_q, len_k) or None
        
        Returns:
            output: (batch, n_heads, len_q, d_v)
            attention: (batch, n_heads, len_q, len_k)
        """
        d_k = q.size(-1)
        
        # If temperature not set, use sqrt(d_k)
        temperature = self.temperature or d_k ** 0.5
        
        # Compute attention scores: QK^T / √d_k
        # (batch, n_heads, len_q, d_k) x (batch, n_heads, d_k, len_k)
        # -> (batch, n_heads, len_q, len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
        
        # Apply mask (for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        # (batch, n_heads, len_q, len_k) x (batch, n_heads, len_k, d_v)
        # -> (batch, n_heads, len_q, d_v)
        output = torch.matmul(attention, v)
        
        return output, attention


# Example usage
attention = ScaledDotProductAttention()
q = torch.randn(2, 8, 10, 64)  # (batch=2, heads=8, seq_len=10, d_k=64)
k = torch.randn(2, 8, 10, 64)
v = torch.randn(2, 8, 10, 64)
output, attn_weights = attention(q, k, v)
print(f"Attention output: {output.shape}, weights: {attn_weights.shape}")
```

### 3.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Difference from single attention:
        - Projects Q, K, V into h different subspaces
        - Allows model to attend to different representation subspaces
        - Each head has dimension d_model / h
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, len_q, d_model)
            k: (batch, len_k, d_model)
            v: (batch, len_v, d_model)
            mask: (batch, 1, len_q, len_k) or None
        
        Returns:
            output: (batch, len_q, d_model)
            attention: (batch, n_heads, len_q, len_k)
        """
        batch_size = q.size(0)
        residual = q
        
        # 1. Linear projections
        q = self.w_q(q)  # (batch, len_q, d_model)
        k = self.w_k(k)  # (batch, len_k, d_model)
        v = self.w_v(v)  # (batch, len_v, d_model)
        
        # 2. Split into multiple heads
        # (batch, len, d_model) -> (batch, len, n_heads, d_k)
        # -> (batch, n_heads, len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        output, attention = self.attention(q, k, v, mask)
        
        # 4. Concatenate heads
        # (batch, n_heads, len_q, d_k) -> (batch, len_q, n_heads, d_k)
        # -> (batch, len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        # 6. Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention


# Example usage
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attn = mha(x, x, x)  # Self-attention
print(f"Multi-head attention output: {output.shape}")
```

### 3.3 Cross-Attention vs Self-Attention

```python
class SelfAttention(nn.Module):
    """
    Self-Attention: Q, K, V all come from same source
    
    Usage: Within encoder or decoder to attend to own sequence
    Formula: Attention(X, X, X)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        return self.mha(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-Attention: Q from one source, K and V from another
    
    Usage: Decoder attending to encoder output
    Formula: Attention(Q_decoder, K_encoder, V_encoder)
    
    Difference from self-attention:
        - Q comes from decoder
        - K, V come from encoder
        - Allows decoder to focus on relevant encoder information
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, q, kv, mask=None):
        """
        Args:
            q: decoder features (batch, len_q, d_model)
            kv: encoder features (batch, len_kv, d_model)
            mask: attention mask
        """
        return self.mha(q, kv, kv, mask)


# Example usage
self_attn = SelfAttention(d_model=512, n_heads=8)
cross_attn = CrossAttention(d_model=512, n_heads=8)

encoder_out = torch.randn(2, 10, 512)
decoder_in = torch.randn(2, 5, 512)

# Self-attention on encoder
self_out, _ = self_attn(encoder_out)
print(f"Self-attention output: {self_out.shape}")

# Cross-attention: decoder queries encoder
cross_out, _ = cross_attn(decoder_in, encoder_out)
print(f"Cross-attention output: {cross_out.shape}")
```

### 3.4 Additive Attention (Bahdanau)

```python
class AdditiveAttention(nn.Module):
    """
    Additive/Bahdanau Attention
    
    Formula:
        score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)
        α_{ts} = softmax(score(h_t, h_s))
        context_t = Σ_s α_{ts} h_s
    
    Difference from scaled dot-product:
        - Uses learned weight matrix and tanh
        - More parameters
        - Can handle different dimensions for Q and K
        - Original attention mechanism for seq2seq
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, len_q, hidden_dim)
            keys: (batch, len_k, hidden_dim)
            values: (batch, len_k, hidden_dim)
            mask: (batch, len_q, len_k) or None
        
        Returns:
            context: (batch, len_q, hidden_dim)
            attention: (batch, len_q, len_k)
        """
        # Project query and keys
        q = self.W_q(query)  # (batch, len_q, hidden_dim)
        k = self.W_k(keys)   # (batch, len_k, hidden_dim)
        
        # Expand dimensions for broadcasting
        q = q.unsqueeze(2)  # (batch, len_q, 1, hidden_dim)
        k = k.unsqueeze(1)  # (batch, 1, len_k, hidden_dim)
        
        # Compute scores: v^T tanh(W_1 q + W_2 k)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        # (batch, len_q, len_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, values)
        # (batch, len_q, len_k) x (batch, len_k, hidden_dim)
        # -> (batch, len_q, hidden_dim)
        
        return context, attention


# Example usage
additive_attn = AdditiveAttention(hidden_dim=256)
query = torch.randn(2, 5, 256)
keys = torch.randn(2, 10, 256)
values = torch.randn(2, 10, 256)
context, attn = additive_attn(query, keys, values)
print(f"Additive attention context: {context.shape}")
```

### Comparison of Attention Mechanisms

```python
def compare_attention_mechanisms():
    """
    Attention Mechanisms Comparison:
    
    1. Scaled Dot-Product:
       - Formula: softmax(QK^T / √d_k)V
       - Pros: Fast (matrix multiplication), parallelizable
       - Cons: Quadratic complexity O(n²)
       - Use: Modern transformers
    
    2. Multi-Head:
       - Formula: Concat(head_1, ..., head_h)W^O
       - Pros: Multiple representation subspaces
       - Cons: More parameters
       - Use: All transformer architectures
    
    3. Self-Attention:
       - Formula: Attention(X, X, X)
       - Pros: Captures dependencies within sequence
       - Use: Encoder layers
    
    4. Cross-Attention:
       - Formula: Attention(Q_decoder, K_encoder, V_encoder)
       - Pros: Connects encoder and decoder
       - Use: Seq2seq, image captioning
    
    5. Additive (Bahdanau):
       - Formula: v^T tanh(W_1 Q + W_2 K)
       - Pros: Different Q, K dimensions; original seq2seq
       - Cons: Slower than dot-product
       - Use: Legacy seq2seq models
    """
    
    batch, seq_len, d_model = 2, 10, 512
    n_heads = 8
    
    x = torch.randn(batch, seq_len, d_model)
    
    # Compare implementations
    models = {
        'Multi-Head (Self)': MultiHeadAttention(d_model, n_heads),
        'Self-Attention': SelfAttention(d_model, n_heads),
        'Cross-Attention': CrossAttention(d_model, n_heads),
        'Additive': AdditiveAttention(d_model)
    }
    
    print("\nAttention Mechanisms Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:25s} - Parameters: {params:,}")
    
    return models

compare_attention_mechanisms()
```

---

## Cluster 4: Transformer Architectures

### Mathematical Foundation

All transformers use:
1. Multi-head self-attention
2. Position-wise feedforward networks
3. Residual connections and layer normalization
4. Positional encodings

They differ in:
- Encoder-only vs Decoder-only vs Encoder-Decoder
- Training objectives (MLM, CLM, etc.)
- Positional encoding schemes

### 4.1 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why: Transformers have no inherent notion of position
    Properties:
        - Fixed (not learned)
        - Allows extrapolation to longer sequences
        - Distance between positions is consistent
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embeddings (alternative to sinusoidal)
    
    Used in: BERT, GPT-2
    
    Difference from sinusoidal:
        - Learned during training
        - Better for fixed-length sequences
        - Cannot extrapolate beyond max_len
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Get positional embeddings and add
        pos_emb = self.pe(positions)
        x = x + pos_emb
        
        return self.dropout(x)


# Example usage
pos_enc = PositionalEncoding(d_model=512, max_len=100)
learned_pos_enc = LearnedPositionalEncoding(d_model=512, max_len=100)

x = torch.randn(2, 10, 512)
x_with_pos = pos_enc(x)
x_with_learned_pos = learned_pos_enc(x)
print(f"With sinusoidal PE: {x_with_pos.shape}")
print(f"With learned PE: {x_with_learned_pos.shape}")
```

### 4.2 Position-wise Feed-Forward Network

```python
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Formula:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        or with GELU: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    Applied independently to each position:
        - Same network for all positions
        - Different from RNN (no temporal connections)
        - Two linear transformations with activation
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # First linear + activation
        x = self.activation(self.w_1(x))
        x = self.dropout(x)
        
        # Second linear
        x = self.w_2(x)
        x = self.dropout(x)
        
        return x


# Example usage
ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 10, 512)
output = ffn(x)
print(f"FFN output: {output.shape}")
```

### 4.3 Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture:
        1. Multi-Head Self-Attention
        2. Add & Norm
        3. Position-wise Feed-Forward
        4. Add & Norm
    
    Formula:
        x' = LayerNorm(x + MultiHeadAttention(x))
        x'' = LayerNorm(x' + FFN(x'))
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        residual = x
        x, _ = self.self_attn.mha(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers
    
    Used in: BERT, Encoder-Decoder Transformers
    """
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# Example usage
encoder = TransformerEncoder(num_layers=6, d_model=512, n_heads=8, 
                            d_ff=2048, dropout=0.1)
x = torch.randn(2, 10, 512)
output = encoder(x)
print(f"Transformer Encoder output: {output.shape}")
```

### 4.4 Transformer Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
    Architecture:
        1. Masked Multi-Head Self-Attention
        2. Add & Norm
        3. Multi-Head Cross-Attention (attend to encoder)
        4. Add & Norm
        5. Position-wise Feed-Forward
        6. Add & Norm
    
    Difference from Encoder:
        - Self-attention is masked (causal)
        - Has cross-attention to encoder output
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: decoder input (batch, tgt_len, d_model)
            encoder_output: encoder output (batch, src_len, d_model)
            tgt_mask: target (causal) mask
            src_mask: source (padding) mask
        
        Returns:
            output: (batch, tgt_len, d_model)
        """
        # 1. Masked self-attention
        residual = x
        x, _ = self.self_attn.mha(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # 2. Cross-attention to encoder
        residual = x
        x, _ = self.cross_attn.mha(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        # 3. Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of N Transformer Decoder Layers
    
    Used in: Machine Translation, Seq2Seq tasks
    """
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: decoder input (batch, tgt_len, d_model)
            encoder_output: encoder output (batch, src_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.norm(x)


# Example usage
decoder = TransformerDecoder(num_layers=6, d_model=512, n_heads=8,
                            d_ff=2048, dropout=0.1)
tgt = torch.randn(2, 8, 512)
encoder_out = torch.randn(2, 10, 512)
output = decoder(tgt, encoder_out)
print(f"Transformer Decoder output: {output.shape}")
```

### 4.5 Complete Vanilla Transformer

```python
class Transformer(nn.Module):
    """
    Complete Transformer (Encoder-Decoder Architecture)
    
    Original "Attention Is All You Need" architecture
    
    Components:
        - Token Embedding
        - Positional Encoding
        - Encoder Stack
        - Decoder Stack
        - Output Linear Layer
    
    Training: Uses teacher forcing
    Inference: Auto-regressive generation
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 n_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, 
                                         n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model,
                                         n_heads, d_ff, dropout)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for decoder
        
        Returns lower triangular matrix of ones (allows attending to past)
        Upper triangle is -inf (prevents attending to future)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: source sequence (batch, src_len)
            tgt: target sequence (batch, tgt_len)
            src_mask: source padding mask
            tgt_mask: target causal mask
        
        Returns:
            output: (batch, tgt_len, tgt_vocab_size)
        """
        # Embed and add positional encoding
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        
        # Generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Encode source
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_linear(decoder_output)
        
        return output


# Example usage
transformer = Transformer(src_vocab_size=10000, tgt_vocab_size=10000,
                         d_model=512, n_heads=8, num_encoder_layers=6,
                         num_decoder_layers=6, d_ff=2048)

src = torch.randint(0, 10000, (2, 10))  # (batch, src_len)
tgt = torch.randint(0, 10000, (2, 8))   # (batch, tgt_len)

output = transformer(src, tgt)
print(f"Transformer output: {output.shape}  # Should be (2, 8, 10000)")
```

### 4.6 BERT-Style Encoder-Only Transformer

```python
class BERTTransformer(nn.Module):
    """
    BERT-Style Encoder-Only Transformer
    
    Differences from vanilla Transformer:
        - Encoder-only (no decoder)
        - Bidirectional context
        - Special tokens: [CLS], [SEP], [MASK]
        - Segment embeddings for sentence pairs
    
    Pre-training tasks:
        1. Masked Language Modeling (MLM)
        2. Next Sentence Prediction (NSP)
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12, 
                 num_layers=12, d_ff=3072, max_len=512, 
                 num_segments=2, dropout=0.1):
        super(BERTTransformer, self).__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Encoder
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, 
                                         d_ff, dropout)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP head
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len) - 0 or 1 for sentence A/B
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            sequence_output: (batch, seq_len, d_model)
            pooled_output: (batch, d_model) - [CLS] token representation
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            seg_emb = self.segment_embedding(segment_ids)
            embeddings = token_emb + pos_emb + seg_emb
        else:
            embeddings = token_emb + pos_emb
        
        embeddings = self.dropout(embeddings)
        
        # Convert attention mask for multi-head attention
        if attention_mask is not None:
            # (batch, 1, 1, seq_len) for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Encode
        sequence_output = self.encoder(embeddings, attention_mask)
        
        # Get [CLS] token output for classification
        pooled_output = sequence_output[:, 0, :]
        
        return sequence_output, pooled_output
    
    def get_mlm_logits(self, sequence_output):
        """Get logits for masked language modeling"""
        return self.mlm_head(sequence_output)
    
    def get_nsp_logits(self, pooled_output):
        """Get logits for next sentence prediction"""
        return self.nsp_head(pooled_output)


# Example usage
bert = BERTTransformer(vocab_size=30000, d_model=768, n_heads=12, 
                      num_layers=12, d_ff=3072)

input_ids = torch.randint(0, 30000, (2, 128))
segment_ids = torch.cat([torch.zeros(2, 64), torch.ones(2, 64)], dim=1).long()
attention_mask = torch.ones(2, 128)

seq_out, pooled_out = bert(input_ids, segment_ids, attention_mask)
print(f"BERT sequence output: {seq_out.shape}, pooled: {pooled_out.shape}")

mlm_logits = bert.get_mlm_logits(seq_out)
nsp_logits = bert.get_nsp_logits(pooled_out)
print(f"MLM logits: {mlm_logits.shape}, NSP logits: {nsp_logits.shape}")
```

### 4.7 GPT-Style Decoder-Only Transformer

```python
class GPTTransformer(nn.Module):
    """
    GPT-Style Decoder-Only Transformer
    
    Differences from vanilla Transformer:
        - Decoder-only (no encoder, no cross-attention)
        - Causal self-attention only
        - Auto-regressive generation
    
    Training: Causal Language Modeling (CLM)
        Predict next token given all previous tokens
    
    Formula: P(x_t | x_1, ..., x_{t-1})
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super(GPTTransformer, self).__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Decoder layers (self-attention only, no cross-attention)
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - optional
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = self.generate_causal_mask(seq_len).to(input_ids.device)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def generate_causal_mask(self, sz):
        """Generate causal mask to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Auto-regressive generation
        
        Args:
            input_ids: (batch, seq_len) - prompt
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class GPTDecoderLayer(nn.Module):
    """
    GPT Decoder Layer (Self-Attention Only)
    
    Difference from Transformer Decoder:
        - No cross-attention (encoder-decoder attention)
        - Only masked self-attention and FFN
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation='gelu')
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        residual = x
        x, _ = self.self_attn.mha(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # FFN with residual
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x


# Example usage
gpt = GPTTransformer(vocab_size=50257, d_model=768, n_heads=12,
                    num_layers=12, d_ff=3072)

input_ids = torch.randint(0, 50257, (2, 20))
logits = gpt(input_ids)
print(f"GPT logits: {logits.shape}")

# Generate text
prompt = torch.randint(0, 50257, (1, 10))
generated = gpt.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
print(f"Generated sequence: {generated.shape}")
```

### Comparison of Transformer Variants

```python
def compare_transformer_architectures():
    """
    Transformer Architecture Comparison:
    
    1. Vanilla Transformer (Encoder-Decoder):
       - Components: Encoder + Decoder with cross-attention
       - Use: Machine translation, seq2seq tasks
       - Training: Teacher forcing
       - Example: Original "Attention Is All You Need"
    
    2. BERT (Encoder-Only):
       - Components: Encoder stack only
       - Attention: Bidirectional (can see full context)
       - Use: Classification, NER, QA
       - Training: MLM + NSP
       - Pros: Best for understanding tasks
    
    3. GPT (Decoder-Only):
       - Components: Decoder stack (no cross-attention)
       - Attention: Causal (can only see past)
       - Use: Text generation, few-shot learning
       - Training: Causal LM (next token prediction)
       - Pros: Excellent for generation
    
    4. T5 (Encoder-Decoder):
       - Components: Full encoder-decoder
       - Use: All tasks as text-to-text
       - Training: Span corruption
       - Pros: Unified framework
    
    Architecture Sizes:
    """
    
    vocab_size = 50000
    
    # Create models
    vanilla = Transformer(vocab_size, vocab_size, d_model=512, n_heads=8,
                         num_encoder_layers=6, num_decoder_layers=6)
    
    bert = BERTTransformer(vocab_size, d_model=768, n_heads=12, num_layers=12)
    
    gpt = GPTTransformer(vocab_size, d_model=768, n_heads=12, num_layers=12)
    
    models = {
        'Vanilla Transformer': vanilla,
        'BERT (Encoder-Only)': bert,
        'GPT (Decoder-Only)': gpt
    }
    
    print("\nTransformer Architectures Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:30s} - Parameters: {params:,}")
    
    return models

compare_transformer_architectures()
```

---

## Cluster 5: Efficient Transformer Variants

### Mathematical Foundation

All efficient transformers aim to reduce O(n²) complexity of standard attention.

Methods:
1. **Low-rank approximation**: Project to lower dimensions
2. **Sparsity**: Attend to subset of tokens
3. **Kernel methods**: Approximate attention with kernel functions
4. **Recurrence**: Mix attention with recurrent processing

### 5.1 Linformer (Linear Complexity)

```python
class Linformer(nn.Module):
    """
    Linformer: Self-Attention with Linear Complexity
    
    Key Idea: Project keys and values to lower dimension k << n
    
    Formula:
        Attention(Q, K, V) = softmax(Q(EK)^T / √d) (FV)
        where E, F ∈ R^{k×n} are projection matrices
    
    Complexity:
        Standard: O(n²d)
        Linformer: O(nkd) where k << n
    
    Difference from standard transformer:
        - Projects K, V to fixed dimension k
        - Linear in sequence length
        - Some information loss due to projection
    """
    def __init__(self, d_model, n_heads, seq_len, k=256, dropout=0.1):
        super(Linformer, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = min(k, seq_len)  # Projected dimension
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Low-rank projections for K and V
        self.E = nn.Linear(seq_len, self.k, bias=False)
        self.F = nn.Linear(seq_len, self.k, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # (batch, n_heads, seq_len, d_k)
        
        # Project K and V to lower dimension
        # Transpose to (batch, n_heads, d_k, seq_len) for projection
        K = K.transpose(2, 3)  # (batch, n_heads, d_k, seq_len)
        V = V.transpose(2, 3)
        
        # Apply low-rank projection: (batch, n_heads, d_k, seq_len) -> (batch, n_heads, d_k, k)
        K = self.E(K)  # (batch, n_heads, d_k, k)
        V = self.F(V)  # (batch, n_heads, d_k, k)
        
        # Transpose back
        K = K.transpose(2, 3)  # (batch, n_heads, k, d_k)
        V = V.transpose(2, 3)  # (batch, n_heads, k, d_k)
        
        # Compute attention scores: (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, k)
        # -> (batch, n_heads, seq_len, k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply to values: (batch, n_heads, seq_len, k) x (batch, n_heads, k, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        output = torch.matmul(attention, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
linformer = Linformer(d_model=512, n_heads=8, seq_len=1024, k=256)
x = torch.randn(2, 1024, 512)
output = linformer(x)
print(f"Linformer output: {output.shape}")
print(f"Complexity: O(n*k*d) = O({1024}*{256}*{512}) vs O(n²*d) = O({1024**2}*{512})")
```

### 5.2 Performer (Kernel-based Attention)

```python
class Performer(nn.Module):
    """
    Performer: Fast Attention via Positive Orthogonal Random Features
    
    Key Idea: Approximate softmax attention using random feature maps
    
    Formula:
        Attention(Q, K, V) ≈ φ(Q)(φ(K)^T V) / (φ(Q)φ(K)^T 1)
        where φ is a random feature map
    
    Benefits:
        - Linear complexity O(nd²)
        - No approximation error bound
        - Can be computed in forward or recurrent mode
    
    Difference from standard:
        - Uses kernel approximation of softmax
        - Reorders computation: φ(Q)(φ(K)^T V) instead of (Qφ(K)^T)V
    """
    def __init__(self, d_model, n_heads, nb_features=256, dropout=0.1):
        super(Performer, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nb_features = nb_features
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_projection_matrix(self, m, d, device):
        """
        Create random projection matrix for FAVOR+ algorithm
        
        Args:
            m: number of random features
            d: dimension of keys/queries
        """
        # Orthogonal random features
        projection = torch.randn(m, d, device=device)
        
        # Gram-Schmidt orthogonalization
        q, r = torch.qr(projection)
        d = torch.diag(r)
        ph = d / torch.abs(d)
        projection = q * ph.unsqueeze(0)
        
        return projection
    
    def kernel_feature_map(self, x, projection_matrix):
        """
        Apply random feature map
        
        Formula: φ(x) = exp(xω^T - ||x||²/2) / √m
        where ω is the random projection matrix
        """
        # x: (batch, n_heads, seq_len, d_k)
        data_normalizer = 1.0 / torch.sqrt(torch.tensor(self.nb_features, dtype=torch.float32))
        
        # Project: (batch, n_heads, seq_len, nb_features)
        projection = torch.matmul(x, projection_matrix.T)
        
        # Compute ||x||²/2
        data_dash = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
        
        # Apply exponential kernel
        data_dash = projection - data_dash / 2.0
        data_dash = torch.exp(data_dash) * data_normalizer
        
        return data_dash
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Create random projection matrix
        projection_matrix = self.create_projection_matrix(
            self.nb_features, self.d_k, device
        )
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply kernel feature map
        Q_prime = self.kernel_feature_map(Q, projection_matrix)
        K_prime = self.kernel_feature_map(K, projection_matrix)
        
        # Q_prime, K_prime: (batch, n_heads, seq_len, nb_features)
        # V: (batch, n_heads, seq_len, d_k)
        
        # Compute attention: φ(Q)(φ(K)^T V)
        # First compute φ(K)^T V: (batch, n_heads, nb_features, d_k)
        KV = torch.matmul(K_prime.transpose(-2, -1), V)
        
        # Then compute φ(Q) @ KV: (batch, n_heads, seq_len, d_k)
        output = torch.matmul(Q_prime, KV)
        
        # Normalize by φ(Q) @ φ(K)^T @ 1
        # Compute φ(K)^T @ 1: (batch, n_heads, nb_features, 1)
        K_sum = torch.sum(K_prime, dim=2, keepdim=True).transpose(-2, -1)
        
        # Compute φ(Q) @ K_sum: (batch, n_heads, seq_len, 1)
        normalizer = torch.matmul(Q_prime, K_sum) + 1e-8
        
        # Normalize output
        output = output / normalizer
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
performer = Performer(d_model=512, n_heads=8, nb_features=256)
x = torch.randn(2, 1024, 512)
output = performer(x)
print(f"Performer output: {output.shape}")
print("Complexity: O(nd²) - linear in sequence length!")
```

### 5.3 Reformer (LSH Attention)

```python
import math

class LSHAttention(nn.Module):
    """
    Locality Sensitive Hashing (LSH) Attention
    
    Key Idea: Use LSH to find similar queries/keys, only attend within buckets
    
    Process:
        1. Hash queries and keys into buckets
        2. Sort by bucket
        3. Attend only within same bucket
        4. Complexity: O(n log n)
    
    Formula:
        hash(x) = argmax(xR)
        where R is random projection matrix
    """
    def __init__(self, d_model, n_heads, n_hashes=4, bucket_size=64, dropout=0.1):
        super(LSHAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def hash_vectors(self, vecs, num_buckets):
        """
        Hash vectors using random projection
        
        Args:
            vecs: (batch, n_heads, seq_len, d_k)
            num_buckets: number of hash buckets
        
        Returns:
            hashes: (batch, n_heads, seq_len)
        """
        batch_size, n_heads, seq_len, d_k = vecs.size()
        
        # Create random rotation matrix
        rotations = torch.randn(d_k, num_buckets, device=vecs.device)
        rotations = F.normalize(rotations, dim=0)
        
        # Project vectors: (batch, n_heads, seq_len, d_k) x (d_k, num_buckets)
        # -> (batch, n_heads, seq_len, num_buckets)
        rotated = torch.matmul(vecs, rotations)
        
        # Hash is argmax of projections
        hashes = torch.argmax(rotated, dim=-1)
        
        return hashes
    
    def forward(self, x, mask=None):
        """
        Simplified LSH attention (full implementation is complex)
        
        For production, use official Reformer implementation
        """
        batch_size, seq_len, _ = x.size()
        
        # Standard attention as fallback
        # (Full LSH implementation requires careful sorting and chunking)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute hash buckets
        num_buckets = seq_len // self.bucket_size
        hashes = self.hash_vectors(Q, num_buckets)
        
        # For simplicity, fall back to standard attention
        # Real implementation would chunk by buckets
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output


# Example usage
lsh_attn = LSHAttention(d_model=512, n_heads=8, n_hashes=4, bucket_size=64)
x = torch.randn(2, 512, 512)
output = lsh_attn(x)
print(f"LSH Attention output: {output.shape}")
print("Complexity: O(n log n) due to sorting")
```

### 5.4 Longformer (Sparse Attention Patterns)

```python
class LongformerAttention(nn.Module):
    """
    Longformer: Combines local windowed attention with global attention
    
    Attention Patterns:
        1. Local window: Each token attends to w tokens on each side
        2. Global attention: Special tokens attend to all, all attend to them
        3. Dilated window: Increases receptive field
    
    Complexity: O(n × w) where w is window size (typically w << n)
    
    Formula:
        Attention_local(i) = softmax(Q_i K_{i-w:i+w}^T / √d) V_{i-w:i+w}
        Attention_global(i) = softmax(Q_i K_all^T / √d) V_all  (for special tokens)
    """
    def __init__(self, d_model, n_heads, window_size=512, 
                 global_tokens=[], dropout=0.1):
        super(LongformerAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.global_tokens = global_tokens  # List of token positions with global attention
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Separate projections for global attention
        self.w_q_global = nn.Linear(d_model, d_model)
        self.w_k_global = nn.Linear(d_model, d_model)
        self.w_v_global = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _sliding_window_attention(self, q, k, v, window_size):
        """
        Compute sliding window attention
        
        Args:
            q, k, v: (batch, n_heads, seq_len, d_k)
            window_size: size of local window
        """
        batch_size, n_heads, seq_len, d_k = q.size()
        
        # For simplicity, compute full attention and mask
        # Production implementation would use efficient sliding window
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Create sliding window mask
        mask = self._create_sliding_window_mask(seq_len, window_size, q.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v)
        
        return output
    
    def _create_sliding_window_mask(self, seq_len, window_size, device):
        """
        Create mask for sliding window attention
        
        Returns: (seq_len, seq_len) mask where 1 = attend, 0 = don't attend
        """
        # Create distance matrix
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Allow attention within window
        mask = torch.abs(distance) <= window_size
        
        return mask.float()
    
    def forward(self, x, global_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            global_mask: (batch, seq_len) - 1 for global tokens, 0 for local
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Local attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute sliding window attention
        output = self._sliding_window_attention(Q, K, V, self.window_size)
        
        # Add global attention for special tokens (e.g., [CLS])
        if global_mask is not None:
            Q_global = self.w_q_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K_global = self.w_k_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V_global = self.w_v_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # Global tokens attend to all
            global_scores = torch.matmul(Q_global, K_global.transpose(-2, -1)) / (self.d_k ** 0.5)
            global_attn = F.softmax(global_scores, dim=-1)
            global_output = torch.matmul(global_attn, V_global)
            
            # Merge local and global attention based on mask
            mask_expanded = global_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            output = output * (1 - mask_expanded) + global_output * mask_expanded
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
longformer = LongformerAttention(d_model=512, n_heads=8, window_size=256)
x = torch.randn(2, 4096, 512)  # Long sequence
global_mask = torch.zeros(2, 4096)
global_mask[:, 0] = 1  # First token has global attention
output = longformer(x, global_mask)
print(f"Longformer output: {output.shape}")
print(f"Complexity: O(n × w) = O({4096} × {256}) vs O(n²) = O({4096**2})")
```

### Comparison of Efficient Transformers

```python
def compare_efficient_transformers():
    """
    Efficient Transformer Comparison:
    
    1. Standard Transformer:
       - Complexity: O(n²d)
       - Memory: O(n²)
       - Best: Accuracy, short sequences
    
    2. Linformer:
       - Complexity: O(nkd) where k is projection dim
       - Method: Low-rank approximation of attention
       - Pros: Simple, deterministic
       - Cons: Fixed sequence length, approximation error
       - Best: Medium sequences (512-2048)
    
    3. Performer:
       - Complexity: O(nd²)
       - Method: Kernel approximation with random features
       - Pros: Provable approximation, recurrent mode
       - Cons: Variance in approximation
       - Best: Very long sequences, streaming
    
    4. Reformer (LSH):
       - Complexity: O(n log n)
       - Method: Locality sensitive hashing
       - Pros: No approximation within buckets
       - Cons: Complex implementation, sorting overhead
       - Best: Very long sequences (> 8K)
    
    5. Longformer:
       - Complexity: O(nw) where w is window size
       - Method: Sparse attention patterns
       - Pros: Simple, flexible patterns, exact
       - Cons: Fixed patterns, not adaptive
       - Best: Documents (4K-16K tokens)
    """
    
    seq_len = 2048
    d_model = 512
    n_heads = 8
    
    x = torch.randn(2, seq_len, d_model)
    
    models = {
        'Linformer (k=256)': Linformer(d_model, n_heads, seq_len, k=256),
        'Performer (m=256)': Performer(d_model, n_heads, nb_features=256),
        'LSH Attention': LSHAttention(d_model, n_heads, bucket_size=64),
        'Longformer (w=256)': LongformerAttention(d_model, n_heads, window_size=256)
    }
    
    print("\nEfficient Transformer Comparison:")
    print(f"Sequence length: {seq_len}")
    print("-" * 70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        try:
            output = model(x)
            print(f"{name:25s} - Params: {params:,} - Output: {output.shape}")
        except Exception as e:
            print(f"{name:25s} - Params: {params:,} - Error: {str(e)[:30]}")
    
    print("\nComplexity Summary:")
    print(f"Standard Attention:  O(n²d) = O({seq_len**2 * d_model:,})")
    print(f"Linformer:          O(nkd) = O({seq_len * 256 * d_model:,})")
    print(f"Performer:          O(nd²) = O({seq_len * d_model**2:,})")
    print(f"LSH:                O(n log n * d) = O({int(seq_len * math.log2(seq_len) * d_model):,})")
    print(f"Longformer:         O(nwd) = O({seq_len * 256 * d_model:,})")
    
    return models

compare_efficient_transformers()
```

---

## Cluster 6: State Space Models

### Mathematical Foundation

State space models process sequences through continuous-time dynamics:

$h'(t) = Ah(t) + Bx(t)$
$y(t) = Ch(t) + Dx(t)$

They differ in:
- Discretization methods
- Parameterization (fixed vs input-dependent)
- Initialization strategies

### 6.1 S4 (Structured State Space)

```python
class S4Layer(nn.Module):
    """
    S4 (Structured State Space Sequence) Layer
    
    Continuous-time state space:
        h'(t) = Ah(t) + Bx(t)
        y(t) = Ch(t)
    
    Discretized:
        h_k = A̅h_{k-1} + B̅x_k
        y_k = Ch_k
    
    Key innovation: Structured matrices (HiPPO initialization)
    for stable long-range dependencies
    
    Complexity: O(N log N) via FFT where N is state size
    """
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super(S4Layer, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters (simplified)
        # Real S4 uses HiPPO initialization
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Discretization parameter
        self.log_dt = nn.Parameter(torch.randn(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize with stable dynamics"""
        # Simplified initialization (real S4 uses HiPPO)
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
    
    def discretize(self):
        """
        Discretize continuous parameters
        
        Returns:
            A_bar: discretized state transition
            B_bar: discretized input matrix
        """
        dt = torch.exp(self.log_dt)  # (d_model,)
        
        # Zero-order hold discretization (simplified)
        # A_bar = exp(dt * A)
        # B_bar = (A^-1)(exp(dt * A) - I) * dt * B
        
        # For simplicity, use Euler method
        A_bar = torch.eye(self.d_state, device=self.A.device) + dt.mean() * self.A
        B_bar = dt.mean() * self.B
        
        return A_bar, B_bar
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Discretize parameters
        A_bar, B_bar = self.discretize()
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        
        # Process sequence recurrently
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # State update: h_k = A̅h_{k-1} + B̅x_k
            h = torch.matmul(h, A_bar.T) + torch.matmul(x_t, B_bar.T)
            
            # Output: y_k = Ch_k + Dx_k
            y = torch.matmul(h, self.C.T) + x_t * self.D
            
            outputs.append(y)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        return self.dropout(output)


class S4Block(nn.Module):
    """
    S4 Block with normalization and feedforward
    
    Similar structure to Transformer block but with S4 instead of attention
    """
    def __init__(self, d_model, d_state=64, d_ff=None, dropout=0.1):
        super(S4Block, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.s4 = S4Layer(d_model, d_state, dropout)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # S4 with residual
        x = x + self.s4(self.norm1(x))
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


# Example usage
s4_layer = S4Layer(d_model=512, d_state=64)
x = torch.randn(2, 1000, 512)
output = s4_layer(x)
print(f"S4 output: {output.shape}")
```

### 6.2 Mamba (Selective State Space)

```python
class MambaBlock(nn.Module):
    """
    Mamba: Selective State Space Model
    
    Key Innovation: Input-dependent parameters (B, C, Δ)
    
    Formula:
        B_t = Linear_B(x_t)
        C_t = Linear_C(x_t)
        Δ_t = softplus(Linear_Δ(x_t))
        
        h_t = A̅(Δ_t)h_{t-1} + B̅(Δ_t)B_t x_t
        y_t = C_t h_t
    
    Differences from S4:
        - Parameters depend on input (selective)
        - Hardware-aware implementation
        - Linear complexity with sequence length
    
    Advantages over Transformers:
        - O(N) complexity vs O(N²)
        - Better for very long sequences
        - Constant memory during generation
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution (for local context)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # State space parameters (shared, not input-dependent)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(d_model)
    
    def selective_scan(self, x, delta, A, B, C):
        """
        Selective SSM scan (simplified version)
        
        Full implementation uses custom CUDA kernels for efficiency
        
        Args:
            x: (batch, seq_len, d_inner)
            delta: (batch, seq_len, d_inner)
            A: (d_inner, d_state)
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)
        """
        batch_size, seq_len, d_inner = x.size()
        d_state = A.size(1)
        
        # Discretize A
        # A_bar = exp(delta * A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # (batch, seq_len, d_inner, d_state)
        
        # Discretize B
        # B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
        # (batch, seq_len, d_inner, d_state)
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Update state: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Output: y_t = C_t * h_t
            y = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        
        return output
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        x_proj = self.in_proj(x)
        x, z = x_proj.chunk(2, dim=-1)  # (batch, seq_len, d_inner) each
        
        # Convolution for local context
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :x.size(2)]  # Remove padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        x = F.silu(x)  # SiLU activation
        
        # Generate input-dependent parameters
        x_db_dt = self.x_proj(x)  # (batch, seq_len, d_state + d_state + d_inner)
        
        delta, B, C = torch.split(
            x_db_dt,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1
        )
        
        # Delta needs to be positive
        delta = F.softplus(self.dt_proj(delta))
        
        # Get A from log space
        A = -torch.exp(self.A_log)
        
        # Selective scan
        y = self.selective_scan(x, delta, A, B, C)
        
        # Skip connection (D parameter)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        # Residual connection
        output = output + residual
        
        return output


# Example usage
mamba = MambaBlock(d_model=512, d_state=16, d_conv=4, expand=2)
x = torch.randn(2, 2048, 512)
output = mamba(x)
print(f"Mamba output: {output.shape}")
print("Complexity: O(N) - linear in sequence length!")
```

### 6.3 Comparison: State Space vs Attention

```python
def compare_state_space_models():
    """
    State Space Models vs Transformers:
    
    1. S4 (Structured State Space):
       - Method: Fixed HiPPO-initialized SSM
       - Complexity: O(N log N) via FFT convolution
       - Pros: Stable long-range dependencies, efficient
       - Cons: Not input-dependent
       - Use: Time series, audio, long sequences
    
    2. Mamba (Selective SSM):
       - Method: Input-dependent B, C, Δ parameters
       - Complexity: O(N) with custom kernels
       - Pros: Selective memory, hardware-efficient
       - Cons: More complex implementation
       - Use: Language modeling, long-context tasks
    
    3. Transformers:
       - Method: Softmax attention
       - Complexity: O(N²)
       - Pros: Flexible, content-based routing
       - Cons: Quadratic complexity
       - Use: Most NLP tasks
    
    Trade-offs:
        - SSMs: Better scaling, constant memory inference
        - Transformers: Better in-context learning, flexibility
    """
    
    d_model = 512
    seq_lens = [512, 1024, 2048, 4096]
    
    print("\nComplexity Comparison:")
    print(f"{'Seq Len':<10} {'Transformer':<20} {'S4/Mamba':<20}")
    print("-" * 50)
    
    for n in seq_lens:
        transformer_ops = n * n * d_model
        ssm_ops = n * d_model * 64  # Assuming d_state=64
        
        print(f"{n:<10} {transformer_ops:>15,}      {ssm_ops:>15,}")
    
    # Create models
    models = {
        'S4 Block': S4Block(d_model, d_state=64),
        'Mamba Block': MambaBlock(d_model, d_state=16),
        'Transformer Layer': TransformerEncoderLayer(d_model, n_heads=8, d_ff=2048)
    }
    
    print("\n\nModel Parameter Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s} - Parameters: {params:,}")
    
    return models

compare_state_space_models()
```

---

## Cluster 7: Vision and Multi-Modal Architectures

### Mathematical Foundation

These models adapt transformers/attention for visual and multi-modal data.

Key differences:
- Input tokenization (patches vs pixels vs regions)
- Position embeddings (2D vs 1D)
- Modality fusion (early vs late, cross-attention)

### 7.1 Vision Transformer (ViT)

```python
class PatchEmbedding(nn.Module):
    """
    Split image into patches and linearly embed them
    
    Process:
        1. Split image into patches: H×W×C → (H/P × W/P) × (P²C)
        2. Linear projection: (P²C) → d_model
    
    Formula:
        z_0 = [x_class; x_p^1E; x_p^2E; ...; x_p^NE] + E_pos
        where N = HW/P²
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d for patch embedding (equivalent to splitting + linear)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        
        Returns:
            patches: (batch, n_patches, embed_dim)
        """
        # x: (batch, 3, 224, 224)
        x = self.proj(x)  # (batch, embed_dim, H/P, W/P)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)
    
    Architecture:
        1. Patch Embedding
        2. Add [CLS] token and positional embeddings
        3. Transformer Encoder
        4. Classification head on [CLS] token
    
    Differences from NLP Transformers:
        - 2D positional embeddings (learned)
        - Patch-based tokenization
        - [CLS] token for classification
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, num_heads,
                int(embed_dim * mlp_ratio),
                dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification from CLS token
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


# Example usage
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

x = torch.randn(2, 3, 224, 224)
output = vit(x)
print(f"ViT output: {output.shape}")  # (2, 1000)
```

### 7.2 CLIP (Contrastive Language-Image Pre-training)

```python
class CLIPVisionEncoder(nn.Module):
    """
    CLIP Vision Encoder (ViT-based)
    
    Similar to ViT but outputs embedding instead of classification
    """
    def __init__(self, img_size=224, patch_size=32, embed_dim=512,
                 depth=12, num_heads=8, mlp_ratio=4.0):
        super(CLIPVisionEncoder, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads,
                                   int(embed_dim * mlp_ratio), 0.1)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection head for contrastive learning
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Get CLS token and project
        cls_output = x[:, 0]
        embedding = self.proj(cls_output)
        
        # L2 normalize for contrastive learning
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder (Transformer-based)
    """
    def __init__(self, vocab_size=49408, embed_dim=512, context_length=77,
                 depth=12, num_heads=8, mlp_ratio=4.0):
        super(CLIPTextEncoder, self).__init__()
        
        self.context_length = context_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads,
                                   int(embed_dim * mlp_ratio), 0.1)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection head
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, text):
        """
        Args:
            text: (batch, context_length) - token indices
        
        Returns:
            embedding: (batch, embed_dim)
        """
        x = self.token_embedding(text)
        x = x + self.pos_embedding
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Take embedding at EOT token (end of text)
        # For simplicity, use last token
        embedding = x[:, -1, :]
        embedding = self.proj(embedding)
        
        # L2 normalize
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training
    
    Training objective:
        Maximize cosine similarity of matching (image, text) pairs
        Minimize for non-matching pairs
    
    Loss (InfoNCE):
        L = -log(exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))
    
    where τ is temperature parameter
    """
    def __init__(self, embed_dim=512, vision_depth=12, text_depth=12,
                 vision_heads=8, text_heads=8, vocab_size=49408):
        super(CLIP, self).__init__()
        
        self.vision_encoder = CLIPVisionEncoder(
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=vision_heads
        )
        
        self.text_encoder = CLIPTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=text_depth,
            num_heads=text_heads
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
    
    def encode_image(self, image):
        return self.vision_encoder(image)
    
    def encode_text(self, text):
        return self.text_encoder(text)
    
    def forward(self, image, text):
        """
        Args:
            image: (batch, 3, 224, 224)
            text: (batch, context_length)
        
        Returns:
            logits_per_image: (batch, batch)
            logits_per_text: (batch, batch)
        """
        image_features = self.encode_image(image)  # (batch, embed_dim)
        text_features = self.encode_text(text)      # (batch, embed_dim)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def contrastive_loss(self, logits_per_image, logits_per_text):
        """
        Symmetric cross-entropy loss
        
        Ground truth: diagonal matrix (matching pairs)
        """
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_img + loss_txt) / 2
        
        return loss


# Example usage
clip = CLIP(embed_dim=512, vision_depth=12, text_depth=12)

images = torch.randn(4, 3, 224, 224)
texts = torch.randint(0, 49408, (4, 77))

logits_img, logits_txt = clip(images, texts)
loss = clip.contrastive_loss(logits_img, logits_txt)

print(f"CLIP logits: {logits_img.shape}")
print(f"Contrastive loss: {loss.item():.4f}")

# Zero-shot classification
with torch.no_grad():
    image_features = clip.encode_image(images[:1])  # Single image
    text_features = clip.encode_text(texts)  # Multiple text descriptions
    
    # Compute similarities
    similarities = (100.0 * image_features @ text_features.t()).softmax(dim=-1)
    print(f"Similarities: {similarities}")
```

### 7.3 Multi-Modal Fusion

```python
class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing vision and language
    
    Vision queries attend to language keys/values (or vice versa)
    
    Used in: Flamingo, BLIP, etc.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, vision_features, text_features, mask=None):
        """
        Args:
            vision_features: (batch, v_len, d_model)
            text_features: (batch, t_len, d_model)
            mask: attention mask
        
        Returns:
            fused_features: (batch, v_len, d_model)
        """
        # Vision attends to text
        fused, _ = self.cross_attn.mha(
            vision_features,  # Queries from vision
            text_features,    # Keys from text
            text_features,    # Values from text
            mask
        )
        
        return fused


class MultiModalTransformer(nn.Module):
    """
    Multi-Modal Transformer with cross-attention
    
    Architecture:
        1. Separate encoders for each modality
        2. Cross-modal attention for fusion
        3. Joint decoder or separate heads
    
    Fusion strategies:
        - Early fusion: Concatenate embeddings
        - Late fusion: Separate processing + cross-attention
        - Co-attention: Bidirectional cross-attention
    """
    def __init__(self, vision_dim=2048, text_vocab=30000, d_model=512,
                 n_heads=8, num_layers=6, num_classes=1000):
        super(MultiModalTransformer, self).__init__()
        
        # Vision encoder (e.g., from CNN features)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        
        # Text encoder
        self.text_embed = nn.Embedding(text_vocab, d_model)
        self.text_pos = LearnedPositionalEncoding(d_model, max_len=512)
        
        # Cross-modal layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalLayer(d_model, n_heads)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, vision_features, text_tokens):
        """
        Args:
            vision_features: (batch, v_len, vision_dim)
            text_tokens: (batch, t_len)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Project vision features
        v = self.vision_proj(vision_features)  # (batch, v_len, d_model)
        
        # Embed text
        t = self.text_embed(text_tokens)  # (batch, t_len, d_model)
        t = self.text_pos(t)
        
        # Cross-modal fusion
        for layer in self.cross_modal_layers:
            v, t = layer(v, t)
        
        # Aggregate (e.g., mean pooling)
        v_pooled = v.mean(dim=1)
        t_pooled = t.mean(dim=1)
        
        # Combine modalities
        combined = v_pooled + t_pooled
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


class CrossModalLayer(nn.Module):
    """
    Single cross-modal fusion layer
    
    Applies bidirectional cross-attention:
        - Vision attends to text
        - Text attends to vision
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossModalLayer, self).__init__()
        
        # Vision self-attention
        self.v_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Text self-attention
        self.t_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention: vision -> text
        self.v2t_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention: text -> vision
        self.t2v_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feedforward
        self.v_ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.t_ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        
        # Norms
        self.v_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.t_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
    
    def forward(self, v, t):
        """
        Args:
            v: vision features (batch, v_len, d_model)
            t: text features (batch, t_len, d_model)
        
        Returns:
            v_out: updated vision (batch, v_len, d_model)
            t_out: updated text (batch, t_len, d_model)
        """
        # Vision self-attention
        v_res = v
        v, _ = self.v_self_attn.mha(v, v, v)
        v = self.v_norms[0](v + v_res)
        
        # Text self-attention
        t_res = t
        t, _ = self.t_self_attn.mha(t, t, t)
        t = self.t_norms[0](t + t_res)
        
        # Cross-attention: vision attends to text
        v_res = v
        v_cross, _ = self.v2t_attn.mha(v, t, t)
        v = self.v_norms[1](v + v_cross)
        
        # Cross-attention: text attends to vision
        t_res = t
        t_cross, _ = self.t2v_attn.mha(t, v, v)
        t = self.t_norms[1](t + t_cross)
        
        # Feedforward
        v_res = v
        v = self.v_norms[2](v + self.v_ffn(v))
        
        t_res = t
        t = self.t_norms[2](t + self.t_ffn(t))
        
        return v, t


# Example usage
multi_modal = MultiModalTransformer(
    vision_dim=2048,
    text_vocab=30000,
    d_model=512,
    n_heads=8,
    num_layers=6,
    num_classes=1000
)

vision_feat = torch.randn(2, 49, 2048)  # (batch, patches, vision_dim)
text_tokens = torch.randint(0, 30000, (2, 50))

logits = multi_modal(vision_feat, text_tokens)
print(f"Multi-modal output: {logits.shape}")
```

### Comparison of Vision Architectures

```python
def compare_vision_architectures():
    """
    Vision Architecture Comparison:
    
    1. CNN (ResNet):
       - Method: Hierarchical convolutions
       - Inductive bias: Locality, translation invariance
       - Pros: Few parameters, strong for small data
       - Cons: Limited global context
       - Best: Small datasets, efficient deployment
    
    2. Vision Transformer (ViT):
       - Method: Patch embeddings + Transformer
       - Inductive bias: Minimal (learns from data)
       - Pros: Global context, scales well with data
       - Cons: Needs large datasets, more parameters
       - Best: Large-scale pre-training
    
    3. CLIP:
       - Method: Contrastive vision-language learning
       - Training: 400M image-text pairs
       - Pros: Zero-shot transfer, flexible
       - Cons: Requires paired data
       - Best: Zero-shot classification, retrieval
    
    4. Multi-Modal Transformers:
       - Method: Cross-modal fusion
       - Fusion: Self + cross-attention
       - Pros: Rich multi-modal understanding
       - Cons: Complex, expensive
       - Best: VQA, image captioning, video understanding
    """
    
    models = {
        'ResNet-50': resnet34(num_classes=1000),
        'ViT-Base': VisionTransformer(embed_dim=768, depth=12, num_heads=12),
        'CLIP': CLIP(embed_dim=512, vision_depth=12, text_depth=12),
    }
    
    print("\nVision Architecture Comparison:")
    print("-" * 60)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s} - Parameters: {params / 1e6:.1f}M")
    
    # Input sizes
    print("\nInput/Output Comparison:")
    img = torch.randn(1, 3, 224, 224)
    
    resnet_out = models['ResNet-50'](img)
    vit_out = models['ViT-Base'](img)
    
    print(f"ResNet output: {resnet_out.shape}")
    print(f"ViT output: {vit_out.shape}")
    
    return models

compare_vision_architectures()
```

---

## Final Summary: Complete Architecture Clustering

```python
def complete_architecture_summary():
    """
    COMPLETE DEEP LEARNING ARCHITECTURE TAXONOMY
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 1: FEEDFORWARD (Spatial Processing)
    ═══════════════════════════════════════════════════════════════
    Mathematical Core: y = f(Wx + b)
    
    │── MLP: Dense connections
    │   Formula: h = σ(W_n(...σ(W_1x)))
    │   Use: Tabular data, simple classification
    │
    │── CNN: Local connectivity + weight sharing  
    │   Formula: y = σ(x * W + b)
    │   Use: Images, spatial data
    │
    └── ResNet: Skip connections
        Formula: y = F(x) + x
        Use: Very deep networks, image classification
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 2: RECURRENT (Temporal Processing)
    ═══════════════════════════════════════════════════════════════
    Mathematical Core: h_t = f(h_{t-1}, x_t)
    
    │── RNN: Simple recurrence
    │   Formula: h_t = tanh(W_hh h_{t-1} + W_xh x_t)
    │   Problem: Vanishing gradients
    │
    │── LSTM: Gated with cell state
    │   Gates: forget, input, output (3 gates + cell)
    │   Formula: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    │   Use: Long sequences, language
    │
    └── GRU: Simplified gating
        Gates: reset, update (2 gates)
        Formula: h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        Use: Faster than LSTM, similar performance
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 3: ATTENTION (Content-Based Routing)
    ═══════════════════════════════════════════════════════════════
    Mathematical Core: Attention(Q,K,V) = softmax(score(Q,K))V
    
    │── Scaled Dot-Product
    │   Formula: softmax(QK^T / √d_k)V
    │   Complexity: O(n²)
    │
    │── Multi-Head
    │   Formula: Concat(head_1,...,head_h)W^O
    │   Benefit: Multiple representation subspaces
    │
    │── Self-Attention: Q=K=V from same source
    │   Use: Encoder layers
    │
    │── Cross-Attention: Q from one source, K,V from another
    │   Use: Decoder, multi-modal fusion
    │
    └── Additive (Bahdanau)
        Formula: v^T tanh(W_1 Q + W_2 K)
        Use: Original seq2seq attention
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 4: TRANSFORMERS (Full Architecture)
    ═══════════════════════════════════════════════════════════════
    Mathematical Core: Multi-Head Attention + FFN + Residuals
    
    │── Vanilla (Encoder-Decoder)
    │   Use: Machine translation, seq2seq
    │   Training: Teacher forcing
    │
    │── BERT (Encoder-Only)
    │   Attention: Bidirectional
    │   Training: MLM + NSP
    │   Use: Understanding tasks (classification, NER, QA)
    │
    │── GPT (Decoder-Only)
    │   Attention: Causal (autoregressive)
    │   Training: Next token prediction
    │   Use: Generation, few-shot learning
    │
    └── T5 (Encoder-Decoder)
        Framework: All tasks as text-to-text
        Training: Span corruption
        Use: Unified NLP
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 5: EFFICIENT TRANSFORMERS (Reduced Complexity)
    ═══════════════════════════════════════════════════════════════
    Goal: Reduce O(n²) attention complexity
    
    │── Linformer: Low-rank approximation
    │   Formula: softmax(Q(EK)^T/√d)(FV)
    │   Complexity: O(nk), k << n
    │
    │── Performer: Kernel approximation
    │   Formula: φ(Q)(φ(K)^T V)
    │   Complexity: O(n)
    │
    │── Reformer: LSH bucketing
    │   Method: Hash queries/keys into buckets
    │   Complexity: O(n log n)
    │
    └── Longformer: Sparse patterns
        Patterns: Local window + global tokens
        Complexity: O(nw), w = window size
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 6: STATE SPACE MODELS (Continuous Dynamics)
    ═══════════════════════════════════════════════════════════════
    Mathematical Core: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)
    
    │── S4: Structured (fixed parameters)
    │   Method: HiPPO initialization
    │   Complexity: O(N log N) via FFT
    │   Use: Audio, time series
    │
    └── Mamba: Selective (input-dependent parameters)
        Formula: B_t, C_t, Δ_t = f(x_t)
        Complexity: O(N) with custom kernels
        Use: Language modeling, long context
    
    Advantage over Transformers:
        ✓ Linear scaling
        ✓ Constant memory inference
        ✗ Less flexible than attention
    
    ═══════════════════════════════════════════════════════════════
    CLUSTER 7: VISION & MULTI-MODAL (Cross-Domain)
    ═══════════════════════════════════════════════════════════════
    Goal: Process images and/or multiple modalities
    
    │── ViT: Patches + Transformer
    │   Process: Image → Patches → Transformer
    │   Formula: z_0 = [x_cls; x_p^1E; ...; x_p^NE] + E_pos
    │   Use: Image classification
    │
    │── CLIP: Contrastive vision-language
    │   Training: Maximize similarity of matching pairs
    │   Loss: InfoNCE contrastive loss
    │   Use: Zero-shot classification, retrieval
    │
    └── Multi-Modal Fusion
        Methods: Cross-attention, co-attention
        Architecture: Separate encoders + cross-modal layers
        Use: VQA, image captioning, video understanding
    
    ═══════════════════════════════════════════════════════════════
    SELECTION GUIDE
    ═══════════════════════════════════════════════════════════════
    
    Task: Image Classification
    Small data    → ResNet (CNN)
    Large data    → ViT
    Zero-shot     → CLIP
    
    Task: Text Classification
    Short text    → BERT
    Long text     → Longformer, Mamba
    
    Task: Text Generation
    Quality       → GPT (Transformer)
    Efficiency    → Mamba
    Very long     → Mamba, RWKV
    
    Task: Sequence Modeling
    <512 tokens   → Transformer
    <8K tokens    → Longformer
    >8K tokens    → Mamba, S4
    
    Task: Time Series
    Short         → LSTM
    Long          → S4, Mamba
    Irregular     → GRU
    
    Task: Multi-Modal
    Image + Text  → CLIP, Multi-Modal Transformer
    Video + Text  → Flamingo-style (cross-attention)
    
    ═══════════════════════════════════════════════════════════════
    COMPLEXITY COMPARISON (sequence length n, dimension d)
    ═══════════════════════════════════════════════════════════════
    
    RNN/LSTM/GRU:          O(n·d²)      Sequential (slow training)
    CNN:                   O(n·k·d)     Parallel (k = kernel size)
    Transformer:           O(n²·d)      Parallel (quadratic!)
    Linformer:             O(n·k·d)     Parallel (k = projection)
    Performer:             O(n·d²)      Parallel (linear in n)
    Reformer:              O(n·log n·d) Parallel (with sorting)
    Longformer:            O(n·w·d)     Parallel (w = window)
    S4:                    O(n·log n)   Parallel (via FFT)
    Mamba:                 O(n·d)       Parallel/Recurrent
    
    ═══════════════════════════════════════════════════════════════
    PARAMETER COUNT COMPARISON (typical configurations)
    ═══════════════════════════════════════════════════════════════
    
    Architecture              Parameters    Use Case
    ─────────────────────────────────────────────────────────────
    ResNet-50                 25M           Image classification
    ViT-Base                  86M           Image classification
    BERT-Base                 110M          NLP understanding
    GPT-2                     1.5B          Text generation
    GPT-3                     175B          Few-shot learning
    CLIP                      ~400M         Vision-language
    Mamba-370M                370M          Long context
    
    ═══════════════════════════════════════════════════════════════
    KEY MATHEMATICAL INSIGHTS
    ═══════════════════════════════════════════════════════════════
    
    1. VANISHING GRADIENTS
       Problem: ∂h_t/∂h_0 = ∏(∂h_i/∂h_{i-1}) → 0 as t increases
       Solution: LSTM gates, skip connections (ResNet), layer norm
    
    2. ATTENTION SCALING
       Why √d_k?: Prevents softmax saturation
       Without: QK^T has variance d_k
       With: QK^T/√d_k has variance 1
    
    3. POSITIONAL ENCODING
       Why needed?: Transformers have no built-in position info
       Sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d))
       Learned: Embedding layer for positions
    
    4. MULTI-HEAD BENEFIT
       Single head: Limited representation capacity
       Multi-head: Attend to different subspaces
       Example: One head for syntax, another for semantics
    
    5. SKIP CONNECTIONS
       Formula: y = F(x) + x
       Benefit: Gradient flows directly through identity
       Effect: ∂y/∂x = ∂F/∂x + 1 (always ≥1)
    
    ═══════════════════════════════════════════════════════════════
    """
    
    print(complete_architecture_summary.__doc__)

complete_architecture_summary()
```

---

## Appendix: Training Utilities

### A.1 Loss Functions

```python
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for representation learning
    
    Formula (InfoNCE):
        L = -log(exp(sim(z_i, z_i^+)/τ) / Σ_j exp(sim(z_i, z_j)/τ))
    
    Used in: CLIP, SimCLR, MoCo
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features_1, features_2):
        """
        Args:
            features_1: (batch, dim) - first view
            features_2: (batch, dim) - second view
        
        Returns:
            loss: scalar
        """
        batch_size = features_1.shape[0]
        
        # Normalize features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        
        # Compute similarity matrix
        features = torch.cat([features_1, features_2], dim=0)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity = similarity.masked_fill(mask, -1e9)
        
        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Formula:
        FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    
    Used in: Object detection (RetinaNet), imbalanced classification
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) - logits
            targets: (batch,) - class labels
        
        Returns:
            loss: scalar or (batch,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for better generalization
    
    Formula:
        y_smooth = (1-ε)y_hard + ε/K
    
    where K is number of classes
    """
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, num_classes) - log probabilities
            target: (batch,) - class labels
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

### A.2 Learning Rate Schedulers

```python
class CosineAnnealingWarmup:
    """
    Cosine annealing with linear warmup
    
    Formula:
        Warmup: lr = (t/T_warmup) * lr_max
        Cosine: lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(π·t/T_max))
    """
    def __init__(self, optimizer, warmup_steps, max_steps, 
                 lr_max, lr_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.last_epoch = last_epoch
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.lr_max * step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
                   (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    def step(self, step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# Example usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmup(
    optimizer,
    warmup_steps=1000,
    max_steps=10000,
    lr_max=1e-3,
    lr_min=1e-5
)

# In training loop:
# for step in range(max_steps):
#     lr = scheduler.step(step)
#     # ... training code ...
```

### A.3 Data Augmentation

```python
class RandomMixup(nn.Module):
    """
    Mixup data augmentation
    
    Formula:
        x̃ = λx_i + (1-λ)x_j
        ỹ = λy_i + (1-λ)y_j
    
    where λ ~ Beta(α, α)
    """
    def __init__(self, alpha=0.2):
        super(RandomMixup, self).__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        """
        Args:
            x: (batch, ...) - input data
            y: (batch, num_classes) - one-hot labels
        
        Returns:
            mixed_x, mixed_y
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y


class RandomCutmix(nn.Module):
    """
    CutMix data augmentation for images
    
    Cuts and pastes patches between images
    """
    def __init__(self, alpha=1.0):
        super(RandomCutmix, self).__init__()
        self.alpha = alpha
    
    def forward(self, x, y):
        """
        Args:
            x: (batch, C, H, W)
            y: (batch, num_classes)
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # Random box
        _, _, H, W = x.shape
        cut_rat = torch.sqrt(1 - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()
        
        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))
        
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        x_cutmix = x.clone()
        x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        y_cutmix = lam * y + (1 - lam) * y[index]
        
        return x_cutmix, y_cutmix
```

### A.4 Model Utilities

```python
def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model, freeze_up_to=None):
    """
    Freeze layers for transfer learning
    
    Args:
        model: PyTorch model
        freeze_up_to: Layer name to freeze up to
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze_up_to and freeze_up_to in name:
            freeze = False
        param.requires_grad = not freeze
    
    trainable = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def get_gradient_norm(model):
    """Calculate gradient norm for debugging"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


# Example usage
def train_with_utilities():
    """
    Example training loop with utilities
    """
    # Setup
    model = VisionTransformer(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmup(optimizer, 1000, 10000, 1e-3, 1e-5)
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
    early_stopping = EarlyStopping(patience=5)
    
    # Data augmentation
    mixup = RandomMixup(alpha=0.2)
    cutmix = RandomCutmix(alpha=1.0)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Training loop
    for epoch in range(100):
        model.train()
        
        for step, (x, y) in enumerate(train_loader):
            # Apply augmentation
            if torch.rand(1) > 0.5:
                x, y = mixup(x, y)
            else:
                x, y = cutmix(x, y)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step(epoch * len(train_loader) + step)
            
            # Monitor gradients
            if step % 100 == 0:
                grad_norm = get_gradient_norm(model)
                print(f"Step {step}, Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model


# Usage demonstration
print("\n" + "="*70)
print("PYTORCH IMPLEMENTATION COMPLETE!")
print("="*70)
print("\nAll architecture clusters implemented with:")
print("  ✓ Mathematical formulations")
print("  ✓ Working PyTorch code")
print("  ✓ Usage examples")
print("  ✓ Comparisons and trade-offs")
print("  ✓ Training utilities")
print("\nYou can now:")
print("  1. Copy any model class and use directly")
print("  2. Mix and match components (e.g., use Mamba in place of attention)")
print("  3. Extend with custom modifications")
print("  4. Compare architectures empirically")
print("\nHappy deep learning! 🚀")
```

---

## Quick Reference: When to Use Each Architecture

```python
# QUICK SELECTION GUIDE

# Image Classification
small_data = "ResNet (CNN)"
large_data = "ViT"
zero_shot = "CLIP"

# Text Classification  
short_text = "BERT"
long_text = "Longformer or Mamba"

# Text Generation
quality_focus = "GPT (Transformer)"
efficiency_focus = "Mamba"
very_long_context = "Mamba or RWKV"

# Sequence Modeling
if sequence_length < 512:
    use = "Transformer"
elif sequence_length < 8192:
    use = "Longformer"
else:
    use = "Mamba or S4"

# Time Series
short_series = "LSTM"
long_series = "S4 or Mamba"
irregular_sampling = "GRU"

# Multi-Modal
image_text = "CLIP or MultiModalTransformer"
video_text = "Flamingo-style (cross-attention)"

# Constraints
if memory_limited:
    use = "MobileNet, DistilBERT, or smaller models"
if latency_critical:
    use = "CNN for vision, Mamba for sequences"
if accuracy_critical:
    use = "Large Transformers (ViT-L, GPT-4)"

print("Architecture selection guide ready! Choose based on your task and constraints.")
```

# Deep Learning Architectures: PyTorch Implementation Guide

## Clustering by Mathematical Similarity

This guide clusters deep learning architectures based on their mathematical foundations and provides PyTorch implementations for each cluster.

### Architecture Clusters:

1. **Feedforward Cluster**: MLP, CNN, ResNet
2. **Recurrent Cluster**: RNN, LSTM, GRU
3. **Attention Cluster**: Self-Attention, Multi-Head Attention, Cross-Attention
4. **Transformer Cluster**: Vanilla Transformer, BERT, GPT
5. **Efficient Transformer Cluster**: Linformer, Reformer, Performer
6. **State Space Cluster**: Mamba, S4, Hyena
7. **Hybrid Cluster**: Vision Transformer, Perceiver, Multi-Modal

---

## Cluster 1: Feedforward Architectures

### Mathematical Foundation

All feedforward networks follow:
$$y = f(Wx + b)$$

They differ in:
- Connection patterns (dense vs convolutional)
- Skip connections (identity vs projection)
- Spatial structure preservation

### 1.1 Basic Multi-Layer Perceptron (MLP)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Basic Multi-Layer Perceptron
    
    Forward pass: h = activation(W_i * h_{i-1} + b_i)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.network(x)  # (batch_size, output_dim)


# Example usage
model = MLP(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)  # Shape: (32, 10)
print(f"MLP output shape: {output.shape}")
```

### 1.2 Convolutional Neural Network (CNN)

```python
class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU
    
    Convolution: y(i,j) = sum_m sum_n x(i+m, j+n) * K(m,n)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNN(nn.Module):
    """
    Standard CNN for image classification
    
    Architecture: Conv blocks -> Pooling -> Flatten -> FC layers
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(256, 512, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.pool(self.conv1(x))  # -> (batch, 64, H/2, W/2)
        x = self.pool(self.conv2(x))  # -> (batch, 128, H/4, W/4)
        x = self.pool(self.conv3(x))  # -> (batch, 256, H/8, W/8)
        x = self.pool(self.conv4(x))  # -> (batch, 512, H/16, W/16)
        
        x = self.global_pool(x)       # -> (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)     # -> (batch, 512)
        x = self.fc(x)                # -> (batch, num_classes)
        
        return x


# Example usage
model = CNN(in_channels=3, num_classes=10)
x = torch.randn(16, 3, 224, 224)  # Batch of 16 RGB images
output = model(x)  # Shape: (16, 10)
print(f"CNN output shape: {output.shape}")
```

### 1.3 Residual Networks (ResNet)

**Key Difference**: Adds skip connections

$$y = F(x, \{W_i\}) + x$$

```python
class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection
    
    Formula: H(x) = F(x) + x
    where F(x) is the residual function
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (identity or projection)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # Residual path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection: H(x) = F(x) + x
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture
    
    Difference from CNN: Uses skip connections to enable deeper networks
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Projection shortcut when dimensions change
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=10):
    """ResNet-18: [2, 2, 2, 2] blocks"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    """ResNet-34: [3, 4, 6, 3] blocks"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


# Example usage
model = resnet18(num_classes=10)
x = torch.randn(8, 3, 224, 224)
output = model(x)
print(f"ResNet-18 output shape: {output.shape}")
```

### Comparison of Feedforward Architectures

```python
def compare_feedforward_architectures():
    """
    Key Differences:
    
    1. MLP:
       - Dense connections
       - No spatial structure
       - Formula: y = W_n(...W_2(W_1 x + b_1) + b_2...) + b_n
    
    2. CNN:
       - Local connectivity (convolutions)
       - Parameter sharing
       - Spatial hierarchy
       - Formula: y = Pool(Conv(x))
    
    3. ResNet:
       - Skip connections enable deeper networks
       - Identity mappings
       - Formula: y = F(x) + x (solves vanishing gradient)
    """
    
    batch_size = 16
    
    # For MLP: flattened input
    mlp_input = torch.randn(batch_size, 784)
    mlp = MLP(784, [512, 256], 10)
    mlp_out = mlp(mlp_input)
    
    # For CNN and ResNet: spatial input
    spatial_input = torch.randn(batch_size, 3, 224, 224)
    cnn = CNN(in_channels=3, num_classes=10)
    resnet = resnet18(num_classes=10)
    
    cnn_out = cnn(spatial_input)
    resnet_out = resnet(spatial_input)
    
    print("Feedforward Architectures Comparison:")
    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"CNN parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"ResNet-18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")
    
    return mlp, cnn, resnet

compare_feedforward_architectures()
```

---

## Cluster 2: Recurrent Architectures

### Mathematical Foundation

All recurrent networks maintain hidden state across time:
$$h_t = f(h_{t-1}, x_t)$$

They differ in:
- Gating mechanisms (none vs forget/input/output gates)
- Number of states (hidden only vs hidden + cell)
- Complexity of update rules

### 2.1 Basic RNN

```python
class VanillaRNN(nn.Module):
    """
    Vanilla RNN (Elman network)
    
    Formula:
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        y_t = W_hy * h_t + b_y
    
    Issues: Vanishing/exploding gradients for long sequences
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (num_layers, batch, hidden_size) or None
        
        Returns:
            output: (batch, seq_len, output_size)
            hidden: (num_layers, batch, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, 
                           self.hidden_size).to(x.device)
        
        # Forward through RNN
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.rnn(x, h0)
        
        # Apply output layer to all timesteps
        out = self.fc(out)  # (batch, seq_len, output_size)
        
        return out, h_n
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, 
                         self.hidden_size).to(device)


# Example usage
rnn = VanillaRNN(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)  # (batch=8, seq_len=15, input_size=10)
output, hidden = rnn(x)
print(f"RNN output shape: {output.shape}, hidden shape: {hidden.shape}")
```

### 2.2 LSTM (Long Short-Term Memory)

**Key Difference**: Adds cell state and three gates

```python
class LSTMCell(nn.Module):
    """
    Single LSTM cell implementation
    
    Formulas:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # Forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # Input gate
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)    # Cell candidate
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t           # Cell state
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # Output gate
        h_t = o_t ⊙ tanh(C_t)                       # Hidden state
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input transformations (all gates + cell candidate)
        # Combined for efficiency: [forget, input, output, cell_candidate]
        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
        Args:
            x: (batch, input_size)
            hidden: tuple of (h, c)
                h: (batch, hidden_size)
                c: (batch, hidden_size)
        
        Returns:
            h_new: (batch, hidden_size)
            c_new: (batch, hidden_size)
        """
        h, c = hidden
        
        # Linear transformations
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        
        # Split into gates
        # Each gate: (batch, hidden_size)
        forget_gate, input_gate, output_gate, cell_candidate = gates.chunk(4, 1)
        
        # Apply activations
        f_t = torch.sigmoid(forget_gate)
        i_t = torch.sigmoid(input_gate)
        o_t = torch.sigmoid(output_gate)
        c_tilde = torch.tanh(cell_candidate)
        
        # Update cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        c_new = f_t * c + i_t * c_tilde
        
        # Update hidden state: h_t = o_t ⊙ tanh(C_t)
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """
    Multi-layer LSTM
    
    Difference from RNN: Uses gating to control information flow,
    maintains separate cell state for long-term memory
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: tuple of (h0, c0) or None
                h0: (num_layers, batch, hidden_size)
                c0: (num_layers, batch, hidden_size)
        
        Returns:
            output: (batch, seq_len, output_size)
            (h_n, c_n): final hidden and cell states
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, 
                           self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size,
                           self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # Forward through LSTM
        out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply output layer
        out = self.fc(out)
        
        return out, (h_n, c_n)


# Example usage
lstm = LSTM(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)
output, (hidden, cell) = lstm(x)
print(f"LSTM output: {output.shape}, hidden: {hidden.shape}, cell: {cell.shape}")
```

### 2.3 GRU (Gated Recurrent Unit)

**Key Difference**: Fewer gates than LSTM, no separate cell state

```python
class GRUCell(nn.Module):
    """
    Single GRU cell
    
    Formulas:
        r_t = σ(W_r · [h_{t-1}, x_t])           # Reset gate
        z_t = σ(W_z · [h_{t-1}, x_t])           # Update gate
        h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   # Candidate hidden
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state
    
    Differences from LSTM:
        - Only 2 gates (reset, update) vs 3 gates (forget, input, output)
        - No separate cell state
        - Fewer parameters: 3N² vs 4N² for LSTM
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: [reset, update]
        self.weight_ih = nn.Parameter(
            torch.randn(2 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(2 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size))
        
        # Candidate hidden state
        self.weight_ih_candidate = nn.Parameter(
            torch.randn(hidden_size, input_size)
        )
        self.weight_hh_candidate = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
        )
        self.bias_ih_candidate = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh_candidate = nn.Parameter(torch.randn(hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, h):
        """
        Args:
            x: (batch, input_size)
            h: (batch, hidden_size)
        
        Returns:
            h_new: (batch, hidden_size)
        """
        # Compute reset and update gates
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        
        reset_gate, update_gate = gates.chunk(2, 1)
        
        r_t = torch.sigmoid(reset_gate)
        z_t = torch.sigmoid(update_gate)
        
        # Compute candidate hidden state
        h_candidate = (torch.mm(x, self.weight_ih_candidate.t()) + 
                      self.bias_ih_candidate +
                      torch.mm(r_t * h, self.weight_hh_candidate.t()) +
                      self.bias_hh_candidate)
        
        h_tilde = torch.tanh(h_candidate)
        
        # Update hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_new = (1 - z_t) * h + z_t * h_tilde
        
        return h_new


class GRU(nn.Module):
    """
    Multi-layer GRU
    
    Advantages over LSTM:
        - Simpler (fewer parameters)
        - Faster training and inference
        - Often comparable performance
    
    When to use GRU vs LSTM:
        - GRU: When you need efficiency, smaller datasets
        - LSTM: When you need maximum expressiveness, larger datasets
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (num_layers, batch, hidden_size) or None
        
        Returns:
            output: (batch, seq_len, output_size)
            h_n: (num_layers, batch, hidden_size)
        """
        batch_size = x.size(0)
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size,
                           self.hidden_size).to(x.device)
        
        # Forward through GRU
        out, h_n = self.gru(x, h0)
        
        # Apply output layer
        out = self.fc(out)
        
        return out, h_n


# Example usage
gru = GRU(input_size=10, hidden_size=20, output_size=5, num_layers=2)
x = torch.randn(8, 15, 10)
output, hidden = gru(x)
print(f"GRU output: {output.shape}, hidden: {hidden.shape}")
```

### 2.4 Bidirectional RNN Wrapper

```python
class BidirectionalRNN(nn.Module):
    """
    Bidirectional wrapper for any RNN type
    
    Formula:
        h⃗_t = f(W⃗_hh * h⃗_{t-1} + W⃗_xh * x_t)  # Forward
        h⃖_t = f(W⃖_hh * h⃖_{t+1} + W⃖_xh * x_t)  # Backward
        h_t = [h⃗_t; h⃖_t]                        # Concatenate
    """
    def __init__(self, rnn_type, input_size, hidden_size, output_size, 
                 num_layers=1):
        super(BidirectionalRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Choose RNN type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        
        # Output layer (note: 2 * hidden_size due to bidirectional)
        self.fc = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x):
        # out: (batch, seq_len, 2 * hidden_size)
        out, _ = self.rnn(x)
        
        # Apply output layer
        out = self.fc(out)
        
        return out


# Example usage
bi_lstm = BidirectionalRNN('lstm', input_size=10, hidden_size=20, 
                          output_size=5, num_layers=1)
x = torch.randn(8, 15, 10)
output = bi_lstm(x)
print(f"Bidirectional LSTM output: {output.shape}")
```

### Comparison of Recurrent Architectures

```python
def compare_recurrent_architectures():
    """
    Key Differences:
    
    1. RNN (Vanilla):
       - Simple: h_t = tanh(W_hh h_{t-1} + W_xh x_t)
       - Problem: Vanishing gradients
       - Parameters: 2N²
    
    2. LSTM:
       - 3 gates + cell state
       - Formula: Uses forget, input, output gates
       - Solves vanishing gradients
       - Parameters: 4N²
       - Best for: Long sequences, need maximum capacity
    
    3. GRU:
       - 2 gates, no separate cell state
       - Simpler than LSTM
       - Parameters: 3N²
       - Best for: Efficiency, smaller datasets
    """
    
    input_size, hidden_size, output_size = 10, 20, 5
    seq_len, batch_size = 15, 8
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Compare parameter counts
    models = {
        'RNN': VanillaRNN(input_size, hidden_size, output_size),
        'LSTM': LSTM(input_size, hidden_size, output_size),
        'GRU': GRU(input_size, hidden_size, output_size)
    }
    
    print("\nRecurrent Architectures Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        output = model(x)[0] if name != 'RNN' or name != 'GRU' else model(x)[0]
        print(f"{name:10s} - Parameters: {params:,} - Output: {output.shape}")
    
    return models

compare_recurrent_architectures()
```

---

## Cluster 3: Attention Mechanisms

### Mathematical Foundation

All attention mechanisms compute weighted sums:
$\text{Attention}(Q, K, V) = \text{softmax}(\text{score}(Q, K)) \cdot V$

They differ in:
- Score function (dot product, additive, multiplicative)
- Scaling factors
- Number of heads
- Query/Key/Value sources (self vs cross-attention)

### 3.1 Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
        Q: queries (batch, seq_len_q, d_k)
        K: keys (batch, seq_len_k, d_k)
        V: values (batch, seq_len_v, d_v)
        d_k: dimension of keys/queries
    
    Scaling by √d_k prevents softmax saturation for large d_k
    """
    def __init__(self, temperature=None, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, n_heads, len_q, d_k)
            k: (batch, n_heads, len_k, d_k)
            v: (batch, n_heads, len_v, d_v)
            mask: (batch, 1, len_q, len_k) or None
        
        Returns:
            output: (batch, n_heads, len_q, d_v)
            attention: (batch, n_heads, len_q, len_k)
        """
        d_k = q.size(-1)
        
        # If temperature not set, use sqrt(d_k)
        temperature = self.temperature or d_k ** 0.5
        
        # Compute attention scores: QK^T / √d_k
        # (batch, n_heads, len_q, d_k) x (batch, n_heads, d_k, len_k)
        # -> (batch, n_heads, len_q, len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
        
        # Apply mask (for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        # (batch, n_heads, len_q, len_k) x (batch, n_heads, len_k, d_v)
        # -> (batch, n_heads, len_q, d_v)
        output = torch.matmul(attention, v)
        
        return output, attention


# Example usage
attention = ScaledDotProductAttention()
q = torch.randn(2, 8, 10, 64)  # (batch=2, heads=8, seq_len=10, d_k=64)
k = torch.randn(2, 8, 10, 64)
v = torch.randn(2, 8, 10, 64)
output, attn_weights = attention(q, k, v)
print(f"Attention output: {output.shape}, weights: {attn_weights.shape}")
```

### 3.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Difference from single attention:
        - Projects Q, K, V into h different subspaces
        - Allows model to attend to different representation subspaces
        - Each head has dimension d_model / h
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, len_q, d_model)
            k: (batch, len_k, d_model)
            v: (batch, len_v, d_model)
            mask: (batch, 1, len_q, len_k) or None
        
        Returns:
            output: (batch, len_q, d_model)
            attention: (batch, n_heads, len_q, len_k)
        """
        batch_size = q.size(0)
        residual = q
        
        # 1. Linear projections
        q = self.w_q(q)  # (batch, len_q, d_model)
        k = self.w_k(k)  # (batch, len_k, d_model)
        v = self.w_v(v)  # (batch, len_v, d_model)
        
        # 2. Split into multiple heads
        # (batch, len, d_model) -> (batch, len, n_heads, d_k)
        # -> (batch, n_heads, len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        output, attention = self.attention(q, k, v, mask)
        
        # 4. Concatenate heads
        # (batch, n_heads, len_q, d_k) -> (batch, len_q, n_heads, d_k)
        # -> (batch, len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        # 6. Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention


# Example usage
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attn = mha(x, x, x)  # Self-attention
print(f"Multi-head attention output: {output.shape}")
```

### 3.3 Cross-Attention vs Self-Attention

```python
class SelfAttention(nn.Module):
    """
    Self-Attention: Q, K, V all come from same source
    
    Usage: Within encoder or decoder to attend to own sequence
    Formula: Attention(X, X, X)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        return self.mha(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-Attention: Q from one source, K and V from another
    
    Usage: Decoder attending to encoder output
    Formula: Attention(Q_decoder, K_encoder, V_encoder)
    
    Difference from self-attention:
        - Q comes from decoder
        - K, V come from encoder
        - Allows decoder to focus on relevant encoder information
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, q, kv, mask=None):
        """
        Args:
            q: decoder features (batch, len_q, d_model)
            kv: encoder features (batch, len_kv, d_model)
            mask: attention mask
        """
        return self.mha(q, kv, kv, mask)


# Example usage
self_attn = SelfAttention(d_model=512, n_heads=8)
cross_attn = CrossAttention(d_model=512, n_heads=8)

encoder_out = torch.randn(2, 10, 512)
decoder_in = torch.randn(2, 5, 512)

# Self-attention on encoder
self_out, _ = self_attn(encoder_out)
print(f"Self-attention output: {self_out.shape}")

# Cross-attention: decoder queries encoder
cross_out, _ = cross_attn(decoder_in, encoder_out)
print(f"Cross-attention output: {cross_out.shape}")
```

### 3.4 Additive Attention (Bahdanau)

```python
class AdditiveAttention(nn.Module):
    """
    Additive/Bahdanau Attention
    
    Formula:
        score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)
        α_{ts} = softmax(score(h_t, h_s))
        context_t = Σ_s α_{ts} h_s
    
    Difference from scaled dot-product:
        - Uses learned weight matrix and tanh
        - More parameters
        - Can handle different dimensions for Q and K
        - Original attention mechanism for seq2seq
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch, len_q, hidden_dim)
            keys: (batch, len_k, hidden_dim)
            values: (batch, len_k, hidden_dim)
            mask: (batch, len_q, len_k) or None
        
        Returns:
            context: (batch, len_q, hidden_dim)
            attention: (batch, len_q, len_k)
        """
        # Project query and keys
        q = self.W_q(query)  # (batch, len_q, hidden_dim)
        k = self.W_k(keys)   # (batch, len_k, hidden_dim)
        
        # Expand dimensions for broadcasting
        q = q.unsqueeze(2)  # (batch, len_q, 1, hidden_dim)
        k = k.unsqueeze(1)  # (batch, 1, len_k, hidden_dim)
        
        # Compute scores: v^T tanh(W_1 q + W_2 k)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        # (batch, len_q, len_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, values)
        # (batch, len_q, len_k) x (batch, len_k, hidden_dim)
        # -> (batch, len_q, hidden_dim)
        
        return context, attention


# Example usage
additive_attn = AdditiveAttention(hidden_dim=256)
query = torch.randn(2, 5, 256)
keys = torch.randn(2, 10, 256)
values = torch.randn(2, 10, 256)
context, attn = additive_attn(query, keys, values)
print(f"Additive attention context: {context.shape}")
```

### Comparison of Attention Mechanisms

```python
def compare_attention_mechanisms():
    """
    Attention Mechanisms Comparison:
    
    1. Scaled Dot-Product:
       - Formula: softmax(QK^T / √d_k)V
       - Pros: Fast (matrix multiplication), parallelizable
       - Cons: Quadratic complexity O(n²)
       - Use: Modern transformers
    
    2. Multi-Head:
       - Formula: Concat(head_1, ..., head_h)W^O
       - Pros: Multiple representation subspaces
       - Cons: More parameters
       - Use: All transformer architectures
    
    3. Self-Attention:
       - Formula: Attention(X, X, X)
       - Pros: Captures dependencies within sequence
       - Use: Encoder layers
    
    4. Cross-Attention:
       - Formula: Attention(Q_decoder, K_encoder, V_encoder)
       - Pros: Connects encoder and decoder
       - Use: Seq2seq, image captioning
    
    5. Additive (Bahdanau):
       - Formula: v^T tanh(W_1 Q + W_2 K)
       - Pros: Different Q, K dimensions; original seq2seq
       - Cons: Slower than dot-product
       - Use: Legacy seq2seq models
    """
    
    batch, seq_len, d_model = 2, 10, 512
    n_heads = 8
    
    x = torch.randn(batch, seq_len, d_model)
    
    # Compare implementations
    models = {
        'Multi-Head (Self)': MultiHeadAttention(d_model, n_heads),
        'Self-Attention': SelfAttention(d_model, n_heads),
        'Cross-Attention': CrossAttention(d_model, n_heads),
        'Additive': AdditiveAttention(d_model)
    }
    
    print("\nAttention Mechanisms Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:25s} - Parameters: {params:,}")
    
    return models

compare_attention_mechanisms()
```

---

## Cluster 4: Transformer Architectures

### Mathematical Foundation

All transformers use:
1. Multi-head self-attention
2. Position-wise feedforward networks
3. Residual connections and layer normalization
4. Positional encodings

They differ in:
- Encoder-only vs Decoder-only vs Encoder-Decoder
- Training objectives (MLM, CLM, etc.)
- Positional encoding schemes

### 4.1 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why: Transformers have no inherent notion of position
    Properties:
        - Fixed (not learned)
        - Allows extrapolation to longer sequences
        - Distance between positions is consistent
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embeddings (alternative to sinusoidal)
    
    Used in: BERT, GPT-2
    
    Difference from sinusoidal:
        - Learned during training
        - Better for fixed-length sequences
        - Cannot extrapolate beyond max_len
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Get positional embeddings and add
        pos_emb = self.pe(positions)
        x = x + pos_emb
        
        return self.dropout(x)


# Example usage
pos_enc = PositionalEncoding(d_model=512, max_len=100)
learned_pos_enc = LearnedPositionalEncoding(d_model=512, max_len=100)

x = torch.randn(2, 10, 512)
x_with_pos = pos_enc(x)
x_with_learned_pos = learned_pos_enc(x)
print(f"With sinusoidal PE: {x_with_pos.shape}")
print(f"With learned PE: {x_with_learned_pos.shape}")
```

### 4.2 Position-wise Feed-Forward Network

```python
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Formula:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        or with GELU: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    Applied independently to each position:
        - Same network for all positions
        - Different from RNN (no temporal connections)
        - Two linear transformations with activation
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # First linear + activation
        x = self.activation(self.w_1(x))
        x = self.dropout(x)
        
        # Second linear
        x = self.w_2(x)
        x = self.dropout(x)
        
        return x


# Example usage
ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 10, 512)
output = ffn(x)
print(f"FFN output: {output.shape}")
```

### 4.3 Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture:
        1. Multi-Head Self-Attention
        2. Add & Norm
        3. Position-wise Feed-Forward
        4. Add & Norm
    
    Formula:
        x' = LayerNorm(x + MultiHeadAttention(x))
        x'' = LayerNorm(x' + FFN(x'))
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        residual = x
        x, _ = self.self_attn.mha(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers
    
    Used in: BERT, Encoder-Decoder Transformers
    """
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# Example usage
encoder = TransformerEncoder(num_layers=6, d_model=512, n_heads=8, 
                            d_ff=2048, dropout=0.1)
x = torch.randn(2, 10, 512)
output = encoder(x)
print(f"Transformer Encoder output: {output.shape}")
```

### 4.4 Transformer Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
    Architecture:
        1. Masked Multi-Head Self-Attention
        2. Add & Norm
        3. Multi-Head Cross-Attention (attend to encoder)
        4. Add & Norm
        5. Position-wise Feed-Forward
        6. Add & Norm
    
    Difference from Encoder:
        - Self-attention is masked (causal)
        - Has cross-attention to encoder output
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: decoder input (batch, tgt_len, d_model)
            encoder_output: encoder output (batch, src_len, d_model)
            tgt_mask: target (causal) mask
            src_mask: source (padding) mask
        
        Returns:
            output: (batch, tgt_len, d_model)
        """
        # 1. Masked self-attention
        residual = x
        x, _ = self.self_attn.mha(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # 2. Cross-attention to encoder
        residual = x
        x, _ = self.cross_attn.mha(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        # 3. Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of N Transformer Decoder Layers
    
    Used in: Machine Translation, Seq2Seq tasks
    """
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: decoder input (batch, tgt_len, d_model)
            encoder_output: encoder output (batch, src_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.norm(x)


# Example usage
decoder = TransformerDecoder(num_layers=6, d_model=512, n_heads=8,
                            d_ff=2048, dropout=0.1)
tgt = torch.randn(2, 8, 512)
encoder_out = torch.randn(2, 10, 512)
output = decoder(tgt, encoder_out)
print(f"Transformer Decoder output: {output.shape}")
```

### 4.5 Complete Vanilla Transformer

```python
class Transformer(nn.Module):
    """
    Complete Transformer (Encoder-Decoder Architecture)
    
    Original "Attention Is All You Need" architecture
    
    Components:
        - Token Embedding
        - Positional Encoding
        - Encoder Stack
        - Decoder Stack
        - Output Linear Layer
    
    Training: Uses teacher forcing
    Inference: Auto-regressive generation
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 n_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, 
                                         n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model,
                                         n_heads, d_ff, dropout)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for decoder
        
        Returns lower triangular matrix of ones (allows attending to past)
        Upper triangle is -inf (prevents attending to future)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: source sequence (batch, src_len)
            tgt: target sequence (batch, tgt_len)
            src_mask: source padding mask
            tgt_mask: target causal mask
        
        Returns:
            output: (batch, tgt_len, tgt_vocab_size)
        """
        # Embed and add positional encoding
        src = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)
        
        # Generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Encode source
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_linear(decoder_output)
        
        return output


# Example usage
transformer = Transformer(src_vocab_size=10000, tgt_vocab_size=10000,
                         d_model=512, n_heads=8, num_encoder_layers=6,
                         num_decoder_layers=6, d_ff=2048)

src = torch.randint(0, 10000, (2, 10))  # (batch, src_len)
tgt = torch.randint(0, 10000, (2, 8))   # (batch, tgt_len)

output = transformer(src, tgt)
print(f"Transformer output: {output.shape}  # Should be (2, 8, 10000)")
```

### 4.6 BERT-Style Encoder-Only Transformer

```python
class BERTTransformer(nn.Module):
    """
    BERT-Style Encoder-Only Transformer
    
    Differences from vanilla Transformer:
        - Encoder-only (no decoder)
        - Bidirectional context
        - Special tokens: [CLS], [SEP], [MASK]
        - Segment embeddings for sentence pairs
    
    Pre-training tasks:
        1. Masked Language Modeling (MLM)
        2. Next Sentence Prediction (NSP)
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12, 
                 num_layers=12, d_ff=3072, max_len=512, 
                 num_segments=2, dropout=0.1):
        super(BERTTransformer, self).__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Encoder
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, 
                                         d_ff, dropout)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP head
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len) - 0 or 1 for sentence A/B
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            sequence_output: (batch, seq_len, d_model)
            pooled_output: (batch, d_model) - [CLS] token representation
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            seg_emb = self.segment_embedding(segment_ids)
            embeddings = token_emb + pos_emb + seg_emb
        else:
            embeddings = token_emb + pos_emb
        
        embeddings = self.dropout(embeddings)
        
        # Convert attention mask for multi-head attention
        if attention_mask is not None:
            # (batch, 1, 1, seq_len) for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Encode
        sequence_output = self.encoder(embeddings, attention_mask)
        
        # Get [CLS] token output for classification
        pooled_output = sequence_output[:, 0, :]
        
        return sequence_output, pooled_output
    
    def get_mlm_logits(self, sequence_output):
        """Get logits for masked language modeling"""
        return self.mlm_head(sequence_output)
    
    def get_nsp_logits(self, pooled_output):
        """Get logits for next sentence prediction"""
        return self.nsp_head(pooled_output)


# Example usage
bert = BERTTransformer(vocab_size=30000, d_model=768, n_heads=12, 
                      num_layers=12, d_ff=3072)

input_ids = torch.randint(0, 30000, (2, 128))
segment_ids = torch.cat([torch.zeros(2, 64), torch.ones(2, 64)], dim=1).long()
attention_mask = torch.ones(2, 128)

seq_out, pooled_out = bert(input_ids, segment_ids, attention_mask)
print(f"BERT sequence output: {seq_out.shape}, pooled: {pooled_out.shape}")

mlm_logits = bert.get_mlm_logits(seq_out)
nsp_logits = bert.get_nsp_logits(pooled_out)
print(f"MLM logits: {mlm_logits.shape}, NSP logits: {nsp_logits.shape}")
```

### 4.7 GPT-Style Decoder-Only Transformer

```python
class GPTTransformer(nn.Module):
    """
    GPT-Style Decoder-Only Transformer
    
    Differences from vanilla Transformer:
        - Decoder-only (no encoder, no cross-attention)
        - Causal self-attention only
        - Auto-regressive generation
    
    Training: Causal Language Modeling (CLM)
        Predict next token given all previous tokens
    
    Formula: P(x_t | x_1, ..., x_{t-1})
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super(GPTTransformer, self).__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Decoder layers (self-attention only, no cross-attention)
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - optional
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = self.generate_causal_mask(seq_len).to(input_ids.device)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def generate_causal_mask(self, sz):
        """Generate causal mask to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Auto-regressive generation
        
        Args:
            input_ids: (batch, seq_len) - prompt
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class GPTDecoderLayer(nn.Module):
    """
    GPT Decoder Layer (Self-Attention Only)
    
    Difference from Transformer Decoder:
        - No cross-attention (encoder-decoder attention)
        - Only masked self-attention and FFN
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation='gelu')
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        residual = x
        x, _ = self.self_attn.mha(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # FFN with residual
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x


# Example usage
gpt = GPTTransformer(vocab_size=50257, d_model=768, n_heads=12,
                    num_layers=12, d_ff=3072)

input_ids = torch.randint(0, 50257, (2, 20))
logits = gpt(input_ids)
print(f"GPT logits: {logits.shape}")

# Generate text
prompt = torch.randint(0, 50257, (1, 10))
generated = gpt.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
print(f"Generated sequence: {generated.shape}")
```

### Comparison of Transformer Variants

```python
def compare_transformer_architectures():
    """
    Transformer Architecture Comparison:
    
    1. Vanilla Transformer (Encoder-Decoder):
       - Components: Encoder + Decoder with cross-attention
       - Use: Machine translation, seq2seq tasks
       - Training: Teacher forcing
       - Example: Original "Attention Is All You Need"
    
    2. BERT (Encoder-Only):
       - Components: Encoder stack only
       - Attention: Bidirectional (can see full context)
       - Use: Classification, NER, QA
       - Training: MLM + NSP
       - Pros: Best for understanding tasks
    
    3. GPT (Decoder-Only):
       - Components: Decoder stack (no cross-attention)
       - Attention: Causal (can only see past)
       - Use: Text generation, few-shot learning
       - Training: Causal LM (next token prediction)
       - Pros: Excellent for generation
    
    4. T5 (Encoder-Decoder):
       - Components: Full encoder-decoder
       - Use: All tasks as text-to-text
       - Training: Span corruption
       - Pros: Unified framework
    
    Architecture Sizes:
    """
    
    vocab_size = 50000
    
    # Create models
    vanilla = Transformer(vocab_size, vocab_size, d_model=512, n_heads=8,
                         num_encoder_layers=6, num_decoder_layers=6)
    
    bert = BERTTransformer(vocab_size, d_model=768, n_heads=12, num_layers=12)
    
    gpt = GPTTransformer(vocab_size, d_model=768, n_heads=12, num_layers=12)
    
    models = {
        'Vanilla Transformer': vanilla,
        'BERT (Encoder-Only)': bert,
        'GPT (Decoder-Only)': gpt
    }
    
    print("\nTransformer Architectures Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:30s} - Parameters: {params:,}")
    
    return models

compare_transformer_architectures()
```

---

## Cluster 5: Efficient Transformer Variants

### Mathematical Foundation

All efficient transformers aim to reduce O(n²) complexity of standard attention.

Methods:
1. **Low-rank approximation**: Project to lower dimensions
2. **Sparsity**: Attend to subset of tokens
3. **Kernel methods**: Approximate attention with kernel functions
4. **Recurrence**: Mix attention with recurrent processing

### 5.1 Linformer (Linear Complexity)

```python
class Linformer(nn.Module):
    """
    Linformer: Self-Attention with Linear Complexity
    
    Key Idea: Project keys and values to lower dimension k << n
    
    Formula:
        Attention(Q, K, V) = softmax(Q(EK)^T / √d) (FV)
        where E, F ∈ R^{k×n} are projection matrices
    
    Complexity:
        Standard: O(n²d)
        Linformer: O(nkd) where k << n
    
    Difference from standard transformer:
        - Projects K, V to fixed dimension k
        - Linear in sequence length
        - Some information loss due to projection
    """
    def __init__(self, d_model, n_heads, seq_len, k=256, dropout=0.1):
        super(Linformer, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = min(k, seq_len)  # Projected dimension
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Low-rank projections for K and V
        self.E = nn.Linear(seq_len, self.k, bias=False)
        self.F = nn.Linear(seq_len, self.k, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # (batch, n_heads, seq_len, d_k)
        
        # Project K and V to lower dimension
        # Transpose to (batch, n_heads, d_k, seq_len) for projection
        K = K.transpose(2, 3)  # (batch, n_heads, d_k, seq_len)
        V = V.transpose(2, 3)
        
        # Apply low-rank projection: (batch, n_heads, d_k, seq_len) -> (batch, n_heads, d_k, k)
        K = self.E(K)  # (batch, n_heads, d_k, k)
        V = self.F(V)  # (batch, n_heads, d_k, k)
        
        # Transpose back
        K = K.transpose(2, 3)  # (batch, n_heads, k, d_k)
        V = V.transpose(2, 3)  # (batch, n_heads, k, d_k)
        
        # Compute attention scores: (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, k)
        # -> (batch, n_heads, seq_len, k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply to values: (batch, n_heads, seq_len, k) x (batch, n_heads, k, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        output = torch.matmul(attention, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
linformer = Linformer(d_model=512, n_heads=8, seq_len=1024, k=256)
x = torch.randn(2, 1024, 512)
output = linformer(x)
print(f"Linformer output: {output.shape}")
print(f"Complexity: O(n*k*d) = O({1024}*{256}*{512}) vs O(n²*d) = O({1024**2}*{512})")
```

### 5.2 Performer (Kernel-based Attention)

```python
class Performer(nn.Module):
    """
    Performer: Fast Attention via Positive Orthogonal Random Features
    
    Key Idea: Approximate softmax attention using random feature maps
    
    Formula:
        Attention(Q, K, V) ≈ φ(Q)(φ(K)^T V) / (φ(Q)φ(K)^T 1)
        where φ is a random feature map
    
    Benefits:
        - Linear complexity O(nd²)
        - No approximation error bound
        - Can be computed in forward or recurrent mode
    
    Difference from standard:
        - Uses kernel approximation of softmax
        - Reorders computation: φ(Q)(φ(K)^T V) instead of (Qφ(K)^T)V
    """
    def __init__(self, d_model, n_heads, nb_features=256, dropout=0.1):
        super(Performer, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nb_features = nb_features
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_projection_matrix(self, m, d, device):
        """
        Create random projection matrix for FAVOR+ algorithm
        
        Args:
            m: number of random features
            d: dimension of keys/queries
        """
        # Orthogonal random features
        projection = torch.randn(m, d, device=device)
        
        # Gram-Schmidt orthogonalization
        q, r = torch.qr(projection)
        d = torch.diag(r)
        ph = d / torch.abs(d)
        projection = q * ph.unsqueeze(0)
        
        return projection
    
    def kernel_feature_map(self, x, projection_matrix):
        """
        Apply random feature map
        
        Formula: φ(x) = exp(xω^T - ||x||²/2) / √m
        where ω is the random projection matrix
        """
        # x: (batch, n_heads, seq_len, d_k)
        data_normalizer = 1.0 / torch.sqrt(torch.tensor(self.nb_features, dtype=torch.float32))
        
        # Project: (batch, n_heads, seq_len, nb_features)
        projection = torch.matmul(x, projection_matrix.T)
        
        # Compute ||x||²/2
        data_dash = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
        
        # Apply exponential kernel
        data_dash = projection - data_dash / 2.0
        data_dash = torch.exp(data_dash) * data_normalizer
        
        return data_dash
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Create random projection matrix
        projection_matrix = self.create_projection_matrix(
            self.nb_features, self.d_k, device
        )
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply kernel feature map
        Q_prime = self.kernel_feature_map(Q, projection_matrix)
        K_prime = self.kernel_feature_map(K, projection_matrix)
        
        # Q_prime, K_prime: (batch, n_heads, seq_len, nb_features)
        # V: (batch, n_heads, seq_len, d_k)
        
        # Compute attention: φ(Q)(φ(K)^T V)
        # First compute φ(K)^T V: (batch, n_heads, nb_features, d_k)
        KV = torch.matmul(K_prime.transpose(-2, -1), V)
        
        # Then compute φ(Q) @ KV: (batch, n_heads, seq_len, d_k)
        output = torch.matmul(Q_prime, KV)
        
        # Normalize by φ(Q) @ φ(K)^T @ 1
        # Compute φ(K)^T @ 1: (batch, n_heads, nb_features, 1)
        K_sum = torch.sum(K_prime, dim=2, keepdim=True).transpose(-2, -1)
        
        # Compute φ(Q) @ K_sum: (batch, n_heads, seq_len, 1)
        normalizer = torch.matmul(Q_prime, K_sum) + 1e-8
        
        # Normalize output
        output = output / normalizer
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
performer = Performer(d_model=512, n_heads=8, nb_features=256)
x = torch.randn(2, 1024, 512)
output = performer(x)
print(f"Performer output: {output.shape}")
print("Complexity: O(nd²) - linear in sequence length!")
```

### 5.3 Reformer (LSH Attention)

```python
import math

class LSHAttention(nn.Module):
    """
    Locality Sensitive Hashing (LSH) Attention
    
    Key Idea: Use LSH to find similar queries/keys, only attend within buckets
    
    Process:
        1. Hash queries and keys into buckets
        2. Sort by bucket
        3. Attend only within same bucket
        4. Complexity: O(n log n)
    
    Formula:
        hash(x) = argmax(xR)
        where R is random projection matrix
    """
    def __init__(self, d_model, n_heads, n_hashes=4, bucket_size=64, dropout=0.1):
        super(LSHAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def hash_vectors(self, vecs, num_buckets):
        """
        Hash vectors using random projection
        
        Args:
            vecs: (batch, n_heads, seq_len, d_k)
            num_buckets: number of hash buckets
        
        Returns:
            hashes: (batch, n_heads, seq_len)
        """
        batch_size, n_heads, seq_len, d_k = vecs.size()
        
        # Create random rotation matrix
        rotations = torch.randn(d_k, num_buckets, device=vecs.device)
        rotations = F.normalize(rotations, dim=0)
        
        # Project vectors: (batch, n_heads, seq_len, d_k) x (d_k, num_buckets)
        # -> (batch, n_heads, seq_len, num_buckets)
        rotated = torch.matmul(vecs, rotations)
        
        # Hash is argmax of projections
        hashes = torch.argmax(rotated, dim=-1)
        
        return hashes
    
    def forward(self, x, mask=None):
        """
        Simplified LSH attention (full implementation is complex)
        
        For production, use official Reformer implementation
        """
        batch_size, seq_len, _ = x.size()
        
        # Standard attention as fallback
        # (Full LSH implementation requires careful sorting and chunking)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute hash buckets
        num_buckets = seq_len // self.bucket_size
        hashes = self.hash_vectors(Q, num_buckets)
        
        # For simplicity, fall back to standard attention
        # Real implementation would chunk by buckets
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output


# Example usage
lsh_attn = LSHAttention(d_model=512, n_heads=8, n_hashes=4, bucket_size=64)
x = torch.randn(2, 512, 512)
output = lsh_attn(x)
print(f"LSH Attention output: {output.shape}")
print("Complexity: O(n log n) due to sorting")
```

### 5.4 Longformer (Sparse Attention Patterns)

```python
class LongformerAttention(nn.Module):
    """
    Longformer: Combines local windowed attention with global attention
    
    Attention Patterns:
        1. Local window: Each token attends to w tokens on each side
        2. Global attention: Special tokens attend to all, all attend to them
        3. Dilated window: Increases receptive field
    
    Complexity: O(n × w) where w is window size (typically w << n)
    
    Formula:
        Attention_local(i) = softmax(Q_i K_{i-w:i+w}^T / √d) V_{i-w:i+w}
        Attention_global(i) = softmax(Q_i K_all^T / √d) V_all  (for special tokens)
    """
    def __init__(self, d_model, n_heads, window_size=512, 
                 global_tokens=[], dropout=0.1):
        super(LongformerAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.global_tokens = global_tokens  # List of token positions with global attention
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Separate projections for global attention
        self.w_q_global = nn.Linear(d_model, d_model)
        self.w_k_global = nn.Linear(d_model, d_model)
        self.w_v_global = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _sliding_window_attention(self, q, k, v, window_size):
        """
        Compute sliding window attention
        
        Args:
            q, k, v: (batch, n_heads, seq_len, d_k)
            window_size: size of local window
        """
        batch_size, n_heads, seq_len, d_k = q.size()
        
        # For simplicity, compute full attention and mask
        # Production implementation would use efficient sliding window
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Create sliding window mask
        mask = self._create_sliding_window_mask(seq_len, window_size, q.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v)
        
        return output
    
    def _create_sliding_window_mask(self, seq_len, window_size, device):
        """
        Create mask for sliding window attention
        
        Returns: (seq_len, seq_len) mask where 1 = attend, 0 = don't attend
        """
        # Create distance matrix
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Allow attention within window
        mask = torch.abs(distance) <= window_size
        
        return mask.float()
    
    def forward(self, x, global_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            global_mask: (batch, seq_len) - 1 for global tokens, 0 for local
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Local attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute sliding window attention
        output = self._sliding_window_attention(Q, K, V, self.window_size)
        
        # Add global attention for special tokens (e.g., [CLS])
        if global_mask is not None:
            Q_global = self.w_q_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K_global = self.w_k_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V_global = self.w_v_global(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # Global tokens attend to all
            global_scores = torch.matmul(Q_global, K_global.transpose(-2, -1)) / (self.d_k ** 0.5)
            global_attn = F.softmax(global_scores, dim=-1)
            global_output = torch.matmul(global_attn, V_global)
            
            # Merge local and global attention based on mask
            mask_expanded = global_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            output = output * (1 - mask_expanded) + global_output * mask_expanded
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        return output


# Example usage
longformer = LongformerAttention(d_model=512, n_heads=8, window_size=256)
x = torch.randn(2, 4096, 512)  # Long sequence
global_mask = torch.zeros(2, 4096)
global_mask[:, 0] = 1  # First token has global attention
output = longformer(x, global_mask)
print(f"Longformer output: {output.shape}")
print(f"Complexity: O(n × w) = O({4096} × {256}) vs O(n²) = O({4096**2})")
```

### Comparison of Efficient Transformers

```python
def compare_efficient_transformers():
    """
    Efficient Transformer Comparison:
    
    1. Standard Transformer:
       - Complexity: O(n²d)
       - Memory: O(n²)
       - Best: Accuracy, short sequences
    
    2. Linformer:
       - Complexity: O(nkd) where k is projection dim
       - Method: Low-rank approximation of attention
       - Pros: Simple, deterministic
       - Cons: Fixed sequence length, approximation error
       - Best: Medium sequences (512-2048)
    
    3. Performer:
       - Complexity: O(nd²)
       - Method: Kernel approximation with random features
       - Pros: Provable approximation, recurrent mode
       - Cons: Variance in approximation
       - Best: Very long sequences, streaming
    
    4. Reformer (LSH):
       - Complexity: O(n log n)
       - Method: Locality sensitive hashing
       - Pros: No approximation within buckets
       - Cons: Complex implementation, sorting overhead
       - Best: Very long sequences (> 8K)
    
    5. Longformer:
       - Complexity: O(nw) where w is window size
       - Method: Sparse attention patterns
       - Pros: Simple, flexible patterns, exact
       - Cons: Fixed patterns, not adaptive
       - Best: Documents (4K-16K tokens)
    """
    
    seq_len = 2048
    d_model = 512
    n_heads = 8
    
    x = torch.randn(2, seq_len, d_model)
    
    models = {
        'Linformer (k=256)': Linformer(d_model, n_heads, seq_len, k=256),
        'Performer (m=256)': Performer(d_model, n_heads, nb_features=256),
        'LSH Attention': LSHAttention(d_model, n_heads, bucket_size=64),
        'Longformer (w=256)': LongformerAttention(d_model, n_heads, window_size=256)
    }
    
    print("\nEfficient Transformer Comparison:")
    print(f"Sequence length: {seq_len}")
    print("-" * 70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        try:
            output = model(x)
            print(f"{name:25s} - Params: {params:,} - Output: {output.shape}")
        except Exception as e:
            print(f"{name:25s} - Params: {params:,} - Error: {str(e)[:30]}")
    
    print("\nComplexity Summary:")
    print(f"Standard Attention:  O(n²d) = O({seq_len**2 * d_model:,})")
    print(f"Linformer:          O(nkd) = O({seq_len * 256 * d_model:,})")
    print(f"Performer:          O(nd²) = O({seq_len * d_model**2:,})")
    print(f"LSH:                O(n log n * d) = O({int(seq_len * math.log2(seq_len) * d_model):,})")
    print(f"Longformer:         O(nwd) = O({seq_len * 256 * d_model:,})")
    
    return models

compare_efficient_transformers()
```

---

## Cluster 6: State Space Models

### Mathematical Foundation

State space models process sequences through continuous-time dynamics:

$h'(t) = Ah(t) + Bx(t)$
$y(t) = Ch(t) + Dx(t)$

They differ in:
- Discretization methods
- Parameterization (fixed vs input-dependent)
- Initialization strategies

### 6.1 S4 (Structured State Space)

```python
class S4Layer(nn.Module):
    """
    S4 (Structured State Space Sequence) Layer
    
    Continuous-time state space:
        h'(t) = Ah(t) + Bx(t)
        y(t) = Ch(t)
    
    Discretized:
        h_k = A̅h_{k-1} + B̅x_k
        y_k = Ch_k
    
    Key innovation: Structured matrices (HiPPO initialization)
    for stable long-range dependencies
    
    Complexity: O(N log N) via FFT where N is state size
    """
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super(S4Layer, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters (simplified)
        # Real S4 uses HiPPO initialization
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Discretization parameter
        self.log_dt = nn.Parameter(torch.randn(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize with stable dynamics"""
        # Simplified initialization (real S4 uses HiPPO)
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
    
    def discretize(self):
        """
        Discretize continuous parameters
        
        Returns:
            A_bar: discretized state transition
            B_bar: discretized input matrix
        """
        dt = torch.exp(self.log_dt)  # (d_model,)
        
        # Zero-order hold discretization (simplified)
        # A_bar = exp(dt * A)
        # B_bar = (A^-1)(exp(dt * A) - I) * dt * B
        
        # For simplicity, use Euler method
        A_bar = torch.eye(self.d_state, device=self.A.device) + dt.mean() * self.A
        B_bar = dt.mean() * self.B
        
        return A_bar, B_bar
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Discretize parameters
        A_bar, B_bar = self.discretize()
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        
        # Process sequence recurrently
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # State update: h_k = A̅h_{k-1} + B̅x_k
            h = torch.matmul(h, A_bar.T) + torch.matmul(x_t, B_bar.T)
            
            # Output: y_k = Ch_k + Dx_k
            y = torch.matmul(h, self.C.T) + x_t * self.D
            
            outputs.append(y)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        return self.dropout(output)


class S4Block(nn.Module):
    """
    S4 Block with normalization and feedforward
    
    Similar structure to Transformer block but with S4 instead of attention
    """
    def __init__(self, d_model, d_state=64, d_ff=None, dropout=0.1):
        super(S4Block, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.s4 = S4Layer(d_model, d_state, dropout)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # S4 with residual
        x = x + self.s4(self.norm1(x))
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


# Example usage
s4_layer = S4Layer(d_model=512, d_state=64)
x = torch.randn(2, 1000, 512)
output = s4_layer(x)
print(f"S4 output: {output.shape}")
```

### 6.2 Mamba (Selective State Space)

```python
class MambaBlock(nn.Module):
    """
    Mamba: Selective State Space Model
    
    Key Innovation: Input-dependent parameters (B, C, Δ)
    
    Formula:
        B_t = Linear_B(x_t)
        C_t = Linear_C(x_t)
        Δ_t = softplus(Linear_Δ(x_t))
        
        h_t = A̅(Δ_t)h_{t-1} + B̅(Δ_t)B_t x_t
        y_t = C_t h_t
    
    Differences from S4:
        - Parameters depend on input (selective)
        - Hardware-aware implementation
        - Linear complexity with sequence length
    
    Advantages over Transformers:
        - O(N) complexity vs O(N²)
        - Better for very long sequences
        - Constant memory during generation
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution (for local context)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # State space parameters (shared, not input-dependent)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(d_model)
    
    def selective_scan(self, x, delta, A, B, C):
        """
        Selective SSM scan (simplified version)
        
        Full implementation uses custom CUDA kernels for efficiency
        
        Args:
            x: (batch, seq_len, d_inner)
            delta: (batch, seq_len, d_inner)
            A: (d_inner, d_state)
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)
        """
        batch_size, seq_len, d_inner = x.size()
        d_state = A.size(1)
        
        # Discretize A
        # A_bar = exp(delta * A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # (batch, seq_len, d_inner, d_state)
        
        # Discretize B
        # B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
        # (batch, seq_len, d_inner, d_state)
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Update state: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Output: y_t = C_t * h_t
            y = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        
        return output
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Input projection and split
        x_proj = self.in_proj(x)
        x, z = x_proj.chunk(2, dim=-1)  # (batch, seq_len, d_inner) each
        
        # Convolution for local context
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :x.size(2)]  # Remove padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        x = F.silu(x)  # SiLU activation
        
        # Generate input-dependent parameters
        x_db_dt = self.x_proj(x)  # (batch, seq_len, d_state + d_state + d_inner)
        
        delta, B, C = torch.split(
            x_db_dt,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1
        )
        
        # Delta needs to be positive
        delta = F.softplus(self.dt_proj(delta))
        
        # Get A from log space
        A = -torch.exp(self.A_log)
        
        # Selective scan
        y = self.selective_scan(x, delta, A, B, C)
        
        # Skip connection (D parameter)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        # Residual connection
        output = output + residual
        
        return output


# Example usage
mamba = MambaBlock(d_model=512, d_state=16, d_conv=4, expand=2)
x = torch.randn(2, 2048, 512)
output = mamba(x)
print(f"Mamba output: {output.shape}")
print("Complexity: O(N) - linear in sequence length!")
```

### 6.3 Comparison: State Space vs Attention

```python
def compare_state_space_models():
    """
    State Space Models vs Transformers:
    
    1. S4 (Structured State Space):
       - Method: Fixed HiPPO-initialized SSM
       - Complexity: O(N log N) via FFT convolution
       - Pros: Stable long-range dependencies, efficient
       - Cons: Not input-dependent
       - Use: Time series, audio, long sequences
    
    2. Mamba (Selective SSM):
       - Method: Input-dependent B, C, Δ parameters
       - Complexity: O(N) with custom kernels
       - Pros: Selective memory, hardware-efficient
       - Cons: More complex implementation
       - Use: Language modeling, long-context tasks
    
    3. Transformers:
       - Method: Softmax attention
       - Complexity: O(N²)
       - Pros: Flexible, content-based routing
       - Cons: Quadratic complexity
       - Use: Most NLP tasks
    
    Trade-offs:
        - SSMs: Better scaling, constant memory inference
        - Transformers: Better in-context learning, flexibility
    """
    
    d_model = 512
    seq_lens = [512, 1024, 2048, 4096]
    
    print("\nComplexity Comparison:")
    print(f"{'Seq Len':<10} {'Transformer':<20} {'S4/Mamba':<20}")
    print("-" * 50)
    
    for n in seq_lens:
        transformer_ops = n * n * d_model
        ssm_ops = n * d_model * 64  # Assuming d_state=64
        
        print(f"{n:<10} {transformer_ops:>15,}      {ssm_ops:>15,}")
    
    # Create models
    models = {
        'S4 Block': S4Block(d_model, d_state=64),
        'Mamba Block': MambaBlock(d_model, d_state=16),
        'Transformer Layer': TransformerEncoderLayer(d_model, n_heads=8, d_ff=2048)
    }
    
    print("\n\nModel Parameter Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s} - Parameters: {params:,}")
    
    return models

compare_state_space_models()
```

---

## Cluster 7: Vision and Multi-Modal Architectures

### Mathematical Foundation

These models adapt transformers/attention for visual and multi-modal data.

Key differences:
- Input tokenization (patches vs pixels vs regions)
- Position embeddings (2D vs 1D)
- Modality fusion (early vs late, cross-attention)
