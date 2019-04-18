# Contextual-Attention
Contextual Attention for Reducing Dimension (Keras &amp; Pytorch)

Follow the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

Keras version originates from : https://github.com/cbaziotis/

# Requirements:

### Keras version:
   - Keras over ver 2.0
   - Tensorflow backend

### Pytorch version: 
   - Pytorch over 0.4.0


# Input & Output Format:

```
Input Shape  [batch_size, time_step, emb_size]

Output Shape [batch_size, emb_size]

Mask Shape   [batch_size, time_step, 1]
```

# Usage:

### Keras version:
```
from Attention_Keras import Attention
model.add(LSTM(64, return_sequences=True))
model.add(Attention())
# next add a Dense layer (for classification/regression) or whatever...
```

### Pytorch version:
```
from Attention_Pytorch import Attention
import torch
import numpy

att = Attention([None, time_step, emb_size])
x = torch.Tensor(np.random.rand(100, 20, 256))
context, context_distribution = att(x) # context (100, 256); context_distribution (100, 20)
```

