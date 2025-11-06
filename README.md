# CSE4007-Artificial-Intelligence-DETR
Implementation of DETR model with hyperparameter tuning

## Positional Encoding of DETR
Transformers lack an inherent notion of spatial layout, so DETR injects position information via fixed sinusoidal encodings.  
This implementation builds 2D coordinate grids, optionally normalizes them with \(2\pi\), and applies sine to even dimensions and cosine to odd ones across exponentially scaled frequencies. The x/y encodings are concatenated and reshaped to match the input tensor.  
This parameter-free encoding supplies absolute location signals for attention and follows the original DETR design.

---

## Attention Module of DETR
`DetrAttention` implements multi-head attention for both self- and cross-attention. Queries (Q), keys (K), values (V), and the output are produced by distinct linear projections, enabling attention over multiple subspaces in parallel.  
Object-query embeddings are added on the query path, and spatial positional embeddings can be added on the key/value path to align queries with image regions.  
Optional attention masks (e.g., for padding) are supported; scores are normalized with softmax and combined with V, then projected back to the model dimension.

---

## Encoder Layers (encoder layer and encoder) of DETR
Each encoder layer consists of:  
- Multi-head self-attention (with positional encodings added to Q/K)  
- A two-layer feed-forward network (FFN)  
- Residual connections, LayerNorm, and dropout around both sublayers  

The encoder stacks these layers to refine flattened image features. Positional information is fused inside attention, maintaining spatial awareness across depth.

---

## Decoder Layers (decoder layer and decoder) of DETR
Each decoder layer contains:  
- Self-attention over decoder states (object queries + query position embeddings)  
- Cross-attention from queries to encoder outputs (queries use query position embeddings; encoder features may receive spatial positional embeddings)  
- A two-layer FFN, with residuals, LayerNorm, and dropout after each sublayer  

Following DETR, the decoder applies a final LayerNorm to stabilize outputs before prediction heads.  
Across layers, the decoder iteratively refines a fixed set of object queries into object-aware representations.

---

## the Heads (box & class) and the Final Architecture of DETR
The full pipeline connects:  
**CNN backbone → 1×1 conv projection to `d_model` → Transformer encoder → Transformer decoder (with learned query embeddings).**  

Two prediction heads turn decoder outputs into detections:  
- **Classification head:** linear layer producing logits over object classes plus a special “no-object” class.  
- **Bounding-box head:** small MLP regressing normalized \([0,1]\) box coordinates \((c_x, c_y, w, h)\). Hidden layers use ReLU; the final layer is linear, and outputs are passed through sigmoid.  

---

## Hungarian Matcher
DETR trains with a set-prediction objective. The Hungarian matcher computes a one-to-one assignment between predictions and ground truth by minimizing a weighted sum of:  
- Classification cost (negative log-probability of the target class)  
- L1 box distance  
- GIoU cost (negated IoU-based score)  

This matching removes the need for anchors and NMS, enabling end-to-end training.

---

## Object Loss
An auxiliary metric compares the predicted number of objects (non “no-object” queries) to the ground-truth count via L1 error.  
This does not backpropagate but helps monitor how well the model separates objects from background.

---

## Class Loss
Classification uses cross-entropy over the fixed set of queries.  
Unmatched queries are labeled as “no-object”; its contribution is down-weighted by an `eos_coef` (empty-class weight) to reduce imbalance and stabilize training.

---

## Bounding Box Loss
Box regression combines:  
- **L1 loss** on \((c_x, c_y, w, h)\)  
- **GIoU loss** on corner-format boxes (computed as \(1-\text{GIoU}\))  

Losses are aggregated over matched pairs and weighted in the total objective.

---

## Results
Training with a stronger learning-rate schedule, 40 epochs, and batch size 8 improved mAP by approximately **+0.32** over the baseline configuration.  
Optimizer choice (Adam vs. AdamW) was less critical than the scheduler settings in this setup.  
Data augmentation was intentionally minimal to keep the focus on architectural and training-parameter changes.
