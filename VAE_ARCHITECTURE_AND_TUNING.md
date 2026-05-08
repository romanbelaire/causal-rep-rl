# VAE Critic Architecture and Tuning Guide

## Current Architecture

### Data Flow
```
obs (147-dim) → encoder → (mu, log_std) → z (32-dim latent) → value_head → value
                                              ↓
                                         policy (also uses z)
```

**Answer to your question**: Yes, **z feeds directly into value_head**. There is no separate representation - the VAE latent `z` is used by both:
- **Value head**: `z → value_head → value`
- **Policy**: `z → policy → actions`

The architecture is: `obs → encoder → z → {value_head, policy}`

## Critical Issue: VAE Loss Not Used!

### Problem
Looking at the training code, **VAE reconstruction and KL losses are NOT being used** during training. The encoder is only getting gradients from value prediction loss.

**Current training flow**:
```python
# In PPO/TRPO update():
values = critic(obs).squeeze(-1)  # Only value prediction
value_loss = mse_loss(values, returns)
value_loss.backward()  # Only value loss gradients
```

**What's missing**:
- No reconstruction loss (decoder not trained)
- No KL loss (encoder not regularized)
- Encoder only optimized for value prediction, not representation quality

### Impact
This explains why:
1. **70% of encoder parameters are unused** (line 125, 230, 336, 442, 548 in output)
2. **Encoder might not learn meaningful representations**
3. **Convexity violations** (negative μ) - encoder not learning structured representations

## Recommended Fixes

### 1. Add VAE Loss to Training (CRITICAL)

The VAE critic needs reconstruction and KL losses to learn good representations. Modify the training code to include VAE loss:

**Option A: Add VAE loss to value loss**
```python
# In algorithm update():
values, vae_info = critic(obs, return_latent=True)
value_loss = mse_loss(values, returns)
vae_loss = vae_info["vae_loss"]  # recon_loss + beta * kl_loss
total_critic_loss = value_loss + vae_coef * vae_loss
total_critic_loss.backward()
```

**Option B: Separate VAE loss term**
Train encoder/decoder separately from value head with VAE loss.

### 2. Architecture Tuning Recommendations

#### Current Config:
```json
"latent_dim": 32,
"encoder_hidden": [512, 512],
"decoder_hidden": [256, 256],
"value_hidden": [256, 256],
"beta": 1.0
```

#### Recommendations:

**A. Increase Latent Dimension** (if representation is too compressed)
```json
"latent_dim": 64  // or 128
```
- **Why**: 32-dim might be too small for rich representations
- **Trade-off**: Larger latent = more capacity but harder to regularize

**B. Adjust Encoder Capacity**
```json
"encoder_hidden": [512, 512, 256]  // Add layer, or
"encoder_hidden": [256, 256]       // Reduce if overfitting
```
- **Current**: [512, 512] is reasonable
- **If encoder params unused**: Try smaller [256, 256] or add depth [512, 512, 256]
- **If underfitting**: Increase to [512, 512, 512] or [1024, 512]

**C. Adjust Beta (KL Weight)**
```json
"beta": 0.1  // Start lower, increase gradually
```
- **Current**: 1.0 might be too high, causing posterior collapse
- **Recommendation**: Start with 0.1-0.5, increase if needed
- **Beta-VAE**: Lower beta = better reconstruction, higher beta = better disentanglement

**D. Value Head Architecture**
```json
"value_hidden": [128, 128]  // Current is fine, or
"value_hidden": [256]       // Simpler if overfitting
```
- Current [256, 256] is reasonable
- Can simplify to [128] or [256] if needed

### 3. Training Hyperparameters

**Learning Rate**:
- Current: `1e-4` for unified optimizer
- **Recommendation**: Use separate learning rates:
  ```json
  "critic_lr": 3e-4,  // Higher for critic (as in ICNN config)
  "learning_rate": 1e-4  // For policy
  ```

**VAE Loss Coefficient**:
- Add `"vae_coef": 0.1` to balance value loss and VAE loss
- Start small (0.01-0.1), increase if encoder not learning

## Architecture Question: Direct z → V or Separate Representation?

### Current: Direct z → V (Recommended)
```
obs → encoder → z → value_head → value
              → policy
```

**Pros**:
- ✅ Simpler architecture
- ✅ z is learned to be useful for both value and policy
- ✅ Fewer parameters
- ✅ Standard VAE approach

**Cons**:
- ⚠️ z must serve both purposes (value + policy)
- ⚠️ Might create tension between value prediction and policy representation

### Alternative: Separate Representation (Not Recommended)
```
obs → encoder → z → value_head → value
              → z_policy → policy
```

**Why not recommended**:
- More complex
- Requires two encoders or splitting z
- No clear benefit for your use case
- Current direct approach is standard and works well

## Specific Tuning for Your Case

Based on your output showing:
- 70% encoder parameters unused
- Negative μ (convexity violations)
- Training working but encoder might not be optimal

### Priority 1: Add VAE Loss (MOST IMPORTANT)
Without reconstruction loss, the encoder won't learn good representations.

### Priority 2: Tune Beta
Try `"beta": 0.1` or `"beta": 0.5` to prevent posterior collapse.

### Priority 3: Increase Latent Dimension
Try `"latent_dim": 64` to give encoder more capacity.

### Priority 4: Monitor VAE Metrics
Add logging for:
- Reconstruction loss
- KL loss
- Latent z statistics (mean, std, diversity)

## Implementation Priority

1. **CRITICAL**: Add VAE loss to training (see code changes needed below)
2. **HIGH**: Tune beta (try 0.1, 0.5, 1.0)
3. **MEDIUM**: Increase latent_dim to 64
4. **LOW**: Adjust encoder_hidden (current is fine)

## Code Changes Needed

To add VAE loss, you'll need to modify the algorithm update functions to:
1. Call `critic(obs, return_latent=True)` instead of `critic(obs)`
2. Extract VAE loss from returned dict
3. Add VAE loss to total critic loss with appropriate coefficient

This is the most important fix for VAE encoder quality.

