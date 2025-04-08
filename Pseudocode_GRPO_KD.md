Pseudocode for GRPO and PPO with Knowledge Distillation RewardsThis document presents pseudocode for integrating knowledge distillation (KD) concepts, inspired by MiniLLM and GEMMA 3, into Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO) for training language models.Core Components:Student Policy (\pi_{\theta}): The language model being trained.Teacher Policy (\pi_{T}): A larger, more capable model used for guidance.Reference Policy (\pi_{ref}): A fixed copy of the student policy (often the initial state) used for KL regularization.Prompt Dataset (D): A collection of input prompts.Pre-training Dataset (D_{PT}): A collection of general text data for the L_{PT} loss.Group Size (G): Number of responses generated per prompt in GRPO.Hyperparameters: \beta (KL weight), \lambda (L_{PT} weight), \epsilon (small constant for normalization).1. GRPO with Teacher Log-Likelihood RewardThis approach uses the log-probability assigned by the teacher model to the student's generated sequence as the primary reward signal within the GRPO framework.Algorithm 1: GRPO with Teacher Log-Likelihood Reward

Initialize student policy parameters θ
Initialize reference policy π_ref with initial θ
Set hyperparameters: group size G, KL weight β, learning rate α

for each training iteration:
  Sample a batch of prompts D_b from dataset D

  // Store trajectories (prompts, outputs, rewards, advantages)
  Trajectories = []

  for each prompt q in D_b:
    // 1. Generate G responses using the current student policy π_θ
    Outputs_q = [π_θ.generate(q) for _ in range(G)]  // List of G output sequences [o_1, ..., o_G]

    // 2. Compute reward for each output using the teacher model
    Rewards_q = []
    for output o_j in Outputs_q:
      // Reward is the log-probability assigned by the teacher
      r_j = log π_T(o_j | q)
      Rewards_q.append(r_j)

    // 3. Compute group baseline and normalized advantages
    μ_q = mean(Rewards_q)
    σ_q = std(Rewards_q)
    Advantages_q = [(r_j - μ_q) / (σ_q + ε) for r_j in Rewards_q]

    // Store prompt, outputs, rewards, advantages for this group
    for j in range(G):
      Trajectories.append((q, Outputs_q[j], Rewards_q[j], Advantages_q[j]))

  // 4. Compute GRPO Loss
  L_policy = 0
  L_kl = 0
  for (q, o, r, A) in Trajectories:
    // Policy loss term (negative for gradient ascent)
    log_prob_student = log π_θ(o | q)
    L_policy += - A * log_prob_student

    // KL divergence penalty w.r.t. reference policy
    log_prob_ref = log π_ref(o | q)
    kl_div = log_prob_student - log_prob_ref // Approximation or exact calculation
    L_kl += kl_div

  // Average losses over the batch
  L_policy = L_policy / len(Trajectories)
  L_kl = L_kl / len(Trajectories)

  // Total GRPO loss
  Loss = L_policy + β * L_kl

  // 5. Update student policy parameters θ using gradient descent
  ∇θ = compute_gradients(Loss, θ)
  θ = update_parameters(θ, ∇θ, α)

  // Optional: Update reference policy π_ref periodically (e.g., copy θ)

end for
2. GRPO with MiniLLM-style Reward (Teacher LL + L_{PT})This approach combines the teacher log-likelihood reward with an auxiliary pre-training loss (L_{PT}) to prevent catastrophic forgetting, inspired by MiniLLM.Algorithm 2: GRPO with MiniLLM-style Reward

Initialize student policy parameters θ
Initialize reference policy π_ref with initial θ
Set hyperparameters: group size G, KL weight β, L_PT weight λ, learning rate α

for each training iteration:
  // --- GRPO Part (similar to Algorithm 1) ---
  Sample a batch of prompts D_b from dataset D
  Trajectories = []
  for each prompt q in D_b:
    Outputs_q = [π_θ.generate(q) for _ in range(G)]
    Rewards_q = [log π_T(o_j | q) for o_j in Outputs_q]
    μ_q = mean(Rewards_q)
    σ_q = std(Rewards_q)
    Advantages_q = [(r_j - μ_q) / (σ_q + ε) for r_j in Rewards_q]
    for j in range(G):
      Trajectories.append((q, Outputs_q[j], Rewards_q[j], Advantages_q[j]))

  // Compute GRPO policy loss and KL loss
  L_policy = 0
  L_kl = 0
  for (q, o, r, A) in Trajectories:
    log_prob_student = log π_θ(o | q)
    L_policy += - A * log_prob_student
    log_prob_ref = log π_ref(o | q)
    L_kl += (log_prob_student - log_prob_ref)
  L_policy = L_policy / len(Trajectories)
  L_kl = L_kl / len(Trajectories)

  // --- L_PT Part ---
  Sample a batch of pre-training text D_pt_b from D_PT
  L_pt = 0
  for text_sequence d in D_pt_b:
    // Compute negative log-likelihood on pre-training data
    L_pt += - log π_θ(d)
  L_pt = L_pt / len(D_pt_b)

  // --- Total Loss ---
  // Combine GRPO loss components with L_PT
  Loss = L_policy + β * L_kl + λ * L_pt

  // 5. Update student policy parameters θ using gradient descent
  ∇θ = compute_gradients(Loss, θ)
  θ = update_parameters(θ, ∇θ, α)

  // Optional: Update reference policy π_ref periodically

end for
3. PPO with Teacher Log-Likelihood RewardThis adapts the teacher log-likelihood reward for the standard PPO algorithm, which includes a learned value function (critic).Algorithm 3: PPO with Teacher Log-Likelihood Reward

Initialize student policy parameters θ
Initialize value function parameters φ
Initialize reference policy π_ref with initial θ (or use π_θ_old from previous iteration)
Set hyperparameters: PPO clip ε_clip, KL weight β (optional, often part of reward), value loss coeff c_vf, entropy coeff c_ent, learning rate α, epochs K, minibatch size M

for each training iteration:
  // --- Data Collection Phase ---
  Sample a batch of prompts D_b from dataset D
  Rollouts = [] // Store (prompt, output, reward, log_prob_old, value_estimate)

  for each prompt q in D_b:
    // 1. Generate response using the *current* policy π_θ
    output o = π_θ.generate(q)
    log_prob_old = log π_θ(o | q) // Log prob under the policy used for generation

    // 2. Compute reward using the teacher model
    reward r = log π_T(o | q)

    // 3. Estimate value using the current value function V_φ
    value_estimate v = V_φ(q) // Or V_φ(state representation)

    Rollouts.append((q, o, r, log_prob_old, v))

  // --- Advantage Calculation ---
  Compute advantages A_t and returns G_t for each step in Rollouts
  // (e.g., using Generalized Advantage Estimation - GAE)
  Advantages = compute_gae(Rollouts, V_φ, γ, λ_gae)
  Returns = compute_returns(Rollouts, γ) // Target for value function

  // --- Optimization Phase ---
  for epoch in range(K): // Multiple optimization epochs
    Sample minibatches from Rollouts, Advantages, Returns
    for minibatch (q_mb, o_mb, r_mb, logp_old_mb, v_mb, A_mb, G_mb):

      // Compute current log probabilities and value estimates
      logp_curr_mb = log π_θ(o_mb | q_mb)
      v_curr_mb = V_φ(q_mb)
      entropy_mb = compute_entropy(π_θ(o_mb | q_mb))

      // 1. Compute PPO Policy Loss (Clipped Surrogate Objective)
      ratios = exp(logp_curr_mb - logp_old_mb)
      surr1 = ratios * A_mb
      surr2 = clip(ratios, 1 - ε_clip, 1 + ε_clip) * A_mb
      L_policy = - mean(min(surr1, surr2))

      // 2. Compute Value Function Loss
      L_value = mean((v_curr_mb - G_mb)^2) // Mean Squared Error

      // 3. Compute Entropy Bonus (encourages exploration)
      L_entropy = - mean(entropy_mb)

      // (Optional) KL penalty term if not included in reward
      // L_kl = mean(KL(π_θ(.|q_mb) || π_ref(.|q_mb)))

      // Total PPO Loss
      Loss = L_policy + c_vf * L_value + c_ent * L_entropy // + β * L_kl if used

      // Update policy θ and value function φ using gradient descent
      ∇θ, ∇φ = compute_gradients(Loss, θ, φ)
      θ = update_parameters(θ, ∇θ, α)
      φ = update_parameters(φ, ∇φ, α)

  // Update reference policy if needed (e.g., π_ref = π_θ before next iteration)

end for
4. PPO with MiniLLM-style Reward (Teacher LL + L_{PT})This integrates the combined Teacher LL + L_{PT} reward into the PPO framework. The L_{PT} term can be added directly to the reward signal or handled as a separate loss term during optimization. Here, we add it to the reward.Algorithm 4: PPO with MiniLLM-style Reward

Initialize student policy parameters θ
Initialize value function parameters φ
Initialize reference policy π_ref with initial θ
Set hyperparameters: PPO clip ε_clip, L_PT weight λ, value loss coeff c_vf, entropy coeff c_ent, learning rate α, epochs K, minibatch size M

for each training iteration:
  // --- Data Collection Phase ---
  Sample a batch of prompts D_b from dataset D
  Sample a batch of pre-training text D_pt_b from D_PT (match size or handle differently)
  Rollouts = []

  for i, prompt q in enumerate(D_b):
    // 1. Generate response using the current policy π_θ
    output o = π_θ.generate(q)
    log_prob_old = log π_θ(o | q)

    // 2. Compute base reward using the teacher model
    reward_teacher = log π_T(o | q)

    // 3. Compute L_PT term for this step (can be tricky for sequence models)
    // Option A: Use L_PT on a separate batch and add its negative as a global reward bonus (simpler)
    // Option B: Calculate L_PT contribution *related* to this specific generation (complex)
    // Let's assume Option A for simplicity: Calculate average L_PT over D_pt_b once per iteration
    // (Calculation shown in Optimization Phase for clarity)

    // Store intermediate reward (without L_PT for now)
    reward_base = reward_teacher

    // 4. Estimate value using the current value function V_φ
    value_estimate v = V_φ(q)

    Rollouts.append((q, o, reward_base, log_prob_old, v))

  // --- Compute L_PT Loss/Reward Bonus ---
  L_pt_avg = 0
  for text_sequence d in D_pt_b:
    L_pt_avg += - log π_θ(d)
  L_pt_avg = L_pt_avg / len(D_pt_b)
  Reward_pt_bonus = - λ * L_pt_avg // Negative loss becomes a reward bonus

  // Add L_PT bonus to all rewards in the rollouts
  // Note: This is a simplified way; more sophisticated methods might exist.
  for rollout in Rollouts:
     rollout.reward += Reward_pt_bonus // Modify reward in-place or create new list

  // --- Advantage Calculation ---
  // Compute advantages and returns using the *modified* rewards
  Advantages = compute_gae(Rollouts, V_φ, γ, λ_gae) // Uses rewards including L_PT bonus
  Returns = compute_returns(Rollouts, γ)

  // --- Optimization Phase ---
  for epoch in range(K):
    Sample minibatches from Rollouts, Advantages, Returns
    for minibatch (q_mb, o_mb, r_mb, logp_old_mb, v_mb, A_mb, G_mb):
      // Compute current log probabilities, value estimates, entropy
      logp_curr_mb = log π_θ(o_mb | q_mb)
      v_curr_mb = V_φ(q_mb)
      entropy_mb = compute_entropy(π_θ(o_mb | q_mb))

      // 1. Compute PPO Policy Loss
      ratios = exp(logp_curr_mb - logp_old_mb)
      surr1 = ratios * A_mb
      surr2 = clip(ratios, 1 - ε_clip, 1 + ε_clip) * A_mb
      L_policy = - mean(min(surr1, surr2))

      // 2. Compute Value Function Loss
      L_value = mean((v_curr_mb - G_mb)^2)

      // 3. Compute Entropy Bonus
      L_entropy = - mean(entropy_mb)

      // Total PPO Loss (L_PT is implicitly included via the modified rewards/advantages)
      Loss = L_policy + c_vf * L_value + c_ent * L_entropy

      // Update policy θ and value function φ
      ∇θ, ∇φ = compute_gradients(Loss, θ, φ)
      θ = update_parameters(θ, ∇θ, α)
      φ = update_parameters(φ, ∇φ, α)

end for
These pseudocode examples provide a blueprint for implementing GRPO and PPO with rewards derived from knowledge distillation principles. The specific implementation details (like GAE calculation, exact KL divergence computation, handling sequence generation) would require further refinement based on the chosen framework (e.g., PyTorch, TensorFlow) and model architecture.