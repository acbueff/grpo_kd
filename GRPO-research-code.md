Excellent, I’ll get started on developing novel algorithmic extensions to PPO and GRPO for LLM fine-tuning across multiple domains, incorporating knowledge distillation from open-source teacher models. I’ll focus on dense and MoE autoregressive architectures like Qwen 2.5 and LLaMA, and I’ll include full mathematical formulations, detailed pseudocode, and a rigorous evaluation framework aimed at NeurIPS, ICLR, and ICML standards.

I’ll let you know once the full research package is ready for review.

# Extensions of PPO and GRPO for Multi-Domain LLM Fine-Tuning with Knowledge Distillation

## 1. PPO and GRPO in LLM Alignment: Technical Analysis

**Proximal Policy Optimization (PPO).** PPO is a stable policy-gradient algorithm widely used for aligning LLMs via reinforcement learning from human feedback (RLHF) ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=Proximal%20Policy%20Optimization%20%28PPO%29%20,PPO%20can%20be%20written%20as)). For a policy $\pi_\theta$ (LLM with parameters $\theta$) and an old policy $\pi_{\theta_{\text{old}}}$, the PPO objective (per time-step or token) is: 

\[ 
J_{\text{PPO}}(\theta) \;=\; \mathbb{E}_{(s,a)\sim \pi_{\theta_{\text{old}}}}\!\Big[\;\min\!\Big(r_t(\theta)\,A_t,\; \text{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\,A_t\Big)\Big]\,,
\] 

where $r_t(\theta)=\frac{\pi_\theta(a_t\,|\,s_t)}{\pi_{\theta_{\text{old}}}(a_t\,|\,s_t)}$ is the probability ratio for action $a_t$ (e.g. choosing a token) in state $s_t$ (e.g. the prompt context plus previously generated tokens), and $A_t$ is the advantage estimate at $t$ ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=Proximal%20Policy%20Optimization%20%28PPO%29%20,PPO%20can%20be%20written%20as)). The $\min(\cdot,\cdot)$ with clipping factor $1\pm\epsilon$ constrains policy updates, preventing overly large changes that could destabilize training ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=Proximal%20Policy%20Optimization%20%28PPO%29%20,PPO%20can%20be%20written%20as)). In LLM alignment, $A_t$ is often computed via a learned **value function** $V_\psi(s_t)$ (with parameters $\psi$) using Generalized Advantage Estimation (GAE) ([](https://arxiv.org/pdf/2402.03300#:~:text=where%20%F0%9D%90%B4%F0%9D%91%A1is%20the%20advantage%2C%20which,a%20learned%20value%20function%20%F0%9D%91%89%F0%9D%9C%93)). For example, if a scalar reward $R$ is obtained at the end of a generated sequence (e.g. a score from a reward model for the entire response), one can propagate it to each token with discount $\gamma$ and optionally use GAE to reduce variance:
\[ A_t = \sum_{t'=t}^T \gamma^{\,t'-t} \,r_{t'} - V_\psi(s_t)\,, \] 
with $r_{t'}$ being intermediate rewards (often zero except possibly a final reward at $T$). The policy $\pi_\theta$ (actor) and value $V_\psi$ (critic) are optimized jointly: the actor via the PPO loss above, and the critic via an MSE loss $\frac{1}{2}(V_\psi(s_t) - \sum_{t'\ge t}\gamma^{\,t'-t}r_{t'})^2$. 

**Strengths:** PPO’s clipped objective yields reliable convergence and prevents divergence of the LLM policy ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=Proximal%20Policy%20Optimization%20%28PPO%29%20,PPO%20can%20be%20written%20as)), which is crucial when fine-tuning large models on delicate alignment rewards. PPO’s use of a critic to estimate baseline improves sample-efficiency by reducing the variance of policy gradient estimates. In practice, PPO has been effective in aligning LLMs with human preferences (as evidenced by its use in training models like ChatGPT). It can handle long sequences by summing per-token loss contributions, and the learned value function can model long-term rewards, aiding credit assignment over many generated tokens.

**Limitations:** Training a separate value network for an LLM is **computationally expensive** – roughly doubling forward/backprop costs and memory usage (one forward/backpass for the actor and one for the critic per update). The **critic may be unreliable** early in training or in novel domains, leading to biased advantage estimates. In multi-domain settings, a single value function may have difficulty modeling reward distributions across heterogeneous tasks (e.g. code vs. dialogue), potentially harming learning if not carefully tuned. PPO’s stability comes at the cost of introducing a clipping hyperparameter $\epsilon$ and often a KL-divergence penalty to an initial policy to further rein in updates – these require careful balancing. **Computational complexity:** Each PPO update on a batch of size $B$ and sequence length $T$ has time complexity $O(B\cdot T \cdot |\theta|)$ for forward/backprop on the policy (with $|\theta|$ the number of parameters) and $O(B\cdot T \cdot |\psi|)$ for the value network. Memory complexity is also high since activations for both networks must be stored. For LLMs, $|\theta|$ is huge (billions of parameters), so PPO fine-tuning is typically done with small batch sizes and truncated sequence lengths to fit in memory. 

**Group Relative Policy Optimization (GRPO).** GRPO is a recent variant of PPO tailored for LLM training ([](https://arxiv.org/pdf/2402.03300#:~:text=Furthermore%2C%20we%20introduce%20the%20Group,Instruct%2C%20including%20both)) ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=1,the%20objective%20and%20the%20rewards)). GRPO’s key innovation is to **forego the learned critic**, instead deriving the baseline (advantage) from a *group of sampled outputs* for the same prompt ([](https://arxiv.org/pdf/2402.03300#:~:text=GRPO%20foregoes%20the%20critic%20model%2C,domain%20mathematical)). In GRPO, for each query (state) $q$, we sample a *group* of $G$ responses $\{o_i\}_{i=1}^G$ from the current policy (or a reference policy) to obtain a set of reward scores $\{r_i\}_{i=1}^G$ (e.g. from a reward model). These scores allow computation of a **group baseline**: for example, the mean $\bar{r}=\frac{1}{G}\sum_{i=1}^G r_i$ can serve as a baseline, and the *relative* advantage for each output $i$ is $\tilde{A}_i = r_i - \bar{r}$. In practice, Shao *et al.* (2024) normalize the rewards by the group’s standard deviation as well ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=normalizing%20a%20group%20,)) ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5BJ_%7BGRPO%7D%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BG%7D%5Csum_%7Bi%3D1%7D,ref%7D%5D%5Cright)), defining: 
\[ \hat{A}_i \;=\; \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}\,, \] 
and assigning this **same advantage** $\hat{A}_i$ to every time-step $t$ of the $i$-th sequence (assuming the reward is a final score for the whole sequence) ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=normalizing%20a%20group%20,)). Intuitively, each sampled response’s reward is measured relative to other *peer* responses to the same prompt – this **relative advantage estimation** obviates the need for a separate value model and keeps the expected advantage zero-centered within each group, stabilizing updates.

The GRPO objective extends PPO by averaging over the $G$ sampled trajectories and including a KL penalty to a reference policy $\pi_{\text{ref}}$ (often the initial SFT model) for conservativeness ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5BJ_%7BGRPO%7D%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BG%7D%5Csum_%7Bi%3D1%7D,ref%7D%5D%5Cright)). Formally, with $\pi_{\theta_{\text{old}}}$ as the behavior policy for sampling (similar to PPO), GRPO’s objective for a single prompt $q$ with sampled outputs $a_i$ is ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5BJ_%7BGRPO%7D%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BG%7D%5Csum_%7Bi%3D1%7D,ref%7D%5D%5Cright)):

\[ 
J_{\text{GRPO}}(\theta) \;=\; \frac{1}{G}\sum_{i=1}^G \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \Big\{ \min\!\Big[r_{i,t}(\theta)\,\hat{A}_i,\;\text{clip}\big(r_{i,t}(\theta),1-\epsilon,1+\epsilon\big)\,\hat{A}_i\Big] \;-\; \beta\,D_{\mathrm{KL}}\!\big[\pi_\theta(\cdot|s_{i,t}) \,\big\|\, \pi_{\text{ref}}(\cdot|s_{i,t})\big] \Big\}\,. 
\] 

Here $r_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i,t})}$, and $s_{i,t}=(q,\;a_{i,<t})$ is the state (prompt plus partial output) at time $t$ for sequence $i$. The term $D_{\mathrm{KL}}[\pi_\theta \| \pi_{\text{ref}}]$ is the KL-divergence between the current policy and a reference policy at that state, and $\beta$ is a coefficient controlling how strongly the update is regularized to not deviate from the reference distribution ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5BJ_%7BGRPO%7D%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BG%7D%5Csum_%7Bi%3D1%7D,ref%7D%5D%5Cright)). This acts as an extra conservative push, beyond PPO’s clipping, to maintain the new policy close to the reference (e.g. the original model before RL) unless reward gains justify deviation. In practice, this KL penalty can also be implemented as a penalty on the reward at each token: $r_{i,t} = r_{\phi}(q, a_{i}) - \beta \log\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\text{ref}}(a_{i,t}|s_{i,t})}$ ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5Br_t%20%3D%20r_%5Cphi%28q%2Ca_%7B%5Cleq%20t%7D%29%20,t)), meaning the reward model’s score $r_{\phi}$ for the whole sequence is diminished if at token $t$ the policy strays from reference (this yields the same gradient as the explicit KL term). 

**Strengths:** GRPO eliminates the value network, **greatly reducing memory and compute overhead** ([](https://arxiv.org/pdf/2402.03300#:~:text=GRPO%20foregoes%20the%20critic%20model%2C,domain%20mathematical)) ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=1,the%20objective%20and%20the%20rewards)). This is especially advantageous for *very large* or mixture-of-experts LLMs where duplicating the model for a critic is impractical. The **group-based advantage** provides an *on-the-fly baseline* that automatically adapts to each query: even if reward scales differ across domains or prompts, each group’s internal normalization stabilizes the training. By *further penalizing deviations* via the KL term, GRPO often takes more conservative steps than PPO ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=1,the%20objective%20and%20the%20rewards)), which can improve training stability on sensitive alignment tasks (e.g. preventing the model from going off-distribution in pursuit of reward). Empirically, GRPO has been shown to **enhance complex reasoning abilities** of LLMs: for instance, applying GRPO to a math-focused LLM (DeepSeek-Math) yielded notable performance boosts on mathematical reasoning benchmarks (GSM8K and MATH) compared to PPO, while using significantly less memory ([](https://arxiv.org/pdf/2402.03300#:~:text=GRPO%20foregoes%20the%20critic%20model%2C,We)). This suggests GRPO’s relative advantage approach is effective at fine-tuning LLMs on difficult tasks where calibrated rewards are crucial.

**Limitations:** GRPO relies on *sampling multiple outputs per prompt* to compute advantages, which increases the per-step inference cost by a factor of $G$. For example, if $G=8$ responses are needed to get a stable baseline, the training must generate 8 times as many tokens for each prompt compared to PPO (which typically samples 1 trajectory per environment step). This can raise wall-clock time or require parallel generation on more GPUs. The **variance in advantage estimates** is reduced relative to single-sample (thanks to group normalization), but if $G$ is small or the reward model is very noisy, the baseline may still be crude. In edge cases, having no learned value function could hurt *generalization of the baseline*: a critic can, in principle, learn to predict reward from state and generalize across similar states, whereas GRPO’s baseline is non-parametric and local to each prompt group. Thus, if many prompts share common structure where a learned baseline would help, GRPO might need larger $G$ or risk unstable advantages. Another consideration is that GRPO as formulated assumes a *final reward per sequence*. If intermediate token-level rewards are available or needed (e.g. to reward each correct step in reasoning), GRPO could be extended to normalize at each step, but this is non-standard. **Complexity:** Each update requires $G$ forward passes for generation per prompt. With batch size $B$ (prompts), that is $B\times G$ sequences generated. The backward pass still scales with $B \times G \times T$ similar to PPO (since all those tokens contribute to loss), but *no separate critic backward pass* is needed. Memory-wise, storing $G$ sequences’ activations can be heavy; however, in practice one can generate sequentially and discard tokens after computing loss gradients, or use model parallelism. The overall computation roughly matches PPO if $G$ is chosen such that $G \approx$ the ratio of (critic cost)/(single sample cost). For example, if doubling compute for a critic is instead invested in doubling $G$, GRPO achieves similar cost but uses it for sampling rather than value learning. GRPO’s KL evaluation against a reference policy adds a small overhead (one extra forward of the reference per token if computed on the fly, or one can reuse the policy logits of a frozen reference model). In summary, GRPO trades off critic computation for more sampling. On *dense* LLMs like LLaMA-65B, this trade can be favorable (memory saved from not training a value head can be used to increase batch or sequence lengths). On *Mixture-of-Experts (MoE)* models (e.g. a Qwen variant with MoE), avoiding a critic also avoids doubling the gating and expert utilization complexities – one can focus on the single actor which already may activate multiple experts. However, sampling $G$ sequences on an MoE model might be expensive if each triggers different experts; caching or parallelizing can mitigate this.

**Comparison Summary:** Both PPO and GRPO seek to maximize expected reward while keeping policy updates under control (PPO via clipping, GRPO via clipping + KL). PPO uses a learned baseline (critic) and typically one (or a few) trajectories per prompt; GRPO uses multiple trajectories and a group baseline, removing the critic. PPO’s advantage is in potential sample efficiency and well-understood theory (with GAE, value function, etc.), while GRPO’s advantage is reduced resource use and robust relative rewards that adapt per-instance. For multi-domain LLM alignment, **PPO’s critic might struggle** to capture all domains’ reward nuances but can leverage shared structure across domains (if, say, factual QA and dialogue share some value features). **GRPO provides per-prompt calibration**, which naturally handles varying reward scales between domains (each prompt’s group normalizes it). On the flip side, PPO can reuse knowledge via the critic (e.g. a critic may learn that “correct code” usually yields high reward regardless of prompt, aiding code domain even on new prompts), whereas GRPO must sample enough variants each time to infer that. In practice, one might choose PPO if memory permits and if a unified value function is plausible, but choose GRPO for larger models or when reward normalization is paramount (e.g. dealing with different human preference distributions across tasks). Both approaches have been successfully applied to dense LLMs like LLaMA, and the GRPO approach is explicitly designed to improve *reasoning* alignment in LLMs ([](https://arxiv.org/pdf/2402.03300#:~:text=GRPO%20foregoes%20the%20critic%20model%2C,domain%20mathematical)). These serve as the foundation that we will extend with *knowledge distillation* and *multi-domain* innovations.

## 2. Knowledge-Distillation-Based Reward Functions

A central idea of this work is to incorporate **knowledge distillation (KD)** from one or more high-quality *teacher* models into the reward mechanism for RL fine-tuning. Traditionally, knowledge distillation transfers knowledge by training the student to match the teacher’s output distributions (minimizing a divergence like KL or MSE on logits) ([Awesome-Knowledge-Distillation-of-LLMs/README.md at main](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs/blob/main/README.md#:~:text=Awesome,role%20in%20transferring%20advanced)). Here, instead of (or in addition to) a static loss, we design **reward functions** that use teacher model signals to guide the student policy. By using teacher knowledge as part of the reward, the student can *explore* outputs and get feedback that reflects not only a generic preference score but also alignment with the teacher’s expertise. We propose two novel KD-based reward functions:

### 2.1 Reward Function 1: **Teacher Distribution Alignment (TDA) Reward**

**Definition:** The Teacher Distribution Alignment reward directly measures the divergence between the student’s output distribution and the teacher’s output distribution for a given context. Formally, let $T(x)$ be the teacher policy (assumed fixed) and $\pi_\theta(x)$ the student. For a prompt or state $s$ (e.g. an input or conversation context), define the reward for a full generated sequence $y$ (with tokens $y_1,\dots,y_n$) as: 

\[ 
R_{\text{TDA}}(s,y) \;=\; -D_{\mathrm{KL}}\!\big( \pi_T(\cdot\,|\,s) \,\big\|\, \pi_\theta(\cdot\,|\,s) \big)\,. 
\] 

Equivalently, this can be expressed as the **negative KL divergence** between the teacher’s next-token distribution $P_T$ and the student’s $P_\theta$ at each time step, summed across the sequence: 

\[ 
R_{\text{TDA}}(s,y) \;=\; \sum_{t=1}^{|y|} \Big( \mathbb{E}_{a \sim \pi_T(\cdot|s,y_{<t})}\big[\log \pi_T(a|s,y_{<t}) - \log \pi_\theta(a|s,y_{<t})\big] \Big)\,. 
\] 

In practice, a simpler proxy is often used: **log-likelihood under the teacher**. We can define $R_{\text{TDA}}(s,y)$ as the teacher’s log-probability assigned to the student’s generated sequence (or a normalized version thereof). For instance:

\[ 
R_{\text{TDA}}(s,y) \;=\; \frac{1}{|y|}\sum_{t=1}^{|y|} \log P_T(y_t \mid s,\; y_{<t}) \,,
\] 

which is the average cross-entropy (negative per-token loss) of the student’s output under the teacher. A higher reward means the teacher finds the student’s choices likely, implying the student’s behavior aligns with the teacher’s. This reward leverages the full *distributional knowledge* of the teacher, not just a single outcome: if the student chooses a token that the teacher model strongly would not choose (e.g. a factual error or awkward phrasing), the teacher’s probability $P_T(y_t|...)$ will be low, heavily penalizing the reward. On the other hand, if the student closely tracks the teacher’s probable continuations, it gets a high reward. 

**Knowledge Transfer Mechanism:** This reward effectively *distills* the teacher’s knowledge by encouraging the student to mimic the teacher’s policy. It is akin to the classic knowledge distillation loss (teacher-student KL) but framed as an RL reward that the student policy seeks to maximize. Unlike standard offline distillation where the student passively imitates teacher outputs, here the student can **actively explore** variations and receive a reward signal guiding it toward the teacher’s mode of behavior. This addresses exposure bias: the student is generating its own sequence and only gets high reward if that sequence is one the teacher would also likely produce, thereby learning to stay in the teacher’s distribution even when it is driving. Prior work has shown that using a teacher’s policy as a reward signal in RL can outperform direct supervised distillation, mitigating exposure bias and improving sequence-level performance ([LLMR: Knowledge Distillation with a Large Language Model-Induced Reward](https://aclanthology.org/2024.lrec-main.932.pdf#:~:text=We%20conducted%20experiments%20on%20two,reduc%02ing%20the%20computing%20and%20memory)). Our TDA reward is exactly such a signal—by optimizing it, the student should asymptotically match the teacher’s generative behavior on the training distribution.

**Mathematical Properties:** The gradient of the expected TDA reward for the student policy is related to a reverse-KL minimization. Specifically, if we consider an objective $J(\theta)=\mathbb{E}_{s \sim D,\; y \sim \pi_\theta}[R_{\text{TDA}}(s,y)]$, the policy gradient (ignoring baseline) is 
\[ \nabla_\theta J \approx \mathbb{E}_{s,\,y \sim \pi_\theta} \sum_{t} \big(- \nabla_\theta \log \pi_\theta(y_t|s,y_{<t})\big)\, \big(\log \pi_T(y_t|s,y_{<t}) - \log \pi_\theta(y_t|s,y_{<t})\big)\,. \] 
This has a form similar to the gradient of $-D_{\mathrm{KL}}(\pi_T\|\pi_\theta)$, pushing $\pi_\theta$ to increase probability on tokens where it is lower than $\pi_T$. In effect, maximizing TDA reward minimizes the KL from teacher to student, i.e. aligns student to teacher. One must be careful to combine this with other rewards (see below) to avoid the student collapsing to only mimic the teacher without regard to human preference or task reward signals.

**Computational Complexity:** Implementing TDA reward requires access to the teacher model’s output probabilities. During training, for each token the student generates, we compute $P_T(y_t|s,y_{<t})$. If the teacher is a large model (e.g. 65B parameter LLaMA or similar), doing this in real-time is expensive. One way to manage this is to generate a *teacher trajectory* $y^{(T)}$ first (the teacher’s own sample or greedy output for prompt $s$), and then evaluate the student’s output against that. A simpler approximation is to only use the teacher’s final output $y^{(T)}$ as a reference and define a reward based on similarity between $y$ and $y^{(T)}$ (we discuss such a method next). However, the full distribution method uses more of the teacher’s knowledge than a single sample. We might distill a smaller teacher or a distilled version for computing $P_T$. Complexity-wise, if the teacher and student are similar scale, running both per token roughly doubles the forward pass cost. This can be mitigated by computing the teacher’s logits in parallel with the student (if one has model parallel capacity to host both), or by precomputing teacher token log-probs for a given prompt up to certain length (e.g. using teacher to generate a policy trajectory and reuse those probabilities whenever student matches that trajectory). Alternatively, we could only apply the TDA reward on certain important tokens or at the sequence level by comparing perplexities. For analysis, TDA introduces $O(|\theta_T|)$ extra compute per token (where $|\theta_T|$ is teacher’s parameter count). If teacher and student are both large, this is significant. One could use a **smaller teacher** model (e.g. if an open 30B model is teacher for a 7B student) to alleviate cost. Overall, the TDA reward is powerful but may be best used in an offline or guided setting where teacher distributions on many prompts are computed once, or with a distilled teacher model for efficiency.

### 2.2 Reward Function 2: **Teacher Answer Agreement (TAA) Reward**

**Definition:** The Teacher Answer Agreement reward provides feedback based on how closely the student’s *output content* agrees with the teacher’s output on the same input. Instead of using the full probability distribution, this reward looks at the teacher’s *preferred answer* and compares the student to it. This is especially useful in tasks with a clear end result (e.g. question-answering, math problem solving, code generation). We define it as:

\[ 
R_{\text{TAA}}(s,y) \;=\; f\!\big(y,\; y^{(T)}\big)\,,
\] 

where $y^{(T)} = \underset{y}{\operatorname{argmax}}\,P_T(y|s)$ (or more practically, a high-quality sample from $T$ given $s$) is the teacher’s answer to prompt $s$, and $f(y,y^{(T)})$ is a **similarity or correctness function** comparing the student’s output $y$ with the teacher’s output $y^{(T)}$. We design $f$ to be high when $y$ is semantically or exactly similar to $y^{(T)}$ and low when they differ significantly. Possible choices for $f$ include: 

- **Exact Match or Overlap:** $f(y,y^{(T)}) = \mathbb{1}\{y = y^{(T)}\}$ (an indicator for exact string match) or a normalized text similarity score (e.g. BLEU or ROUGE if $y^{(T)}$ is viewed as a reference). This is a hard measure, appropriate when the teacher’s answer is considered ground-truth (e.g. in math or QA tasks where the teacher is accurate).
- **Semantic Similarity:** $f(y,y^{(T)}) = \text{sim}(y, y^{(T)})$ where sim could be cosine similarity between embeddings of $y$ and $y^{(T)}$ (using a language model or encoder). This gives a softer, continuous reward that can capture partial credit when the student output is similar in meaning to the teacher’s.
- **Partial Credit by Structure:** For code, $f$ might be a unit test pass rate if we consider the teacher’s solution as ground truth. For reasoning, $f$ might look at whether the final answer numbers match or steps of reasoning contain the same logical steps.

In general, TAA treats the **teacher as a source of target outputs** – effectively turning the teacher into a surrogate for ground-truth. This is most straightforward when the teacher is known to produce correct or high-quality responses for the task (which is likely if we have an expert teacher model or ensemble).

**Knowledge Transfer Mechanism:** TAA reward **transfers knowledge in a result-oriented manner**. By rewarding the student for matching the teacher’s answer, we encourage the student to arrive at the same conclusions or solutions as the teacher. This can inject factual or logical correctness from the teacher into the student. For instance, if the teacher is a factual Q&A system with access to a knowledge base, its answer $y^{(T)}$ will likely be correct; the student, lacking direct access, is rewarded for producing that correct answer on its own. Over many examples, the student internalizes the teacher’s outputs. Unlike classic knowledge distillation which might force the student to predict the *way* the teacher writes the answer (which TDA does to some extent), TAA cares only about the end result, allowing the student to potentially arrive there via a different wording or path, as long as it agrees with the teacher on key content. This is useful in domains like *creative writing* or *dialogue*, where we might want the student to preserve facts or decisions the teacher would make, but still allow originality in phrasing. The teacher’s output serves as a **knowledge anchor**.

**Mathematical Examples:** A simple instantiation: if $y^{(T)}$ is a sequence of tokens (the teacher’s answer), one could define 
\[ f(y,y^{(T)}) = \frac{1}{L}\sum_{t=1}^L \mathbb{1}\{y_t = y^{(T)}_t\} \] 
for the first $L$ tokens of the shorter sequence (an overlap fraction). This gives a token-level agreement score. However, exact token matches are too strict for natural language; a more robust choice is needed. We could use an **edit distance** based reward: e.g. $f(y,y^{(T)}) = -\text{LevenshteinDistance}(y, y^{(T)})$, which is higher (less negative) when outputs are closer. This is still non-differentiable, but in RL that’s fine (the gradient flows from it as a scalar reward signal). A differentiable approximation could use a differentiable BLEU or a learned scorer.

**Complexity:** Using TAA in training means that for each prompt $s$, we must first obtain $y^{(T)}$ (the teacher’s answer). This can be done by precomputing teacher answers for all training prompts (offline) or generating on the fly (online). If done offline, the cost is a one-time $O(N \cdot |\theta_T| \cdot L)$ for $N$ prompts of length $L$. If online, each training iteration needs to generate a teacher answer for some prompts, which doubles the generation cost similar to TDA’s cost. However, since $y^{(T)}$ can be reused if prompts repeat (and often the training dataset can be fixed), offline generation is feasible. The function $f(y,y^{(T)})$ is typically cheap (string comparison or embedding similarity). One must ensure the teacher answers are high-quality; if the teacher sometimes errs, the student might learn those errors. One mitigation is using an **ensemble of teachers** or a verification step to ensure $y^{(T)}$ is likely correct (for example, only trust the teacher if it has high confidence or cross-verify with a second model). That could be reflected in the reward (perhaps give partial reward if teacher is uncertain). But assuming a strong teacher, TAA provides a **sparse but high-quality signal**: it doesn’t give feedback on every token like TDA does, but a positive reward only when the whole answer is aligned. This can be combined with other rewards for shaping.

**Relation to Preference Models:** Notably, using a teacher’s answer as a target is somewhat analogous to how RLHF uses a reference “ideal” answer to train a reward model or to directly fine-tune (as in DPO or RLAIF). The difference is we are using an actual teacher model’s output as a stand-in for the ideal. In fact, one could train a separate reward model $r_\phi(s,y)$ to score how similar $y$ is to the teacher’s answer and use that as $R_{\text{TAA}}$. This would smooth out the reward surface. For theoretical analysis, if the teacher’s answer is indeed the correct one, then maximizing TAA reward should push the student’s policy toward outputting correct answers. Over a diverse multi-domain set of tasks, this effectively **distills the teacher’s problem-solving ability** into the student.

**Combining Distillation Rewards with Alignment Rewards:** Both TDA and TAA rewards are not ends in themselves – we integrate them into a broader RL objective. Typically, the student also receives a primary reward related to human preferences or task success (call it $R_{\text{task}}$). The KD-based rewards then act as **auxiliary rewards or constraints** to ensure the student doesn’t deviate too far from teacher knowledge while optimizing $R_{\text{task}}$. For example, the total reward might be 
\[ R_{\text{total}} = R_{\text{task}} + \lambda_1 R_{\text{TDA}} + \lambda_2 R_{\text{TAA}}\,, \] 
with $\lambda_1,\lambda_2$ hyperparameters. $R_{\text{TDA}}$ keeps the policy’s style and low-level choices in line with the teacher, and $R_{\text{TAA}}$ makes sure the high-level outputs remain correct as per teacher. This multi-term reward is novel in that it explicitly ties RL updates to a teacher model’s behavior. Recent studies have indeed suggested that large language models can serve as *implicit reward functions* to guide smaller models ([[2406.19774] Direct Preference Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2406.19774#:~:text=to%20student%20models,of%20teacher%20outputs%20over%20student)), supplementing or replacing the need for manual reward design. By formalizing that into our reward functions, we aim to harness teacher model knowledge in a principled way.

**Complexity Considerations:** The introduction of KD-based rewards does increase training complexity: either in requiring additional model inference (teacher forward passes) or precomputed data. However, these costs can be justified by improved sample efficiency – since the student gets a richer reward signal than sparse human feedback, it may converge faster to a good policy. The distillation rewards are also domain-agnostic in the sense that a strong teacher can guide the student across many domains. In multi-domain fine-tuning, one could even have *multiple teachers* (specialists for each domain) contributing to the reward. For example, for code generation tasks, use a code-specific teacher for TAA (so the reward for code tasks is based on agreement with a code expert’s output), while for dialogue use a dialogue specialist. This can be done by conditioning $R_{\text{TAA}}$ on domain: $R_{\text{TAA}}^d(s,y) = f(y, y^{(T_d)})$ where $T_d$ is the teacher for domain $d$. The student then is distilling from an ensemble of teachers. Complexity would scale with number of teachers used, but each only for relevant data. We would need to balance contributions so no single domain’s teacher overwhelms training (perhaps via weighting or curriculum).

In summary, **TDA reward** provides a fine-grained token-level incentive for the student to mimic the teacher’s policy, and **TAA reward** provides a coarse but high-value incentive to get the same final answers as the teacher. Both are novel in the context of LLM RL fine-tuning, going beyond standard KL regularization: they are essentially *learned reward functions* induced by a teacher model’s knowledge. This approach leverages the teacher as a source of truth and style, which is crucial in aligning a student LLM across multiple domains without forgetting the expertise the teacher has. These reward functions will be integrated into the algorithmic innovations described next.

## 3. Novel Algorithmic Extensions for Multi-Domain Fine-Tuning

Building on PPO and GRPO and the new reward functions, we propose three algorithmic innovations to enhance multi-domain LLM alignment:

- **(3.1) Distillation-Augmented PPO (DA-PPO):** a novel extension of PPO that incorporates knowledge distillation signals (like TDA or TAA rewards) directly into the policy update objective.
- **(3.2) Distillation-Augmented GRPO (DA-GRPO):** an extension of GRPO applying similar knowledge distillation principles in the group advantage setting, and adding mechanisms to handle multiple domains or experts.
- **(3.3) Hybrid Reinforcement Learning and Distillation Framework:** a completely new training framework that interleaves policy optimization with teacher-guided distillation in a unified algorithm, aiming to get the best of both worlds (stable RL training + direct knowledge transfer). This may involve a mixture-of-experts (MoE) architecture to specialize on domains, or a two-stage optimization that first distills and then refines with RL, integrated seamlessly.

Each innovation is detailed with modified objective functions, update equations, theoretical rationale, and computational considerations:

### 3.1 **Distillation-Augmented PPO (DA-PPO)**

**Concept:** DA-PPO modifies the standard PPO algorithm to include *knowledge distillation as an auxiliary objective*. Concretely, we augment the PPO loss with an extra term that encourages the student policy to stay close to the teacher policy, effectively blending RL with supervised distillation. The goal is to stabilize training (prevent the model from deviating from known good behaviors) and maintain performance across multiple domains by leveraging teacher knowledge as a regularizer.

**Modified Objective Function:** We define the *augmented* PPO objective for the policy as:

\[ 
J_{\text{DA-PPO}}(\theta) = \mathbb{E}_{s \sim D,\; a \sim \pi_{\theta_{\text{old}}}} \Big[ 
\min\!\big(r_\theta(s,a)A^{\text{task}}(s,a),\; \text{clip}(r_\theta(s,a),1-\epsilon,1+\epsilon)A^{\text{task}}(s,a)\big) \;+\; \alpha \, \Delta_{\text{KD}}(s;\theta) \Big]\,.
\]

Here $A^{\text{task}}(s,a)$ is the advantage with respect to the *task reward* (the usual RLHF reward or domain reward for achieving the task), and $\Delta_{\text{KD}}(s;\theta)$ is a **knowledge-distillation bonus** term. The coefficient $\alpha$ controls the strength of distillation in the objective. $\Delta_{\text{KD}}(s;\theta)$ can be defined in different ways, but a natural choice given our reward functions is **the expected TDA reward** for state $s$ under the current policy. For example:

\[ 
\Delta_{\text{KD}}(s;\theta) = -D_{\mathrm{KL}}\big(\pi_T(\cdot|s)\,\big\|\,\pi_\theta(\cdot|s)\big) \approx \mathbb{E}_{a \sim \pi_T}[ \log \pi_T(a|s) - \log \pi_\theta(a|s) ]\,,
\] 

which is exactly $R_{\text{TDA}}$ for one step (and can be extended to multi-step sequence as before). Another choice is a simpler *behavior cloning term*: $\Delta_{\text{KD}}(s;\theta) = \mathbb{E}_{a \sim \pi_T}[ \log \pi_\theta(a|s) ]$, which is the cross-entropy of the teacher’s action under the student (this omits the $\pi_T$ entropy term and focuses only on student matching teacher). We can also incorporate final outcomes: for example, after a full trajectory $y \sim \pi_\theta$, add $\alpha f(y,y^{(T)})$ as a bonus to the return (which effectively adds an advantage for matching the teacher’s answer). In practice, the simplest is to add a **KL penalty or bonus** between student and teacher policies. If we treat the teacher as a reference policy, this looks analogous to PPO’s KL regularization to the initial model, but here the reference is a more knowledgeable model. So one concrete form is:

\[ 
\Delta_{\text{KD}}(s;\theta) = -\beta_{\text{KD}} \, D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|s) \,\|\, \pi_T(\cdot|s)\big)\,,
\] 

which penalizes divergence from the teacher. Note this is a *reverse* KL (student||teacher) as opposed to the forward KL used in TDA definition; either direction can be used depending on whether we want to penalize the student placing probability mass where teacher wouldn’t (reverse KL) or vice versa. Using reverse KL (student||teacher) as shown means we give strong penalty if student predicts tokens the teacher never would (it heavily constrains student), whereas forward KL (teacher||student) as $R_{\text{TDA}}$ would encourage covering all of teacher’s likely actions. For stability, many implementations use a penalty on student||teacher (similar to how KL to a reference model is done in RLHF PPO). 

So a final form combining these ideas:
\[ 
J_{\text{DA-PPO}}(\theta) = \mathbb{E}_{s, a \sim \pi_{\text{old}}}\Big[ \min(r_\theta A^{\text{task}}, \text{clip}(r_\theta,1\pm\epsilon)A^{\text{task}}) \Big] \;-\; \beta_{\text{ref}} D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}}) \;-\; \beta_{\text{KD}} D_{\mathrm{KL}}(\pi_\theta\|\pi_T)\,. 
\]

Here we included $\pi_{\text{ref}}$ (the initial SFT model) KL as well, since often used in PPO for alignment to avoid drift (with coefficient $\beta_{\text{ref}}$), and an additional term with $\beta_{\text{KD}}$ for KL to the teacher. One could merge $\pi_{\text{ref}}$ and $\pi_T$ if the initial model itself is the teacher, but in our scenario $\pi_T$ is a separate external model. This objective is optimized with gradient ascent on $\theta$. The **policy gradient** for DA-PPO is then:

\[ 
\nabla_\theta J_{\text{DA-PPO}} = \mathbb{E}_{s, a \sim \pi_{\text{old}}} \Big[ \nabla_\theta \log\pi_\theta(a|s) \cdot g(s,a) \Big] - \beta_{\text{KD}} \nabla_\theta D_{\mathrm{KL}}(\pi_\theta\|\pi_T) - \beta_{\text{ref}} \nabla_\theta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})\,,
\] 

where $g(s,a)$ is the clipped advantage term as in PPO (i.e., $g(s,a)=\min(r_\theta A^{\text{task}}, \text{clip}(r_\theta)A^{\text{task}})$). The extra KL terms’ gradients are straightforward: $\nabla_\theta D_{\mathrm{KL}}(\pi_\theta\|\pi_T) = \mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot (1 + \log \pi_\theta(a|s) - \log \pi_T(a|s))]$. In practice, we would estimate the KL on the sampled actions or via all actions if we have logits – implementations often approximate this by just computing the KL on the policy logits as an additional loss term.

**Theoretical Justification:** DA-PPO is essentially **PPO with an auxiliary loss**. From a theoretical perspective, adding a KL to teacher can be seen as keeping the student policy within a trust region centered at the teacher policy. If the teacher is optimal or near-optimal for the given tasks, then constraining the student to remain near the teacher ensures the student doesn’t diverge into low-reward regions. In multi-domain training, this is critical: one domain’s reward might be sparse or tricky, but the teacher’s policy provides shaping. For instance, consider a domain like logical reasoning where reward is only given for a correct final answer. A student without guidance might try many faulty paths; with DA-PPO, the student is nudged to follow the teacher’s reasoning distribution, which has a higher chance of yielding correct answers, thus dramatically improving sample efficiency. This is akin to **kickstarting** in RL ([Breakthroughs in Knowledge Distillation - LinkedIn](https://www.linkedin.com/pulse/breakthroughs-knowledge-distillation-advancing-large-ramachandran-1xkqc#:~:text=Breakthroughs%20in%20Knowledge%20Distillation%20,students%20toward%20optimal%20learning%20outcomes)), where a pre-trained teacher policy helps a student learn faster. Prior works have shown that policy distillation and guided RL can reduce training time and avoid catastrophic forgetting in multi-task settings ([[PDF] Ensemble Policy Distillation in Deep Reinforcement Learning](https://pooyanfazli.com/publications/Sun_AAAI20W.pdf#:~:text=,time%20policy%20distillation%20mechanism%2C)). DA-PPO’s objective can also be interpreted in a Bayesian light: it is like doing maximum a posteriori policy optimization, where the teacher policy represents a prior distribution over good policies, and $R_{\text{task}}$ provides the likelihood of data. By maximizing reward while staying close to teacher (prior), we effectively compute a posterior policy. This yields a principled balance between exploitation of reward and exploration of known good behavior.

**Handling Multi-Domain:** In multi-domain fine-tuning, we may have different reward functions $R_{\text{task}}^d$ for each domain $d$. We can maintain separate critics (value networks) for each domain or a single unified critic with a domain indicator. DA-PPO naturally extends: the KL-to-teacher term can either use a single teacher for all domains or domain-specific teachers ($\pi_{T_d}$ for domain $d$). In the latter case, the objective would sum or average domain-specific KLs weighted by domain occurrence. For example, if $s$ comes from domain $d(s)$, then include $-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{T_{d(s)}})$. This effectively *distills multiple teachers* into one student, similar to multi-teacher distillation in supervised learning ([[PDF] Ensemble Policy Distillation in Deep Reinforcement Learning](https://pooyanfazli.com/publications/Sun_AAAI20W.pdf#:~:text=Learning%20pooyanfazli,time%20policy%20distillation%20mechanism%2C)) but here embedded in RL. The PPO clipping and advantage estimation can still be done per domain (with value nets per domain or a single value net that perhaps takes domain as input feature). The trust region mechanism of PPO combined with KL regularization tends to ensure **no single domain’s updates destabilize the policy** too much, which is important for retaining performance across domains.

**Computational Considerations:** DA-PPO requires computing the KL to teacher and possibly teacher action probabilities for each training sample. If the teacher is large, one might precompute or approximate these. One optimization is to train a smaller **teacher policy head** that mimics the large teacher on the fly (like a distilled policy that is easier to compute KL against). But assuming we directly use the teacher: memory-wise, storing teacher logits for each sample is needed to compute KL – this is $O(B \cdot |A|)$ per batch (where $|A|$ is vocab size, if we compute full KL). Often, RLHF implementations compute a moving average of KL or just sample a few actions to estimate it. Another approach is to incorporate the KL term as part of reward and let the critic approximate it, but here since teacher is fixed, directly computing is fine. 

The additional backward pass for the KL term’s gradient is negligible compared to the main policy gradient (since it’s of the same order of operations). Thus, runtime is dominated by needing teacher’s forward pass (like TDA reward, doubling the forward cost potentially). A trick: if the teacher is an earlier checkpoint of the model or a slightly larger model, it might be possible to share some computations or at least load it on the same device to avoid data transfer overhead. Another factor: the hyperparameters $\beta_{\text{KD}}$ and $\alpha$ control how aggressively we distill. Too high and the policy might ignore the task reward (just imitating teacher, which is counterproductive if teacher is not perfectly aligned with the reward preferences). Too low and the policy might diverge and forget teacher knowledge. We expect to anneal $\beta_{\text{KD}}$ over time: start high (strong distillation) to keep the student near teacher initially, then gradually lower it to allow the student to exceed the teacher if the reward suggests so. This schedule needs tuning but conceptually ensures that early in training (when the student is clueless), it essentially imitates the teacher, and later it has more freedom to improve beyond teacher (which could happen if teacher wasn’t fully optimal or if fine-tuning finds niche improvements).

**Summary:** DA-PPO is our proposed **PPO extension**. It augments the PPO loss with a knowledge distillation term (from an open-source teacher LLM). This drives the student to maintain performance across domains by constantly referring to the teacher’s policy as an anchor. It’s theoretically justified by viewing it as a multi-objective optimization (maximize reward, minimize divergence from teacher), which yields a better optimum when the teacher’s knowledge provides a good prior. We expect DA-PPO to outperform standard PPO in multi-domain settings by achieving higher reward on each domain without sacrificing any domain (reduced forgetting), and by converging faster due to guided exploration. In Section 4 we provide pseudocode for implementing DA-PPO.

### 3.2 **Distillation-Augmented GRPO (DA-GRPO)**

**Concept:** DA-GRPO extends GRPO with knowledge distillation and multi-domain awareness. Since GRPO already includes a KL to a reference model (which could be considered a kind of distillation to the initial model), our extension will incorporate an *additional* term or procedure for distilling a teacher’s knowledge. Moreover, we adapt the group advantage process for multi-domain training by possibly grouping experiences by domain or by including **teacher demonstrations in the group**. A novel idea here is to use the teacher model’s outputs as part of the group baseline computation, effectively treating the teacher as an extra “agent” whose reward sets a standard for the student.

**Modified Objective and Update:** We start from the GRPO objective (with reference KL) given earlier. To add knowledge distillation, we propose two modifications:

1. **Teacher KL Regularization:** Similar to DA-PPO, we add a KL penalty (or negative reward) between the student policy $\pi_\theta$ and teacher $\pi_T$. In GRPO’s context, this can be done by modifying the per-token reward as was done with the reference model. Recall GRPO’s per-token reward (from the blog formulation) was $r_t = r_\phi(q,a) - \beta \log\frac{\pi_\theta(a_t)}{\pi_{\text{ref}}(a_t)}$ ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5Br_t%20%3D%20r_%5Cphi%28q%2Ca_%7B%5Cleq%20t%7D%29%20,t)). We extend this to:
   \[
   r_t^{\text{aug}} = r_\phi(q,a) \;-\; \beta_{\text{ref}} \log\frac{\pi_\theta(a_t|s_{t})}{\pi_{\text{ref}}(a_t|s_{t})} \;-\; \beta_{\text{KD}} \log\frac{\pi_\theta(a_t|s_{t})}{\pi_T(a_t|s_{t})}\,. 
   \] 
   This means the reward model’s score is penalized if the student deviates from either the reference model or the teacher at that token. Equivalently, it adds a term $-\beta_{\text{KD}} \log \pi_\theta + \beta_{\text{KD}} \log \pi_T$ inside the curly braces of the $J_{GRPO}$ objective ([](https://arxiv.org/pdf/2402.03300#:~:text=%14%20%F0%9D%9C%8B%F0%9D%9C%83%28%F0%9D%91%9C%F0%9D%91%96%2C%F0%9D%91%A1%7C%F0%9D%91%9E%2C%20%F0%9D%91%9C%F0%9D%91%96%2C,%F0%9D%91%A1)). The resulting *gradient* will contain an extra term proportional to $(\pi_T/\pi_\theta - 1)$ (similar to earlier derivation) ([](https://arxiv.org/pdf/2402.03300#:~:text=%F0%9D%9C%8B%F0%9D%9C%83%F0%9D%91%9C%F0%9D%91%99%F0%9D%91%91%20%28%F0%9D%91%9C%F0%9D%91%96%2C%F0%9D%91%A1%7C%F0%9D%91%9E%2C%20%F0%9D%91%9C%F0%9D%91%96%2C,%F0%9D%91%A1%29%20%E2%88%92%201)), which pushes $\pi_\theta$ toward $\pi_T$. This is a straightforward integration of teacher regularization into GRPO’s formula.

2. **Teacher-Enhanced Group Advantage:** We propose to incorporate the teacher when computing the group baseline. One approach is to **add the teacher’s own output** for each prompt into the group of $G$ samples. That is, for each query $q$, we sample $G-1$ outputs from the student’s current policy (old parameters) and also include 1 output generated by the teacher $\pi_T$ (or perhaps the teacher’s top answer $y^{(T)}$). This makes the group size $G$ with one “external” sample. We then evaluate the reward model on all $G$ outputs (including the teacher’s output). Denote the teacher’s reward as $r_{\text{teacher}}$. When we normalize the rewards to compute advantages $\hat{A}_i = (r_i - \text{mean}(r)) / \text{std}(r)$, the teacher’s output influences the mean (and std). In particular, if the teacher’s output achieves a high reward, the mean reward of the group will shift upward, meaning the student outputs will have slightly negative advantages by comparison (unless they too get close to that reward). Essentially, the teacher’s presence in the group sets a **baseline performance level** that the student is compared against. The student will get positive advantage only to the extent it matches or exceeds the teacher’s reward. Conversely, if all student outputs are worse than the teacher’s, they all get negative advantages and the policy will be updated to increase the probability of tokens that may lead to higher reward (closer to teacher’s style). This is a form of *relative distillation*: the teacher is used as a reference point in the advantage estimation. Mathematically, if $r_T$ is the teacher reward and $r_{1..G-1}$ are student sample rewards, 
   \[
   \bar{r} = \frac{r_T + \sum_{i=1}^{G-1} r_i}{G}, \qquad \hat{A}_i = \frac{r_i - \bar{r}}{\sigma}, \qquad \hat{A}_{\text{teacher}} = \frac{r_T - \bar{r}}{\sigma}\,.
   \] 
   We do not directly update the teacher (it’s a fixed model), but the teacher’s advantage being high or zero doesn’t matter; we only use student trajectories for gradient (the expectation in $J_{GRPO}$ is still over student samples). But the baseline $\bar{r}$ including $r_T$ means the gradient for student tokens becomes $\propto (r_i - \bar{r})$, which implicitly includes $r_T$. If $r_T$ is highest, $r_i - \bar{r}$ is negative, so student log-probs are decremented (unless they achieve similarly high reward). **Theoretical note:** Including off-policy samples (teacher) in baseline keeps the policy gradient unbiased *for advantages* (since baseline can be any function) – it doesn’t introduce bias because baseline cancels out in expectation. However, it might increase variance if the teacher’s reward is an outlier. But if the teacher is consistently good, it provides a stable high reference. This technique is novel and akin to using an expert’s performance as a baseline in policy gradients, which, to our knowledge, hasn’t been explored much.

**Modified Objective:** Combining the above, the DA-GRPO objective can be expressed in a similar averaged sum as original GRPO, but with these changes. We can summarize it as:

\[ 
J_{\text{DA-GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \left\{\min\!\Big[r_{i,t}(\theta)\,\hat{A}_{i},\; \text{clip}(r_{i,t}(\theta),1\pm\epsilon)\,\hat{A}_{i}\Big] \right\}\!,
\] 

with *augmented rewards* in $r_{i,t}(\theta)$ as described. We implicitly include the KL to teacher in $r_{i,t}$ (so $\beta_{\text{KD}}$ is a hyperparameter here as well) and compute $\hat{A}_i$ with teacher sample included. Additionally, one could incorporate a **teacher-distilled value** in advantage computation: instead of pure reward model score for each sample, perhaps use a combination of reward model and teacher’s evaluation. But that complicates things; our main inclusion is the teacher’s trajectory reward as part of the baseline.

**Theoretical Justification:** DA-GRPO retains the advantages of GRPO (no critic, stability) while injecting teacher guidance. The KL regularization to teacher is similar in spirit to DA-PPO’s justification – it constrains the policy within a vicinity of the teacher’s policy, which is known to be strong. The more interesting aspect is the teacher-augmented baseline. From an advantage estimation perspective, what we’ve done is **use the teacher’s performance as a baseline for what is “good.”** In theory, if the teacher is optimal (or very high reward), the best the student can do is match the teacher, and then its advantage will be zero (if it equals teacher reward) or positive (if somehow exceeds teacher). If the teacher is suboptimal on some tasks, the student can exceed it and get positive advantage. So the student is always pushed to *at least reach the teacher’s level*. This is closely related to *learning from demonstrations* in RL: often an expert demonstration can initialize or guide policy; here it provides a baseline to surpass. An extreme view: if the teacher is perfect (and $\sigma$ is small or we remove normalization for thought experiment), then any student output with reward lower than teacher yields a negative advantage and the policy will be updated to reduce probability of its actions (because they underperformed the teacher). The only way to stop that is to equal the teacher, at which point advantage zero, no update (policy equilibrium). So the fixed point of training would be a student that matches teacher performance on all prompts – i.e. the student has been distilled to the teacher’s capabilities. If the reward model truly measures task performance, matching teacher’s reward means matching teacher’s capability. Thus, this scheme can achieve full distillation in theory. Meanwhile, if the teacher is not perfect, the RL still allows surpassing the teacher because if student finds a way to get higher reward, then those trajectories will have positive advantage (as teacher’s presence just raises the bar, but not unreachable). In summary, DA-GRPO offers a *novel convergence target*: the teacher defines a moving baseline that encourages continual improvement up to teacher’s level and possibly beyond. This is a fresh perspective compared to standard KD which just pulls toward teacher but doesn’t consider the actual task reward – here we consider both the actual reward $r_\phi$ and teacher’s guidance.

**Multi-Domain Adaptation:** GRPO is naturally robust to multi-domain reward scale differences due to per-prompt normalization. Including a teacher per domain is straightforward as well: when sampling for a prompt of domain $d$, we use the teacher specialized for $d$ (if available) as the extra sample. If we have a single general teacher for all domains, use it across. If some domain has no teacher, we could skip adding one (or use the main teacher). The KL penalty to teacher can also be domain-specific ($\beta_{\text{KD}}$ might be larger for domains where we trust the teacher more, etc.). One might also consider grouping by domain in batches to ensure each batch has a consistent teacher baseline – but since GRPO already groups by prompt, and each prompt belongs to a domain, it implicitly does that. Another innovation could be to vary the group size $G$ by domain difficulty: e.g. for very complex domains (code or math), use a larger $G$ (including perhaps multiple outputs from the teacher: maybe teacher’s top 3 answers) to get a more stable baseline. This introduces more computation but could be beneficial – essentially a mixture of experts approach where multiple teacher outputs define a distribution of high-quality answers.

**Computational Considerations:** DA-GRPO needs at least one forward pass of teacher model per prompt (for the teacher sample). That’s similar overhead to DA-PPO’s needs. The difference is in how it’s used: we don’t need to compute the full KL at every token explicitly if we include the teacher sample approach, because the teacher’s influence is partly captured via advantage. However, we still added a token-level KL penalty $\log(\pi_\theta/\pi_T)$ in rewards for precision. One might question if both are needed: possibly not. We could experiment with only teacher in baseline (which is a weaker form of distillation) versus also explicit KL (stronger constraint). Likely both yield the best stability (teacher baseline ensures asymptotic match, KL ensures per-token behavior match during learning).

Memory: storing an extra trajectory (the teacher’s) per prompt is minor overhead (one more sequence in the group). If $G$ was originally, say, 8 for GRPO, now we generate 7 student + 1 teacher = 8. We effectively maintain the same $G$. But note the teacher’s output is not from $\pi_{\text{old}}$ distribution, so strictly speaking GRPO’s expectation should consider that. We treat it as just extending baseline, which does not bias gradient as discussed. Implementation-wise, one would sample student outputs as usual, then append the teacher output, compute all rewards, then compute advantages. The gradient computation would iterate over only the student outputs for policy gradient (since teacher’s output isn’t generated by $\pi_\theta$ – one could include it with importance weighting if desired, but it’s not needed for policy update, only baseline). This is a nuance in coding to ensure we don’t mistakenly treat teacher’s actions as if taken by student.

Another consideration: what if the teacher model is *itself an MoE or extremely large?* Running it might be heavy. But perhaps one could pre-generate teacher outputs for all prompts and their reward, then just treat $r_T$ as a constant baseline (i.e. an offline oracle baseline). This would lose the per-step KL guidance though. If precomputing, we could store $y^{(T)}$ and $r_T$ for each training prompt, and for any new prompt encountered (if any) still need online generation.

**Exploration:** Including the teacher might reduce the student’s exploration, since it is always trying to match/surpass the teacher. If the teacher is somewhat wrong or suboptimal in a domain, the student might not discover better solutions easily. To counter that, we could occasionally omit the teacher baseline (like an $\epsilon$-greedy strategy: in some fraction of updates, do normal GRPO without teacher, allowing the student to explore freely; in most updates, use teacher baseline to exploit known good strategies). This kind of schedule could be mentioned as an implementation note.

In summary, **DA-GRPO** is a novel GRPO-based algorithm that integrates teacher model guidance both through a KL regularizer and through an innovative group baseline including teacher outputs. It leverages the efficiency of GRPO (no critic) and the strength of a teacher’s knowledge to align a student LLM on multiple domains effectively. We expect DA-GRPO to particularly shine in scenarios requiring reasoning or specialized knowledge (where a teacher example can illuminate the path for the student). Its computational cost is comparable to DA-PPO (slightly more sampling, but no value net), which is favorable for large models.

### 3.3 **Hybrid Reinforcement Learning and Distillation Framework (HRLD)**

**Concept:** Our third innovation is a **hybrid framework** that interleaves reinforcement learning with explicit distillation phases in a unified training loop. The motivation is to get the benefits of direct teacher supervision (which can rapidly transfer knowledge) and policy-gradient optimization (which can fine-tune toward the specific reward nuances and human preferences). Instead of optimizing one single objective that combines both (as in DA-PPO/GRPO), we alternate or blend two training modes: (A) a **distillation step** where the student learns from teacher-generated data (using supervised learning on teacher outputs), and (B) a **reinforcement step** where the student interacts with the environment/reward model and updates via PPO/GRPO on the task reward. By switching between these modes (possibly with gradually shifting emphasis), we aim to maintain teacher knowledge and multi-domain competence throughout training while also optimizing the final alignment reward.

**Framework Design:** One can view HRLD as a form of *guided curriculum*: start heavily guided by the teacher, then gradually give the student more autonomy with RL. Specifically, we propose the following procedure:

- **Initialization:** Start with a student model initialized from a base model (e.g. an SFT model or smaller model weights). Also have an available teacher model $T$ (or multiple teachers per domain). Optionally, pre-fill a replay buffer or dataset with some teacher-generated responses for various prompts in each domain.
- **Repeat for $K$ iterations:** each iteration consisting of two phases:
  1. **Distillation Phase:** Sample a batch of prompts from the multi-domain distribution. For each prompt, obtain the teacher’s preferred output $y^{(T)}$ (either by direct generation or by retrieving from a prepared dataset if available). Perform a **supervised update** on the student to maximize the likelihood of $y^{(T)}$. This is effectively minimizing cross-entropy: $L_{\text{KD-supervised}} = -\log \pi_\theta(y^{(T)}|\,s)$. We may do a few gradient steps on this loss (this resembles standard knowledge distillation training). This phase can use a larger learning rate for quick absorption of knowledge or smaller if fine-tuning gradually.
  2. **Reinforcement Phase:** Using the *updated* student from phase 1, interact with the environment or reward model. For example, sample $N$ trajectories from the student (using either on-policy sampling as in PPO/GRPO). Compute rewards for each (including the primary task reward, and potentially also teacher-based rewards too, though we can omit them here since the distillation phase handled that). Then perform an RL update (PPO or GRPO style) on those trajectories to improve the student with respect to the task reward. This update could be a few epochs of PPO on that batch or a single epoch if doing online PPO. In PPO, one might include the usual reference KL to initial model to ensure output doesn't drift too wildly from initial style.

- **End Repeat.**

Over time, we can **anneal the balance** between these two phases. In early training, put heavy emphasis on Phase 1 (distillation), ensuring the student quickly picks up multi-domain skills and general knowledge from the teacher. As training progresses, gradually reduce the weight or frequency of Phase 1 and increase Phase 2, so the student increasingly relies on RL to fine-tune preferences and can deviate from the teacher when beneficial. Eventually, Phase 1 might be used rarely, just to remind the student of any forgotten knowledge.

**Modified Objectives:** This framework doesn’t have one static objective but effectively optimizes a composite objective. We can think of it as optimizing:
\[ 
\mathcal{L}(\theta) = \mathbb{E}_{s}\big[-(1-\beta)\log \pi_\theta(y^{(T)}|s)\big] \;-\; \beta \,\mathbb{E}_{s,y \sim \pi_\theta}[R_{\text{task}}(s,y)]\,,
\] 
where $\beta$ is increasing over time (initially $\beta \approx 0$ so mostly supervised, later $\beta \to 1$ so mostly RL). This two-term objective is optimized alternatingly (which is a form of block coordinate ascent on the RL objective and descent on the distillation objective). In practice, we ensure that the supervised loss and RL loss are balanced; one can also combine them in a single loss per batch (some approaches mix supervised and RL in one mini-batch with a weighted sum of losses). But alternating provides clarity and stability (since each is handled with its specialized optimizer, e.g. supervised can use teacher forcing through the whole sequence, RL uses rollouts).

**Theoretical Justification:** The HRLD framework is inspired by the idea of **stabilizing RL with demonstrations**. In reinforcement learning theory, if you have an expert policy, you can significantly speed up learning by imitation (especially in the initial stages) – this is supported by algorithms like DAGGER (which iteratively mix expert and learner policy) and by the improvements seen when seeding RLHF with a well-supervised model. Our approach formalizes a schedule for this in the context of LLMs. Initially, by imitating the teacher on all domains, the student avoids catastrophic mistakes and mode collapse; it starts off already performing reasonably (perhaps not as good as teacher, but close). Then RL can refine it. Importantly, the interleaving ensures **continual distillation**: if at any point the student starts to deviate in a way that’s not desirable (e.g. forgetting knowledge in domain X because RL reward mostly came from domain Y), the next distillation phase on domain X examples from the teacher will pull it back. Thus, it acts as a safeguard against forgetting or against the reward model pushing the student into an unnatural regime. One can view the supervised distillation steps as **constraint projection** steps, projecting the policy back towards the set of policies that perform like the teacher on the data. This prevents drifting too far due to the sometimes noisy or biased reward signal. In optimization terms, the method resembles **alternating minimization** on a joint loss and can converge to a policy that is a compromise between pure teacher imitation and pure reward maximization. If the reward model is aligned with the teacher’s notion of good outputs, then eventually RL will not conflict with distillation and the student will match teacher on optimal behavior. If there are differences, the process finds a balance.

**Hybrid with Mixture-of-Experts:** As an optional enhancement particularly suited for multi-domain scenarios, we can integrate MoE architecture into this framework. For example, consider the student as a mixture-of-experts model with different experts potentially specializing in different domains. The teacher might also be a set of models (one per domain). In the **distillation phase**, we could route each prompt to the corresponding expert (or a subset of experts) in the student and train that part of the model on the teacher’s output for that domain. In the **reinforcement phase**, we collect trajectories possibly tagged by domain, and update the model parameters (experts and router) via RL. To facilitate this, we might include a domain token in the prompt or use the gating mechanism of the MoE to automatically pick an expert. Over training, experts would specialize: the distillation phase feeds each expert with high-quality data of its domain, and the RL phase adjusts the gating and fine-tunes expert behavior on reward specifics. This yields a **Hybrid MoE Distillation-RL algorithm**, where essentially each expert is like a mini DA-PPO focused on one domain, but they share the overall policy through gating. This could lead to better utilization of model capacity in multi-domain training (avoid interference by isolating domain-specific changes to separate experts). The pseudocode of HRLD can incorporate this by having domain-specific filtering in each phase (this is more of an architecture detail than algorithm, so we will provide general pseudocode but note where MoE could come in).

**Computational Considerations:** HRLD alternates two types of updates, which can be implemented efficiently. The distillation (supervised) updates use teacher outputs – these can be precomputed to avoid running the teacher repeatedly. For example, one could build a dataset by having the teacher generate answers for many prompts in each domain ahead of time. Then Phase 1 just samples from this dataset and does a standard language modeling training step on it. This decouples teacher computation from training loop, greatly saving time if that dataset is large enough to cover the needed distribution. The RL phase then uses the reward model; its cost is similar to normal PPO/GRPO training. So HRLD can be viewed as doing *semi-offline RL*: part of the objective (distillation) is offline on a fixed set of teacher data, part is online (on-policy rollouts for reward). This is akin to using an *experience replay* from demonstration and on-policy data. Stability-wise, we should ensure that the supervised learning rate and RL learning rate are tuned so one phase doesn’t overpower the other in oscillation. One approach is to gradually reduce the supervised learning rate as training goes on, because early on we want big jumps from teacher data, later we only want to nudge the model with teacher data if it strays. Additionally, when using pseudocode or actual implementation, we might maintain two optimizers: one for RL (often an Adam with smaller LR due to high variance gradients) and one for supervised (maybe slightly larger LR). Or we can unify them by scaling losses appropriately.

One challenge is **score normalization between phases**. RL works with rewards which might not directly align with log-likelihood losses. But since we treat them separately, it’s fine – we don’t need to combine losses in one backward pass (which would require weighting). Instead, do separate backward passes. This also means HRLD could be implemented in an alternating fashion: e.g. one epoch of supervised on teacher data, then one epoch of PPO on reward model, etc.

**Benefits:** HRLD is expected to be very effective in *low-resource domains* where human feedback is scarce but a teacher model’s knowledge exists. For example, if one domain is medical QA and we have a strong medical LLM (teacher), but our reward model for that domain is not perfect, HRLD will lean on supervised distillation from the medical LLM to teach the student, rather than solely trust possibly sparse reward signals. At the same time, in domains like casual dialogue where a teacher might not perfectly represent human style, the RL phase can adjust the student to real human preferences.

**Generalizability:** Because the framework is modular, one can plug in any teacher (including non-LLM teachers, e.g. a retrieval system that gives correct facts as “teacher output”) or multiple teachers. It is general with respect to which base RL algorithm is used in Phase 2; we could use PPO or GRPO or any policy gradient method. We chose PPO for pseudocode as it’s common, but GRPO could be used too in phase 2. In fact, one could use GRPO with teacher baseline in phase 2 *and* do supervised teacher distillation in phase 1 – that might be redundant but could yield extremely stable training (this essentially fuses all techniques). For clarity, we’ll present a PPO-based HRLD.

**Potential Downsides:** The alternate training might introduce some instability if the shifts are too large. E.g. after a big supervised update, the policy might violate the trust region assumption of PPO, so the next PPO update’s advantage estimates might be off. To mitigate, one can either do small frequent alternations (so each supervised update is tiny change) or re-calculate baselines after a supervised jump. Some research suggests doing supervised fine-tuning and RL concurrently can destabilize either if not carefully balanced. We will ensure in pseudocode to mention using small steps or mixing gradually. Also, HRLD training is a bit more involved to tune (two learning rates, a schedule for switching). However, the gains in combining methods should outweigh this complexity for a high-stakes multi-domain alignment problem.

In summary, **HRLD** is a new hybrid algorithm that merges knowledge distillation and reinforcement learning in a single workflow. It provides a clear strategy to exploit teacher knowledge fully while still optimizing the final reward. It is novel in that it doesn’t rely solely on a combined loss; instead, it treats the learning process itself as a sequence of different optimizations, which is reminiscent of human learning (study examples, then practice yourself, and repeat). We expect HRLD to result in a student model that nearly matches the teacher in general capability (through distillation) and perhaps exceeds the teacher in alignment with human preferences (through RL fine-tuning), all while being computationally efficient by reusing precomputed teacher outputs. We will now present pseudocode for DA-PPO, DA-GRPO, and HRLD algorithms, and then detail how to evaluate these methods across 7 diverse domains.

## 4. Pseudocode for Proposed Algorithms

In the pseudocode below, we use a Python-like pseudo-language for clarity. We assume the existence of the following components:

- `StudentModel` – the policy network (LLM) we are fine-tuning.
- `TeacherModel` – the teacher LLM (or a function to query it) for knowledge distillation.
- `RewardModel` – a model or function that gives a scalar reward for a (prompt, student_output) pair. This encapsulates human feedback or any task-specific reward. It may include multiple sub-rewards for different criteria.
- (For GRPO) `ReferenceModel` – the baseline reference policy (often the initial student weights) to compute KL penalty.
- Optimizers for policy (and value function if PPO uses one).
- Hyperparameters such as learning rates, batch sizes, clipping epsilon, KL coefficients, etc., which we will mention.

We provide three pseudocode listings, one for each algorithm.

### Algorithm 1: Distillation-Augmented PPO (DA-PPO)

```python
# Algorithm 1: Distillation-Augmented PPO (DA-PPO)

# Hyperparameters and initialization
batch_size = 16            # number of prompts per update batch (example)
ppo_epochs = 4             # number of PPO epochs per batch
epsilon_clip = 0.2         # PPO clipping parameter
lr_policy = 1e-5           # learning rate for policy network
lr_value = 1e-5            # learning rate for value network (critic)
beta_KL_ref = 0.02         # KL penalty coefficient to reference model
beta_KL_teacher = 0.05     # KL penalty coefficient to teacher model (distillation strength)
gamma = 0.99               # discount factor for rewards (if needed for advantage)
lam = 0.95                 # GAE lambda for advantage estimation
update_ref_interval = None # if we update reference periodically (or keep fixed initial)
    
student = StudentModel(...)        # initialize student model (pre-trained weights)
teacher = TeacherModel(...)        # load teacher model (frozen)
reference = ReferenceModel(student) # copy initial student to reference (frozen for KL calc)
value_net = ValueNetwork(...)      # initialize value function network V(s)
optimizer_policy = Adam(student.parameters(), lr=lr_policy)
optimizer_value = Adam(value_net.parameters(), lr=lr_value)

for iteration in range(1, MAX_ITERATIONS+1):
    # 1. Sample a batch of prompts from the multi-domain dataset
    prompts = sample_prompts(batch_size)  # each prompt has a domain attribute implicitly
    
    # 2. Generate trajectories from student (on-policy)
    student_outputs = []
    logprobs = []   # to store log \pi_\theta(a|s) for each token
    values = []     # to store V(s) for each state if using advantage estimation
    with torch.no_grad():
        for prompt in prompts:
            out, lp_seq = student.generate(prompt, return_logprobs=True)
            # out is the generated sequence (list of tokens)
            # lp_seq is the list of log probabilities for each token in out
            student_outputs.append(out)
            logprobs.append(lp_seq)
            # get value predictions for each prefix state if needed for GAE
            V_seq = []
            state = prompt.clone()  # initial state/context
            for t, token in enumerate(out):
                V_state = value_net(state)       # V(s_t)
                V_seq.append(V_state)
                state.append(token)             # update state with generated token
            # After generating all tokens, append value for terminal state (if using bootstrap, but typically 0 if episode ends)
            V_seq.append(torch.tensor(0.0))  # value of terminal (no future reward)
            values.append(V_seq)
    
    # 3. Compute rewards for each trajectory using RewardModel (and possibly KD rewards integrated)
    rewards = []
    for prompt, out in zip(prompts, student_outputs):
        task_reward = RewardModel(prompt, out)          # primary reward from environment/human model
        # Optionally, add teacher distillation reward as part of total reward
        # Here we incorporate teacher guidance via KL penalty instead of as part of reward,
        # so we keep rewards as just task_reward for advantage calc.
        rewards.append(task_reward)
    
    # 4. Compute advantages (using GAE if multiple reward steps, else simple)
    advantages = []
    returns = []
    for i in range(batch_size):
        r_seq = rewards[i]            # if reward is single value for full trajectory
        # If reward model provides a single score, treat it as final reward for all time-steps.
        # Distribute it over the sequence for advantage calculation: e.g., at final step.
        T = len(student_outputs[i])
        # Compute GAE advantage for each time step t
        A_seq = [0] * (T+1)
        adv = 0.0
        for t in reversed(range(T)):
            delta = 0.0
            if t == T-1:
                # final token gets the full reward
                delta = r_seq + 0.0 - values[i][t].item()  # next state value = 0
            else:
                delta = 0.0 + gamma * values[i][t+1].item() - values[i][t].item()
            adv = delta + gamma * lam * adv
            A_seq[t] = adv
        # It's also possible to simply set advantage = reward - V(s_0) for episodic setting.
        advantages.append(A_seq)
        # Compute returns for value net targets (e.g., final reward for all steps or GAE accumulated)
        # Here, we'll use the final reward as return for all steps for simplicity.
        R = r_seq  # total episode return
        returns.append([R] * len(values[i]))  # same length as V_seq (including terminal)
    
    # Convert advantages and returns to tensors for optimization
    # (Flatten across sequences if treating each token as a sample for optimization)
    flat_logprobs = []  # flatten logprobs similarly
    flat_old_logprobs = []  # store old logprobs for ratio calc
    flat_advantages = []
    flat_returns = []
    for i in range(batch_size):
        flat_logprobs.extend(logprobs[i])
        flat_advantages.extend(advantages[i][:-1])  # last element is advantage at terminal which is 0
        flat_returns.extend(returns[i][:-1])
    flat_logprobs = torch.tensor(flat_logprobs)  # log π_θ_old (from sampling) for each token
    flat_advantages = torch.tensor(flat_advantages)
    flat_returns = torch.tensor(flat_returns)
    
    # Store old policy probabilities for ratio (we have logprobs already)
    flat_old_logprobs = flat_logprobs.clone().detach()
    
    # 5. PPO update with clipping and knowledge distillation regularization
    for epoch in range(ppo_epochs):
        # Compute current log probabilities for all those token actions under current policy
        current_logprobs = student.log_probs_of(prompts, student_outputs)  # compute log π_θ(current) for each token in the stored outputs
        # Compute ratio
        ratio = torch.exp(current_logprobs - flat_old_logprobs)
        # PPO surrogate loss
        surrogate1 = ratio * flat_advantages
        surrogate2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * flat_advantages
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        
        # Compute value loss (critic MSE)
        V_pred = value_net.predict(prompts, student_outputs)  # predict value for each state (same dimension as flat_returns)
        value_loss = torch.mean((V_pred - flat_returns)**2)
        
        # Compute KL penalties:
        # KL with reference (initial policy) - measure on the whole output distribution (or average token KL)
        ref_logprobs = reference.log_probs_of(prompts, student_outputs)
        kl_ref = torch.mean(torch.exp(current_logprobs) * (current_logprobs - ref_logprobs))
        # KL with teacher
        teacher_logprobs = teacher.log_probs_of(prompts, student_outputs)
        kl_teacher = torch.mean(torch.exp(current_logprobs) * (current_logprobs - teacher_logprobs))
        # (Note: above is an approximation of KL(π_θ || π_ref) by averaging per-token KL)
        
        # Total loss
        total_loss = policy_loss + c1 * value_loss + beta_KL_ref * kl_ref + beta_KL_teacher * kl_teacher
        # c1 is a coefficient for value loss (like 0.5 usually)
        
        # Gradient descent on total loss
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        total_loss.backward()
        optimizer_policy.step()
        optimizer_value.step()
    
    # 6. (Optional) update reference model periodically to current student to prevent policy drift issues
    if update_ref_interval and iteration % update_ref_interval == 0:
        reference.load_state_dict(student.state_dict())  # update reference to current policy (and potentially adjust beta_KL_ref)
```

*Notes:* In this pseudocode, for brevity, we assumed that reward is given as a single scalar per trajectory. In a real implementation, one might integrate token-level rewards or handle varying sequence lengths more carefully. The key parts are: computing `kl_teacher` using the teacher’s log probabilities on the same actions, and including that in the loss. We set up `beta_KL_teacher` as a hyperparam to tune how strongly to enforce similarity to the teacher. Also note that we used a reference model to compute `kl_ref` for safety (as is common in RLHF PPO); one might use the teacher as the reference directly, but here we keep both – initial ref to keep style and teacher KL to incorporate knowledge. This pseudocode emphasizes clarity of the DA-PPO update; a production implementation would vectorize operations and possibly compute KL in a single forward pass to avoid multiple model evaluations. However, since teacher and reference are static, their logprobs can be precomputed along with initial sampling, saving some compute.

### Algorithm 2: Distillation-Augmented GRPO (DA-GRPO)

```python
# Algorithm 2: Distillation-Augmented GRPO (DA-GRPO)

# Hyperparameters (some similar to PPO but no value net needed)
batch_size = 16
group_size = 8                 # number of output samples per prompt (including teacher if used)
epsilon_clip = 0.2
lr_policy = 1e-5
beta_ref = 0.02               # KL penalty coeff to reference model
beta_teacher = 0.05           # KL penalty coeff to teacher
# Note: Often GRPO can use a larger learning rate since advantage normalization helps.
    
student = StudentModel(..., policy_type="stochastic")  # ensure we can sample from it
teacher = TeacherModel(...)
reference = ReferenceModel(student)  # freeze initial weights as reference
optimizer = Adam(student.parameters(), lr=lr_policy)

for iteration in range(1, MAX_ITERATIONS+1):
    prompts = sample_prompts(batch_size)
    
    # For each prompt, sample group of outputs from student (old policy) and get one teacher output
    all_outputs = []    # will be list of list of outputs per prompt
    all_logprobs = []   # list of list of logprob sequences per prompt
    all_rewards = []    # list of reward list per prompt
    for prompt in prompts:
        # generate G-1 samples from student
        student_samples = []
        student_logprob_samples = []
        for i in range(group_size - 1):
            out, lp_seq = student.generate(prompt, return_logprobs=True)
            student_samples.append(out)
            student_logprob_samples.append(lp_seq)
        # get teacher output
        with torch.no_grad():
            teacher_out, _ = teacher.generate(prompt, return_logprobs=True)
        # Evaluate reward model on all outputs (student and teacher)
        rewards = []
        for out in student_samples:
            r = RewardModel(prompt, out)
            rewards.append(r)
        r_teacher = RewardModel(prompt, teacher_out)
        rewards.append(r_teacher)
        # Now we have group_size outputs (last one is teacher's)
        all_outputs.append(student_samples + [teacher_out])
        all_logprobs.append(student_logprob_samples + [None])  # teacher logprob None, we won't use it for gradient
        all_rewards.append(rewards)
    
    # Compute baseline: mean and std of rewards for each group
    all_advantages = []
    for rewards in all_rewards:
        r_mean = np.mean(rewards)
        r_std  = np.std(rewards) + 1e-8
        advantages = [ (r - r_mean)/r_std for r in rewards ]
        all_advantages.append(advantages)
    
    # Now perform policy gradient update using advantages.
    # We iterate through each prompt and each student output (teacher output is excluded from gradient calculation).
    policy_loss = 0.0
    kl_ref_loss = 0.0
    kl_teacher_loss = 0.0
    count = 0
    for prompt_idx, prompt in enumerate(prompts):
        outputs = all_outputs[prompt_idx]
        logprob_seqs = all_logprobs[prompt_idx]
        advantages = all_advantages[prompt_idx]
        # There are `group_size-1` student outputs and 1 teacher output at index group_size-1.
        G = len(outputs)
        for j in range(G-1):  # iterate over student outputs only
            out = outputs[j]
            adv = advantages[j]        # advantage for this output
            lp_seq = logprob_seqs[j]   # logprobs for each token
            # If using token-level advantage, one could distribute adv to each token.
            # Here for simplicity, we assign the final sequence advantage to each token.
            # Alternatively, multiply adv by 1/|out| per token contribution.
            L = len(out)
            for t in range(L):
                # policy loss contribution: -adv * log π(a_t)
                policy_loss += - adv * lp_seq[t] 
                count += 1
            # Compute KL terms: using per-token penalty as well
            # For each token in student's output, penalize deviation from reference and teacher.
            ref_logprobs = reference.log_probs(prompt, out)  # log probs for each token under reference
            teacher_logprobs = teacher.log_probs(prompt, out)  # log probs for each token under teacher
            for t in range(L):
                # KL divergence terms: treat adv as weight or not? In GRPO, we included KL in reward so it affects advantage, but here we explicitly add.
                # We add as separate loss terms for simplicity.
                # We accumulate KL of this trajectory: 
                kl_ref_loss += torch.exp(lp_seq[t]) * (lp_seq[t] - ref_logprobs[t])  # p*log(p/q) approx
                kl_teacher_loss += torch.exp(lp_seq[t]) * (lp_seq[t] - teacher_logprobs[t])
    # Average losses
    policy_loss = policy_loss / count
```python
        # ... continuing Algorithm 2 pseudocode from above ...
        kl_ref_loss = kl_ref_loss / count
        kl_teacher_loss = kl_teacher_loss / count
        policy_loss = policy_loss / count
        total_loss = policy_loss + beta_ref * kl_ref_loss + beta_teacher * kl_teacher_loss

    # Gradient descent step for policy
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

*Notes:* In this DA-GRPO pseudocode, we integrated the teacher in two ways: by including its output in the reward baseline (through `advantages`) and by adding an explicit `kl_teacher_loss`. The `policy_loss` is computed using the normalized advantages as weights for the log-prob terms (negative because we maximize reward). We assumed a single update per batch for simplicity; in practice, one might iterate a few epochs or use smaller learning rates. Also, `reference.log_probs` and `teacher.log_probs` are used to calculate KL terms token-wise (averaged to approximate full distribution KL). This implementation treats each token in each student output as a training example (weighted by the sequence advantage); an alternative is to treat each sequence as one example and derive the gradient of $J_{GRPO}$ as in the formula ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=%5C%5BJ_%7BGRPO%7D%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BG%7D%5Csum_%7Bi%3D1%7D,ref%7D%5D%5Cright)). Both yield similar outcomes. The teacher’s output is not used in gradient (`logprob_seqs` for teacher is `None` and we skip it), but it influences `advantages`. This ensures the teacher guides the baseline without directly being imitated unless the reward model and KL term dictate so.

### Algorithm 3: Hybrid Reinforcement Learning and Distillation (HRLD)

```python
# Algorithm 3: Hybrid Reinforcement Learning and Distillation (HRLD)

# Hyperparameters
batch_size = 16
ppo_epochs = 3
epsilon_clip = 0.2
lr_rl = 5e-6                 # learning rate during RL phase
lr_sup = 1e-5                # learning rate during supervised phase
alpha_sup = 1.0              # weight for supervised loss (could anneal over time)
alpha_rl = 1.0               # weight for RL loss (could increase over time)
distill_fraction = 0.5       # fraction of steps (or probability) to do supervised vs RL initially
distill_anneal = 0.95        # multiply distill_fraction by this every epoch to anneal down

student = StudentModel(initialization="pretrained_SFT")
teacher = TeacherModel()         # teacher model (assumed frozen)
reward_model = RewardModel()     # reward function or model
optimizer = Adam(student.parameters(), lr=lr_sup)  # will adjust lr per phase

for epoch in range(1, NUM_EPOCHS+1):
    # Possibly adjust the mix of supervised vs RL
    current_distill_prob = min(1.0, distill_fraction * (distill_anneal ** (epoch-1)))
    # We will perform one combined epoch consisting of some supervised steps and some RL steps
    
    # --- Phase A: Supervised Distillation Phase ---
    student.train()   # enable training mode
    num_sup_steps = int(current_distill_prob * SUP_STEPS_PER_EPOCH)
    for step in range(num_sup_steps):
        # Sample a prompt and get teacher output for it
        prompt = sample_prompt()
        with torch.no_grad():
            teacher_output = teacher.generate(prompt)
        # Calculate supervised distillation loss (cross-entropy)
        student_log_probs = student.log_probs(prompt, teacher_output)
        # We maximize log likelihood of teacher_output, so loss = - sum log P(student|teacher_output)
        sup_loss = - torch.sum(student_log_probs)
        optimizer.zero_grad()
        sup_loss.backward()
        optimizer.step()
    # Optionally, one can shuffle through a pre-generated dataset of teacher demonstrations instead of sampling live.
    
    # --- Phase B: Reinforcement Learning Phase ---
    # Switch optimizer to RL learning rate if needed
    for g in optimizer.param_groups:
        g['lr'] = lr_rl
    # Collect on-policy data
    trajectories = []
    for _ in range(batch_size):
        prompt = sample_prompt()
        output, logprob_seq = student.generate(prompt, return_logprobs=True)
        reward = reward_model(prompt, output)
        trajectories.append((prompt, output, logprob_seq, reward))
    # Compute advantages (simple baseline = 0 or using average reward as baseline)
    avg_reward = np.mean([traj[3] for traj in trajectories])
    advantages = [traj[3] - avg_reward for traj in trajectories]
    # PPO update on this batch
    # (We do a single update epoch for brevity; multiple ppo_epochs can be done similarly to DA-PPO above)
    logprob_seqs_old = [traj[2] for traj in trajectories]  # old logprobs
    for _ in range(ppo_epochs):
        # Compute current logprobs for each stored trajectory under current policy
        logprob_seqs_new = [student.log_probs(traj[0], traj[1]) for traj in trajectories]
        policy_losses = []
        for j, traj in enumerate(trajectories):
            old_lp = logprob_seqs_old[j]
            new_lp = logprob_seqs_new[j]
            adv = advantages[j]
            # Compute PPO ratio and loss for each token in trajectory j
            ratio = torch.exp(new_lp - old_lp)
            # Here we take the minimum over the whole trajectory (or apply per token)
            # Simpler: enforce clip per token individually:
            clipped_ratio = torch.clamp(ratio, 1-epsilon_clip, 1+epsilon_clip)
            # Use the trajectory's advantage for all tokens (or divide among tokens)
            token_loss = - torch.min(ratio * adv, clipped_ratio * adv)
            policy_losses.append(torch.mean(token_loss))
        rl_loss = torch.mean(torch.stack(policy_losses))
        optimizer.zero_grad()
        rl_loss.backward()
        optimizer.step()
    # Reset optimizer lr to supervised rate for next epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_sup

    # End of epoch: Optionally evaluate on some validation prompts, adjust hyperparams, etc.
```

*Notes:* The pseudocode for HRLD alternates within each epoch between a supervised distillation phase and an RL phase. We maintain one optimizer for simplicity, adjusting its learning rate depending on phase (alternatively one could use two optimizers or two separate training loops). We use `distill_fraction` to control how much of the training effort goes to supervised vs RL; this fraction is annealed down each epoch to slowly shift emphasis to RL. In the supervised phase, we either sample prompts and get the teacher’s output on the fly or use a precomputed dataset of teacher outputs. We then compute a negative log-likelihood loss for the student on those outputs (this effectively performs behavior cloning of the teacher). In the RL phase, we sample prompts and have the student generate outputs, then get rewards and update the policy via PPO (we used a simplified PPO with one update; in practice this can be more extensive). We chose to use a very simple baseline (average reward) for advantage; one could use a running baseline or a small value network if needed. The code shows a straightforward per-token PPO loss calculation. Note that we reused `optimizer` for both phases, which implies momentum terms, etc., are shared – this is a design choice to simplify implementation, but one might decouple them for finer control (e.g. separate Adam for supervised and PPO to avoid interference of gradients). Also, in a real multi-domain setting, `sample_prompt()` would ensure a mixture of domains over time. Since the teacher model presumably performs well on all domains, the supervised phase inherently balances domain knowledge transfer according to the distribution of sampled prompts. The RL phase will then fine-tune on those same prompts with actual reward feedback.

One can enhance HRLD in various ways: for example, after the supervised phase, initialize the PPO reference policy to the student’s state (so that PPO’s KL is computed to the post-distillation policy, avoiding a sudden jump). Also, as mentioned, using an MoE student would involve routing each prompt to an expert; the pseudocode would then update only the parameters of the chosen expert (and perhaps a shared router) in each step. That could be handled by having `student.generate` internally pick the appropriate expert based on the prompt domain. The core loop remains similar.

## 5. Evaluation Methodology

A rigorous evaluation is crucial to validate the proposed algorithms. We outline an evaluation methodology covering **seven diverse domains** – creative writing, factual question answering, logical reasoning, code generation, dialogue, instruction following, and domain-specific adaptation – with appropriate metrics, datasets, and evaluation protocols for each. We also detail **ablation studies** to understand the contribution of each component, and **statistical testing** to ensure significance of results.

### 5.1 Evaluation Domains and Benchmarks

We identify seven domains and propose benchmarks for each:

1. **Creative Writing:** Evaluate the model’s ability to produce engaging, coherent, and original creative text (stories, poems, etc.). *Datasets/Tasks:* WritingPrompts (a dataset of creative story prompts with human-written stories), or a subset of the BigBench creative tasks (like “fantasy story generation”). *Metrics:* Because creativity is hard to quantify with a single metric, we will use **human evaluation** for qualities like originality, narrative coherence, and style. Additionally, automated metrics like **MAUVE** (for diversity) and perplexity (to ensure fluency) can be used. The model’s outputs for a set of creative prompts will be scored by human judges on a Likert scale for overall quality and by GPT-4 or another strong LLM as an evaluator for consistency. Domain-specific teachers (if used) might be a model fine-tuned on fiction, but evaluation will rely on human judgment primarily.

2. **Factual Question Answering (QA):** Measure the model’s accuracy in answering closed-domain and open-domain questions correctly. *Datasets:* Natural Questions (NQ) ([](https://arxiv.org/pdf/2402.03300#:~:text=reducing%20training%20resources,We)), TriviaQA, or WebQuestions could be used. Also, the TruthfulQA benchmark can assess avoiding misinformation. *Metrics:* **Exact match** and **F1 score** against ground-truth answers (for datasets that have ground truth). For open-ended QA without single correct answers, use **Precision@1** if the answer can be checked in a knowledge source, or have human raters judge correctness. We will also track the rate of factual errors and **hallucinations** (fabricated facts). The model should show improvement in factual accuracy if knowledge distillation succeeded (especially if teacher had high accuracy).

3. **Logical Reasoning:** Test the model on tasks requiring multi-step reasoning, mathematical problem solving, or logical deduction. *Datasets:* GSM8K (Grade School Math problems) for arithmetic reasoning ([](https://arxiv.org/pdf/2402.03300#:~:text=obtains%20a%20substantial%20improvement%20over,We)), the MATH dataset for higher math, and logical puzzles from Big-Bench Hard (BBH) or ProofWiki for theorem-like reasoning. *Metrics:* **Accuracy** on getting the correct final answer for math problems, or stepwise accuracy for multi-step solutions (did the chain-of-thought lead to the right conclusion). We will also look at **reasoning consistency**: e.g., whether the model’s explanation logically follows. If the model provides a chain-of-thought, we can use metrics like **Consistency** or have automated proof checkers verify steps for math. An example: for GSM8K, the percentage of correctly solved problems is a key metric. Human evaluation might be needed for logical puzzles without a single numeric answer (judging if reasoning is valid). The expectation is that GRPO-based training (which was motivated by math reasoning) and teacher guidance will yield high gains here, so we will closely compare baseline PPO vs DA-PPO vs DA-GRPO on these tasks.

4. **Code Generation:** Assess the model’s ability to generate correct and efficient code given a problem description. *Datasets:* HumanEval (OpenAI’s Python coding challenges) is a primary benchmark, measuring the functional correctness of generated code ([Breakthroughs in Knowledge Distillation - LinkedIn](https://www.linkedin.com/pulse/breakthroughs-knowledge-distillation-advancing-large-ramachandran-1xkqc#:~:text=Breakthroughs%20in%20Knowledge%20Distillation%20,students%20toward%20optimal%20learning%20outcomes)). Additionally, MBPP (Mostly Basic Python Problems) and Codeforces programming problems can be used for variety. *Metrics:* **Pass@k** (especially pass@1 and pass@5) is standard for code generation – the fraction of problems solved with at least one correct solution in the top k attempts. We will run the generated code against unit tests or evaluation scripts to determine correctness. Other metrics: **Code BLEU** or edit distance to reference solutions for style (though correctness is paramount). We also measure if the model refrains from producing syntactic errors. Domain-specific adaptation here might involve using a specialized code teacher (like Codex or StarCoder) and checking if student retains coding knowledge. The expectation is that knowledge-distillation will significantly improve pass@1, as the teacher’s expertise guides the student to produce syntactically correct and logically sound solutions.

5. **Dialogue (Open-Ended Conversational Ability):** Evaluate the model in multi-turn dialogue for coherence, context awareness, and adherence to conversational norms (politeness, not going off-topic). *Datasets:* We can use the Multi-turn Dialogue from the BlenderBot evaluation sets or ConvAI2, as well as user queries from OpenAI’s human feedback datasets (if available) for conversational quality. Also, the HH (Harmlessness and Helpfulness) test sets used in Anthropic’s work to see how aligned the model is in dialogue. *Metrics:* **Engagement** and **Consistency** as rated by humans: does the model stay on topic, give helpful responses, and remember context from earlier turns? Also check **Harmlessness**: ensure no toxic or inappropriate content (this is more of a safety metric – important in alignment). Automated metrics could include **Next utterance selection accuracy** on some dialog datasets or embedding-based coherence scores. Likely, we’ll rely on human eval or LLM eval (GPT-4 scoring conversations on helpfulness). We might have crowd workers chat with each model (baseline vs ours) and do pairwise preference testing. Dialogue is a domain where instruction-following and preference alignment is crucial; we expect our algorithms (especially HRLD, which fine-tunes with human feedback reward) to produce more user-aligned dialogues than a baseline.

6. **Instruction Following:** This tests how well the model follows arbitrary instructions from users. *Datasets:* Super-NaturalInstructions (a benchmark of numerous diverse instructions with reference outputs), or the FLAN collection of tasks, can be used to generate evaluation prompts. Also, we can use the HELM benchmark’s instruction following scenarios. *Metrics:* Since instructions can have correct outputs (for tasks like translation or summarization), we use task-specific metrics: e.g., BLEU/ROUGE for summarization or translation tasks, accuracy for classification tasks given in instruction form, etc. For open instructions (like “Describe X in style Y”), we use human evaluation to judge if the instruction was properly adhered to (did the model do what was asked, and only that). **Compliance Rate** is an important metric: the percentage of instructions for which the model’s response satisfies the user’s request. We will test tricky cases too, like instructions that could lead to refusal (to check if the model appropriately refuses when it should, showing alignment). The model’s performance should be near the teacher’s on known tasks (thanks to distillation) and possibly improved on following user intent on novel instructions (due to RL fine-tuning).

7. **Domain-Specific Adaptation:** This evaluates the model’s ability to adapt or perform in a specialized domain (e.g. medical, legal, scientific) that might not be well-represented in the base data. We consider two aspects:
   - **Zero-shot performance in a new domain:** e.g. test the model on medical Q&A (like PubMedQA or MedMCQA) without further fine-tuning, to see if multi-domain training preserved broad knowledge. *Datasets:* For medical, use PubMedQA (factoid questions with answers) or MedMCQA (multiple-choice). For legal, maybe the CaseLaw summaries or LEDGAR dataset. *Metrics:* Accuracy or F1 on these domain-specific tasks, compared to baseline models or teacher if teacher specialized. If our model was properly distilled from a teacher that had that domain knowledge, it should outperform a baseline of similar size that wasn’t.
   - **Fine-tuning efficiency on a new domain:** take the trained student model and fine-tune it on a small amount of data from a new domain, measuring how quickly it reaches a given performance. For instance, fine-tune on 100 examples of a medical dialogue and test on another 100. *Metrics:* The performance (accuracy or BLEU, etc.) after limited fine-tuning, and compare it to how a baseline model (not multi-domain trained) would do with the same fine-tuning. Also measure **catastrophic forgetting**: after adapting to the new domain, evaluate the model on the original domains to see if performance drops significantly or remains robust. Our expectation is that a model trained with multi-domain and teacher signals (especially with HRLD which may encourage a generalizable policy) will be more adaptable (i.e., require fewer examples to learn new domain) and more robust (less forgetting due to perhaps a more diverse training).

Across all domains, we will use a strong **baseline** for comparison: likely the base PPO fine-tuned model (as in standard RLHF) and possibly the teacher itself (if the teacher is an open model like LLaMA-65B and student is LLaMA-7B, we compare student’s performance to teacher’s). We will also compare against recently proposed techniques (e.g. DPO ([](https://arxiv.org/pdf/2402.03300#:~:text=a%20unified%20paradigm%20to%20understand,turn%20v.s)) or RLAIF) if applicable to show our methods’ advantages.

### 5.2 Evaluation Metrics and Criteria

We summarized some metrics per domain above. Here we list general metrics categories and criteria:

- **Task Success Metrics:** These are domain-specific: e.g. accuracy (for QA, reasoning with known answers), pass@k (code), BLEU/ROUGE (for tasks with references like translation or summarization), etc. These measure whether the content of the output meets the objective criteria of the task. We will gather these for each domain’s evaluation set. Higher is better.

- **Quality and Preference Metrics:** These often require human judgment or proxy models:
  - **Human Preference Score:** In tasks like creative writing or dialogue, we will have human evaluators rank outputs from different models or rate a single output. We can perform pairwise comparisons: e.g., show an evaluator the outputs of baseline vs DA-PPO model for the same prompt, ask which is better (blind review). Repeating this yields a **win rate** for our model over baseline.
  - **LLM-based Evaluation:** Use GPT-4 or another advanced model to assign a score to outputs for qualities like coherence, style, helpfulness. This provides an automated yet often correlated metric with human judgment.
  - **Harmlessness & Helpfulness (Alignment) Metrics:** Using something like Anthropic’s HH eval: count how often the model refuses when it should (e.g. if asked for disallowed content), how often it gives harmful content, etc. These are crucial for alignment. We will use curated prompts that test boundaries and have human annotators verify compliance. The metric could be e.g. **% of responses that are rated as not harmful and appropriately helpful**. We expect our RL fine-tuned models to outperform the teacher (which might not be aligned) on this.

- **Knowledge Retention:** We might devise a metric for how well the model retained knowledge from the teacher. For instance, evaluate the student on a set of factual questions that the teacher can answer correctly. If teacher got X% right and student got Y% right, a small gap means good retention. This is indirectly measured by the factual QA metrics and some domain-specific tests. We want to ensure Y is close to X (or even above, if RL improved it using reward model feedback).

- **Computational Efficiency Metrics:** While the main focus is performance, we will also track training efficiency: e.g. how many reward model calls or how many tokens of training were required to reach a certain reward level. Since multi-domain RLHF can be costly, any reduction is beneficial. If our methods converge faster, we might report the number of training steps to reach a threshold performance in each domain.

- **Generalization:** We plan to test on some prompts not seen during training (naturally, we have held-out eval sets). But also cross-domain generalization: e.g. give a prompt that combines domains (like a factual question requiring logical reasoning, or an instruction that involves writing code in a story). These quirky tests (some are in BigBench) can reveal if the model can unify skills. There’s no standard metric for “multi-domain synergy”, but we can qualitatively evaluate a few such cases or use existing challenge tasks from BIG-GT (Beyond Imitation Generalization Test, if any exist).

### 5.3 Ablation Studies

To understand the contribution of each component of our approaches, we will conduct several ablation experiments:

- **KD Reward Ablation:** Remove the knowledge distillation-based reward/penalty from the algorithms. For DA-PPO, set $\beta_{\text{KD}}=0$ (no teacher KL); for DA-GRPO, do not include teacher in baseline or KL. Compare performance on each domain to see how much the teacher guidance was helping. We expect drops in factual accuracy and perhaps reasoning ability without KD, demonstrating its value.

- **Group Baseline vs Critic:** In DA-GRPO, an interesting ablation is to compare against a variant that uses a value function instead of group baseline (i.e. run DA-PPO on the same setting) to see if the group advantage was beneficial. This can isolate the effect of relative advantage. Conversely, run GRPO without teacher in baseline vs with teacher in baseline to see the difference.

- **Hybrid Schedule Ablation:** For HRLD, try training with pure RL (no distillation phase at all) and pure distillation (no RL, which basically yields the teacher’s performance) as two extremes. Also try a simpler approach of just adding the losses (like one-phase training optimizing $\mathcal{L} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{KD}$) to see if our alternating schedule performs better. This will show whether the two-phase method offers stability or performance gains beyond a naive combination.

- **Teacher Model Size/Quality:** If possible, use different teacher models: e.g. a weaker teacher vs a stronger teacher to see impact. For instance, take our student (say 7B) and use a 13B teacher and a 70B teacher. We expect the 70B teacher to impart more knowledge and yield a better student. This ablation validates that the algorithms can leverage teacher improvements.

- **Domain-Specific vs Single Teacher:** If we have multiple teachers (specialized per domain), test training with just one general teacher vs domain teachers. For example, one run where $\pi_T$ is always a single model (maybe the best general model available), and another where we use an ensemble: e.g. a code teacher for code prompts, a math teacher for math prompts, etc. How much does specialization help? This will inform if future work should consider multiple teacher experts. We might measure each domain’s result in those two conditions.

- **Effect of Group Size (GRPO specific):** Run GRPO variations with different group_size $G$. We hypothesize $G$ can be smaller when a teacher is included (because teacher provides a strong baseline, maybe you don’t need as many student samples). So compare $G=8$ with teacher vs $G=8$ without vs $G=4$ with teacher, etc., checking performance and training stability (variance of returns, etc.). This can highlight computational savings possible by teacher augmentation.

- **Ablation of MoE usage:** If we implemented mixture-of-experts in the student (say our best model uses MoE), compare to a dense model of similar FLOPs. Evaluate if the MoE model after training indeed specialized experts (we can inspect which expert triggers on which domain prompts) and if that yields better performance. Also test the MoE model on single-domain prompts to see if unused experts degrade. This ablation clarifies the benefit of MoE architecture in multi-domain RLHF.

For each ablation, we will hold other factors constant and retrain the model or reuse partial training if possible. We will track the same metrics as main experiments to see where differences occur. For example, if removing teacher KL drastically reduces math reasoning accuracy but maybe doesn’t hurt dialogue as much, that gives insight that teacher contributed especially to math domain.

We will also perform an **ablation on reward functions**: try using only TDA vs only TAA vs both. Perhaps one of them has a stronger effect. E.g., in factual QA, TAA (teacher answer agreement) might suffice (since correct final answer is key), whereas in dialogue, TDA (policy matching) might yield more polite style adoption. By turning each on/off, we get a sense of their roles.

### 5.4 Statistical Significance Testing

Given the randomness in both model initialization and sampling-based training, it’s vital to assert that improvements are statistically significant and not due to chance. We outline our significance testing approach:

- **Multiple Training Runs:** We will train each method (baseline PPO, DA-PPO, DA-GRPO, HRLD, etc.) with at least 3 different random seeds (different initialization seeds or data shuffling seeds). This yields multiple independent models for each method. We will evaluate each model on the test sets, obtaining distributions of metrics. Using these, we can perform statistical tests (e.g. a t-test or Wilcoxon signed-rank test) on paired results (since the test sets are the same, we can pair by prompt when comparing models).

- **Significance on Win Rates:** For human preference comparisons, we will perform a **binomial test** or bootstrap confidence interval on the win percentage. For example, if our model’s output is preferred in 60 out of 100 comparisons with baseline, we compute the 95% confidence interval for that proportion and ensure it’s above 50% significantly. We can also use a **Chi-square test** if comparing multiple models at once in a preference study.

- **Permutation Tests/Bootstrap:** For automated metrics like accuracy or BLEU, we will use a bootstrap resampling of the test set to compute confidence intervals for each model’s score. If the intervals for two models do not overlap much, or more rigorously, we do a paired bootstrap test for difference in means. This is especially useful for metrics like exact match which are either 0/1 per example – the paired bootstrap will tell us if one model is better on a significant subset of examples.

- **ANOVA for Ablations:** When doing ablations or comparing more than two methods, we may perform an ANOVA to see if there is a significant effect of the method on the metric, followed by post-hoc tests (like Tukey’s HSD) to identify which pairs differ. However, given many metrics, we might limit to pairwise comparisons on each metric separately to keep things straightforward.

- **Statistical Significance Thresholds:** We will use $\alpha = 0.05$ as the significance level. For metrics aggregated over many examples (like accuracy over 1000 questions), differences of a few points often already have p-value < 0.01. For human eval which might have fewer data points, we’ll ensure at least ~50-100 samples for comparisons to have decent power, and use one-tailed tests if we have a clear hypothesis of direction (we expect our models to be better). All significance tests will be clearly reported, and we will avoid p-hacking by deciding on primary metrics ahead of time (for example, we might declare that the primary metric for each domain is: creative writing – human preference win%, QA – exact match, etc., and focus significance analysis on those).

- **Inter-Annotator Agreement:** For human evaluations, we will also measure consistency among annotators (using Krippendorff’s alpha or Cohen’s kappa) to ensure the evaluation is reliable. If agreement is low, we may increase the number of ratings or refine instructions to evaluators.

- **Cross-Domain Significance:** If we aggregate results across all domains (like an overall win rate or an average score), we have to be cautious as metrics are not directly comparable. Instead, we might perform significance testing domain-by-domain, then use methods to combine p-values (like Fisher’s method) to claim an overall significance of improvement across the board.

In conclusion, our evaluation methodology will thoroughly assess the models on each domain with both quantitative metrics and qualitative judgments. We will demonstrate that our proposed algorithms (DA-PPO, DA-GRPO, HRLD) achieve **state-of-the-art alignment** performance on this multi-domain suite, and that each novel component (knowledge distillation rewards, group baselines, hybrid training) contributes to these improvements. The results will be presented with robust statistical validation, ensuring they meet the bar for publication in top ML venues (NeurIPS/ICLR/ICML) in terms of empirical rigor, reproducibility (through clear pseudocode and descriptions), and the demonstrated **novelty** and effectiveness of our approach. 

Overall, the comprehensive evaluations will highlight that combining reinforcement learning with teacher model distillation can significantly enhance multi-domain LLM fine-tuning, yielding models that are both highly capable (on diverse tasks) and aligned with desired behaviors, all achieved with reasonable computational costs and demonstrated generalizability. ([Group Relative Policy Optimization (GRPO) Illustrated Breakdown | Ebrahim Pichka](https://epichka.com/blog/2025/grpo/#:~:text=1,the%20objective%20and%20the%20rewards)) ([](https://arxiv.org/pdf/2402.03300#:~:text=GRPO%20foregoes%20the%20critic%20model%2C,domain%20mathematical))

