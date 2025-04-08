Perfect. I’ll create a structured research report that explores integrating GRPO with reward functions inspired by MiniLLM and GEMMA 3. This will include pseudocode and PyTorch implementations of the proposed hybrid approach, along with algorithms, flowcharts, and comparison tables. I’ll also evaluate how this method can improve a Faroese-capable small model, identifying potential teacher and student candidates, as well as relevant Faroese benchmarks (including resources like EuroEval).

I’ll get started and update you once the report is ready for your review.

# Integrating GRPO with Knowledge Distillation – A Hybrid Approach

## 1. Group Relative Policy Optimization (GRPO) Overview

**GRPO** (Group Relative Policy Optimization) is a reinforcement learning algorithm introduced in *DeepSeek-R1* to train LLMs for better reasoning ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=Group%20Relative%20Policy%20Optimization%20,and%20long%20chains%20of%20thought)). It builds on **PPO** (Proximal Policy Optimization) but **eliminates the need for a value (critic) model** by using *group-based baselines* ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=to%20improve%20the%20reasoning%20capabilities,to%20improve%20models%20on%20helpfulness)) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,part%20of%20the%20reward%20signal)). In PPO (as used in RLHF for ChatGPT-style training), a separate value network estimates the reward-to-go for each output, which doubles memory use and complicates training ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=Dependency%20on%20a%20Critic%20Model%3A)). GRPO avoids this by **comparing multiple responses to the same prompt and computing advantages *relative to the group***, rather than against an absolute value estimate ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=modifies%20the%20traditional%20Proximal%20Policy,to%20improve%20models%20on%20helpfulness)) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=3,KL%20term%20within%20the%20reward)). This yields a simpler, more memory-efficient RL pipeline.

**Key Idea:** For each prompt (query), the policy (LLM) generates a *group* of $G$ candidate outputs. Each output $o_i$ is scored by a reward function, yielding reward $r_i$. Instead of a learned critic baseline, GRPO uses the **group’s average reward** (or a similar statistic) as a baseline. The *advantage* for each output is $A_i = r_i - \bar{r}_{\text{group}}$ (often further normalized by the group’s standard deviation) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=1,5)) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=Once%20we%20have%20our%20set,deviation%20of%20all%20the%20rewards)). Intuitively, an output is advantaged if it scored above the group’s mean, and disadvantaged if below. This advantage $A_i$ then guides policy updates: outputs better than the group baseline are reinforced, worse ones are suppressed ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=2,KL%20term%20within%20the%20reward)).

 ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)) *Figure: PPO vs GRPO architecture. GRPO forgoes the value network and derives advantages from relative group rewards, reducing memory and compute costs.*

**Training Paradigm:** In practice, GRPO training involves iterative updates similar to PPO but without a value loss. A reference policy (usually the initial model) is kept to compute a KL-divergence penalty ensuring the new policy doesn’t drift too far ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=a%20baseline,part%20of%20the%20reward%20signal)). The GRPO objective $\mathcal{L}_{\text{GRPO}}$ thus includes a policy loss term proportional to $-A_i \cdot \log \pi_\theta(o_i|q)$ (encouraging higher probability for high-advantage outputs and lower for low-advantage) and a KL regularization term that penalizes divergence from the reference model ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=a%20baseline,part%20of%20the%20reward%20signal)). This KL term in GRPO is applied *directly in the loss* (as a regularizer), whereas PPO often folded it into an augmented reward or used clipping ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,part%20of%20the%20reward%20signal)). Pseudocode for one iteration of GRPO:

```pseudo
for batch of prompts D_b in dataset D:
    # 1. Generate a group of G responses for each prompt with current policy
    outputs = [policy.generate(q, num_samples=G) for q in D_b]  # shape: (batch_size, G)
    # 2. Compute reward for each output using reward model or rule
    rewards = [[R(o) for o in outputs_i] for outputs_i in outputs]  # shape: (batch_size, G)
    # 3. Compute group baseline (e.g., mean reward) and advantages for each output
    for each prompt i in batch:
        μ = mean(rewards[i][:]) 
        σ = std(rewards[i][:])
        for each output j:
            A_ij = (rewards[i][j] - μ) / (σ + ε)   # normalized advantage
    # 4. Compute policy loss with advantages (negative, for gradient ascent) and KL penalty
    L_policy = 0
    for each prompt i, output j:
        L_policy += - A_ij * log π_θ(output_j | prompt_i)  
    L_kl = β * KL(π_θ(.|prompt) || π_ref(.|prompt))    # KL w.r.t reference policy
    loss = L_policy + L_kl
    # 5. Update policy model via gradient descent on loss
    optimizer.step(loss.grad())
```

This **relative advantage** approach addresses several issues of PPO:
- *No learned critic:* Reduces memory and training complexity by **dropping the value network entirely** ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=Dependency%20on%20a%20Critic%20Model%3A)) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=In%20PPO%20both%20the%20policy,GRPO%20drops%20the%20value%20model)). Memory usage is nearly halved, enabling RL fine-tuning on smaller GPUs ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=TLDR%20,the%20code%20and%20hardware%20requirements)).
- *Stabilized reward signal:* By normalizing rewards within each group (z-score normalization), the advantage is a bounded, zero-mean signal ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=1,5)) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=Once%20we%20have%20our%20set,deviation%20of%20all%20the%20rewards)). This prevents scale issues when rewards are uncalibrated or when combining multiple reward components.
- *Efficient credit assignment:* All candidate outputs for the same prompt share context, so the baseline reflects difficulty of that prompt. GRPO inherently performs a form of **self-comparison** per query, which is well-suited for tasks like reasoning where only relative correctness may be apparent ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=1,This%20is%20different)).

**Motivation:** DeepSeek found PPO struggled on reasoning tasks due to high variance and need for precise value estimates ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=Traditional%20RL%20methods%20like%20Proximal,to%20reasoning%20tasks%20in%20LLMs)). GRPO was designed to **“let the model compete against itself”** so that it learns from relative wins/losses in each batch rather than absolute scores ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=The%20first%20trick%20is%20that,starts%20by%20generating%20multiple%20outputs)) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=The%20model%20might%20come%20up,answer%3D305)). This proved highly effective: using GRPO with rule-based rewards (e.g. math accuracy, format adherence), DeepSeek-R1 dramatically improved its math word problem solving (pass@1 on AIME benchmark from 15.6% to 71.0%) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,between%20%E2%80%98%E2%80%99%20and%20%E2%80%98%E2%80%99%20tags)). The *trade-off* was that purely optimizing these rewards led to some artifacts (e.g. unnatural language or mixing of languages in outputs) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=Image%3A%20r1)). They mitigated this by an **alternating training schedule**: interleaving supervised fine-tuning (SFT) on human-written text with GRPO RL steps ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=This%20has%20the%20drawback%20of,alternating%20SFT%20%E2%86%92%20RL%20steps)). This multi-stage approach retained fluency while still leveraging RL for reasoning.

In summary, GRPO’s **architecture** replaces the critic with a group-based baseline, and its **training paradigm** involves generating multiple outputs per query, computing relative advantages, and updating the policy with a KL-regularized objective ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=1,This%20is%20different)) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,part%20of%20the%20reward%20signal)). The motivation is a simpler, scalable RL method that “cuts in half” the compute of RLHF pipelines ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=TLDR%20,the%20code%20and%20hardware%20requirements)) and is *“effective and easy to train”* for reasoning LLMs ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=DeepSeek,v3%29%20to%20a%20reasoning)).

## 2. Reward Functions in *MiniLLM*: Reverse KL and Pre-training Loss

[*MiniLLM: Knowledge Distillation of Large Language Models*](https://arxiv.org/abs/2306.08543) proposes a *sequence-level knowledge distillation* method that uses a **reverse KL divergence** objective and an auxiliary pre-training loss ($L_{PT}$) to train a student model ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)) ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)). These serve as *“reward functions”* in a generalized sense: the reverse-KL defines a reward signal for matching the teacher, and $L_{PT}$ preserves prior capabilities.

- **Reverse KL-Based Loss:** MiniLLM opts to minimize $\text{KL}(q_\theta \parallel p)$ where $p(y|x)$ is the teacher distribution and $q_\theta(y|x)$ the student ([](https://arxiv.org/pdf/2306.08543#:~:text=To%20alleviate%20this%20problem%2C%20we,the%20correctness%20of%20the%20generated)) ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)). In contrast, standard distillation often minimizes forward KL $\text{KL}(p \parallel q_\theta)$ (i.e. cross-entropy on teacher-generated text). The choice of reverse KL is crucial: 
  - Minimizing forward KL forces the student to cover all modes of the teacher, even low-probability “long-tail” outputs. A low-capacity student would then assign unjustifiably high probability to sequences that the teacher itself considers extremely unlikely ([](https://arxiv.org/pdf/2306.08543#:~:text=a%20Gaussian%20mixture%20distribution%20with,model%E2%80%99s%20dis%02tribution%20and%20focuses%20on)). This can lead to *mode covering* behavior and odd outputs that mimic rare teacher quirks ([](https://arxiv.org/pdf/2306.08543#:~:text=a%20Gaussian%20mixture%20distribution%20with,model%E2%80%99s%20dis%02tribution%20and%20focuses%20on)).
  - **Reverse KL is mode-seeking ([](https://arxiv.org/pdf/2306.08543#:~:text=To%20alleviate%20this%20problem%2C%20we,the%20correctness%20of%20the%20generated)):** It makes $q_\theta$ focus on the teacher’s **major modes** while ignoring low-probability regions of $p$. In language generation terms, the student will emphasize the *most likely correct or sensible outputs* the teacher would produce and not try to imitate every minor variation ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)). *“The student model avoids learning too many long-tail variants in the teacher’s distribution and focuses on the correctness of the generated contents”* ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)). This bias toward high-probability teacher outputs is desirable for distillation, as it yields a more *conservative and factual student*, reducing the chance of producing improbable or hallucinated text ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)). It’s analogous to preferring the teacher’s single best answer over a mixture of all answers.

  To optimize the reverse KL, MiniLLM uses a **policy gradient approach** ([](https://arxiv.org/pdf/2306.08543#:~:text=truthfulness%20and%20reliability%20,works%20well%20for%20compressing%20large)). We can view $- \log p_{\text{teacher}}(y|x)$ as a per-sequence *loss*, or equivalently $\log p_{\text{teacher}}(y|x)$ as a *reward* for the student’s sample $y$. The gradient of KL$(q||p)$ can be derived as an expectation under the student’s policy ([](https://arxiv.org/pdf/2306.08543#:~:text=truthfulness%20and%20reliability%20,works%20well%20for%20compressing%20large)):
  $$ \nabla_\theta \, \text{KL}(q_\theta \parallel p) \;=\; \mathbb{E}_{y \sim q_\theta}[ \nabla_\theta \log q_\theta(y|x) \,(\log q_\theta(y|x) - \log p(y|x)) ] $$
  Minimizing this is achieved by **reinforcement learning**: treat $R(y) = \log p_{\text{teacher}}(y|x)$ as the reward for student output $y$. Then the policy gradient is:
  $$ \nabla_\theta L_{\text{KD}} = -\mathbb{E}_{y \sim q_\theta}\big[ (\log p_{\text{teacher}}(y|x)) \, \nabla_\theta \log q_\theta(y|x) \big] \,. $$
  This updates the student to assign higher probability to outputs that the teacher assigns high probability (i.e. mimic the teacher’s likely outputs). MiniLLM implements this with several tricks to improve stability ([](https://arxiv.org/pdf/2306.08543#:~:text=truthfulness%20and%20reliability%20,works%20well%20for%20compressing%20large)):
  - *Single-step decomposition:* Rather than backpropagating through an entire generated sequence trajectory, they decompose the reward to each token, reducing variance (similar to treating each token prediction as an action and giving a terminal reward).
  - *Teacher-mixed sampling:* They sometimes sample from the teacher distribution as well to guide the student, mitigating “reward hacking” (the student exploiting the RL objective in unintended ways) ([](https://arxiv.org/pdf/2306.08543#:~:text=truthfulness%20and%20reliability%20,works%20well%20for%20compressing%20large)).
  - *Length normalization:* They normalize rewards w.rt. sequence length to avoid bias toward shorter or longer outputs (long outputs might accrue more reward simply by including many likely tokens).

  The outcome is a *policy optimization procedure for distillation*. Although framed as KD, it’s essentially RL with the **teacher as a reward provider**. This reverse KL approach consistently outperformed forward-KL (standard) distillation in their experiments, yielding *lower exposure bias, better calibration, and more factual long responses* ([](https://arxiv.org/pdf/2306.08543#:~:text=covers%20a%20large%20range%20of,with%20neglectable%20loss%20of%20diversity)).

- **Language Modeling Loss $L_{PT}$:** In knowledge distillation (especially with small students), there’s a risk the student overfits to the teacher and **forgets** its general pre-training abilities. To counter this, MiniLLM adds a **pre-training language modeling loss** $L_{PT}$ as a regularizer ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)). This loss is simply the usual negative log-likelihood on a corpus of generic text:
  $$ L_{PT} = -\mathbb{E}_{d \sim D_{PT}} \log q_\theta(d) \,,$$ 
  where $D_{PT}$ is a large collection of unlabelled text (ideally similar to the data the student was originally pre-trained on) ([llama version in Minillm · Issue #218 · microsoft/LMOps · GitHub](https://github.com/microsoft/LMOps/issues/218#:~:text=As%20for%20my%20understanding%2C%20should,another%20confusing%20point%20here%20is)) ([llama version in Minillm · Issue #218 · microsoft/LMOps · GitHub](https://github.com/microsoft/LMOps/issues/218#:~:text=1,think%201%20epoch%20is%20not)). In implementation, one can sample a batch of text from $D_{PT}$ each step and compute the cross-entropy loss on the student’s predictions for that text ([llama version in Minillm · Issue #218 · microsoft/LMOps · GitHub](https://github.com/microsoft/LMOps/issues/218#:~:text=As%20for%20my%20understanding%2C%20should,another%20confusing%20point%20here%20is)). This loss **preserves the student’s base language proficiency** on “canonical NLP benchmarks” ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)) by preventing the distillation process from distorting the model’s general language model behavior. Ouyang et al. (OpenAI’s InstructGPT) similarly found that mixing in a small amount of original LM training loss during RLHF fine-tuning keeps the model’s language fluency and broad knowledge intact ([](https://arxiv.org/pdf/2306.08543#:~:text=match%20at%20L2025%20on%20the,NLP%20tasks%20while%20keeping%20the)).

  In MiniLLM, the total update at each step is a combination of the **distillation loss** (reverse KL-based) and this **$L_{PT}$ loss** ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)) ([](https://arxiv.org/pdf/2306.08543#:~:text=match%20at%20L2025%20on%20the,NLP%20tasks%20while%20keeping%20the)). They report that removing $L_{PT}$ ($\text{w/o } L_{PT}$) significantly hurts performance on general tasks – for instance, a 1.3B student’s win rate on certain benchmarks dropped from 70.2% to 65.7% without $L_{PT}$ ([[PDF] MiniLLM: Knowledge Distillation of Large Language Models - arXiv](https://arxiv.org/pdf/2306.08543#:~:text=arXiv%20arxiv,MINILLM)) ([](https://arxiv.org/pdf/2306.08543#:~:text=w%2Fo%20LPT%2065,training)). Thus $L_{PT}$ acts as a *reward for retaining prior knowledge*: it gives gradient signal to not stray too far from the original model’s distribution on plain text.

**Summary:** MiniLLM’s training objective is effectively:
$$ \mathcal{L}_{\text{MiniLLM}} = \underbrace{\text{KL}(q_\theta \parallel p_{\text{teacher}})}_{\text{distillation loss}} + \; \lambda \underbrace{L_{PT}}_{\text{pretrain reg.}} \,. $$
The first term is optimized via policy gradient (treating teacher log-probs as reward), the second via standard cross-entropy. In practice, MiniLLM’s Algorithm 1 draws batches of instruction prompts, has the teacher generate responses, and updates $\theta$ with combined gradients from the RL term and the $L_{PT}$ term ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)) ([](https://arxiv.org/pdf/2306.08543#:~:text=Long%20%2B%20%E2%88%87LPT%03%20until%20converge,1%20Experimental%20Setup)). The reverse KL “reward” drives the student toward the teacher’s high-probability outputs (improving *alignment and correctness* ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate))), while $L_{PT}$ prevents catastrophic forgetting (maintaining *language competency* ([](https://arxiv.org/pdf/2306.08543#:~:text=match%20at%20L2025%20on%20the,NLP%20tasks%20while%20keeping%20the))).

## 3. GEMMA 3’s Distillation and Reward Approach

**GEMMA 3** is a family of multimodal LLMs (1B–27B) from Google, and its training pipeline heavily leverages **knowledge distillation from a teacher model, followed by reward-based fine-tuning**. According to the *Gemma 3 Technical Report* and developer blog, Gemma 3’s post-training consists of four stages: *“Distillation from a larger instruct model, RLHF, RLMF (machine feedback for reasoning), and RLEF (execution feedback for coding)”* ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=For%20post,4%20components)). We focus on the **distillation and the reasoning reward (RLMF)**, which correspond conceptually to *“isolating teacher responses”* and *training the student with that guidance*, as asked.

- **Teacher Response Distillation:** In Gemma 3, the first post-training component is *offline distillation* from a stronger teacher. Essentially, they take a large teacher model (not publicly named, but likely a very strong instruction model – possibly a version of Google’s Gemini or PaLM 2), and use it to generate responses which the Gemma 3 student then learns to imitate ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=For%20post,4%20components)). By *isolating the teacher’s responses*, we mean the process of **using the teacher’s outputs as training targets** for the student, independent of any human labeling or environment reward. This is classic sequence-level knowledge distillation: the student’s loss is the cross-entropy on the teacher’s output sequence (or the KL between student and teacher distributions on each prompt). In practice, Gemma 3 implemented a scalable version of this:
  
  > *“We sample 256 logits per token, weighted by teacher probabilities. The student learns the teacher’s distribution within these samples via cross-entropy. The teacher’s target distribution is set to zero for non-sampled logits, and renormalized.”* ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=Distillation,sampled%20logits%2C%20and%20renormalized))

  This means for each token position, rather than using the full softmax over ~50k vocabulary (which is costly for KL on large models), they sample the top-$K$ tokens from the teacher’s distribution (here $K=256$) and have the student match that truncated distribution ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=Distillation,sampled%20logits%2C%20and%20renormalized)). This **logit sampling** is a clever implementation detail to make distillation efficient, yet it retains the essence: the student is directly learning to **predict the teacher’s likely tokens** at each step. By doing so across many prompts, the student gradually acquires the teacher’s knowledge and style. In effect, the teacher’s entire response (content and wording) guides the student’s learning – no external reward model is involved at this stage. Gemma 3 models trained with this distillation showed large gains in capabilities compared to their size; e.g., the Gemma 3 27B model after distillation (and further RL) outperforms some 70B models ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=In%20this%20section%2C%20we%20report,of%20the%20aforementioned%20models%20have)).

  This approach isolates the “teacher signal” in a pure form: before any reward optimization, the student is as close as possible to the teacher in behavior. However, one limitation is that the student might also pick up any *flaws or biases* of the teacher. Also, if the teacher is weaker in a certain domain (e.g. a language the teacher isn’t fluent in), straightforward imitation might propagate those weaknesses.

- **Reward Training (RLMF – Reasoning with Teacher Guidance):** After distillation, Gemma 3 applies *Reinforcement Learning from Machine Feedback* (RLMF) to further improve reasoning and math ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=,model%20predictions%20with%20human%20preferences)). The technical report references improved versions of algorithms like **BOND, WARM, WARP** for this RL fine-tuning ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=match%20at%20L292%20knowledge%20distillation%C2%A0,2024%29%2C%20WARM%C2%A0%28Ram%C3%A9)). These are recent techniques in RLHF:
  - **BOND** (Best-of-N Distillation) involves generating multiple outputs (N samples) for a given prompt (usually by a model or ensemble) and identifying the best output (via a reward model or some criteria), then **distilling that best output** into the model ([[PDF] Accelerating Iterative Best-of-N Distillation for LLM Alignment - arXiv](https://arxiv.org/pdf/2410.20727#:~:text=arXiv%20arxiv,tuning%20from%20some%20reference)). Essentially, it’s like *choosing the teacher’s best response out of many* and training the student on it. This aligns well with GRPO’s philosophy of group-based optimization. In Gemma’s context, BOND could mean they took the *teacher model itself*, had it produce N candidates for a task (say a math problem), then used an oracle (perhaps a programmatic checker or another model) to pick the correct/best answer, and distilled that into Gemma. By doing so repeatedly, the student learns to produce *the kind of output that a teacher would select as optimal out of a sample*. This **isolates the teacher’s *optimal* responses** rather than all responses, sharpening the student’s performance.
  - **WARM** (Weight Averaged Reward Modeling) and **WARP** (Weight Averaged Rewarded Policy) are advanced techniques to stabilize RL by ensembling or averaging models over training ([[PDF] WARM: On the Benefits of Weight Averaged Reward Models - AI-Plans](https://ai-plans.com/file_storage/4493230e-112b-43ed-95f0-b32544914336_warm_on_the_benefits_of_weight_averaged_reward_models_pWW7BvtN7H.pdf#:~:text=plans,WARM%20guides%20the%20RL%20procedure)) ([Figure 3 from WARP: On the Benefits of Weight Averaged Rewarded ...](https://www.semanticscholar.org/paper/WARP%3A-On-the-Benefits-of-Weight-Averaged-Rewarded-Ram'e-Ferret/aef234095eb56f3510737df572fd155668523bb0/figure/4#:~:text=Figure%203%20from%20WARP%3A%20On,policies%20in%20the%20weight)). In Gemma 3, improved WARM/WARP suggest that they trained multiple reward models and/or policies and averaged them to get a more robust final policy ([Gemma3 technical report detailed analysis : r/LocalLLaMA - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1j9iazd/gemma3_technical_report_detailed_analysis/#:~:text=Reddit%20www.reddit.com%20%20,Only)) ([Fine-Tuning Gemma 3 on Your Computer with LoRA and QLoRA (+ ...](https://substack.com/home/post/p-158975895?utm_campaign=post&utm_medium=web#:~:text=They%20used%20a%20combination%20of,like%20BOND%2C%20WARM%2C%20and%20WARP)). The detail is not critical for our purposes, but it indicates that *the RL phase was sophisticated*, using ensemble strategies to reduce noise in rewards and policy updates.

  The result of RLMF is a student model that not only imitates the teacher’s style (thanks to distillation) but can potentially **outperform the teacher on certain tasks** by leveraging machine-generated feedback. For example, the teacher might not always get a math problem right on the first try, but BOND-style iteration can find a correct solution that the student then learns to produce in one shot.

In simpler terms, **Gemma 3’s reward approach first learns directly from the teacher (distillation), then uses the teacher’s *knowledge* as a basis for further RL training**. By isolating teacher responses in the first stage, the student gets a strong prior. In the second stage, the student is trained with the teacher’s *guidance* in a different sense: the teacher (or a procedure involving the teacher) might still be generating the data or evaluations for RL. For instance, one could imagine the *teacher continues to be used to provide critiques or preferences* during RLMF – though the report specifically mentions “machine feedback” (likely automated checks) rather than the teacher itself as the reward giver.

**Teacher Response Isolation – Value:** This two-phase strategy is valuable because it decouples pure imitation learning from reward optimization. The distillation phase ensures the student speaks the language of the teacher (same format, knowledge, base quality), which is a **gentler starting point** for RL than a randomly initialized or purely pre-trained model. Then the reward phase fine-tunes specific abilities (reasoning, coding) *without deviating from the teacher’s general behavior too much*. Empirically, this worked: Gemma 3’s 27B chat model achieves an Elo score (LMSYS Arena) higher than some much larger models, and far above its Gemma 2 predecessor ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=In%20this%20section%2C%20we%20report,of%20the%20aforementioned%20models%20have)).

To summarize GEMMA 3’s relevant techniques:
- **Distillation (Teacher responses as training data):** Student is supervised on the teacher’s outputs, locking in the teacher’s knowledge and style.
- **RLMF (Machine/Teacher-aided RL):** Student is further trained via RL, often using teacher-involved strategies (like best-of-N selection) to maximize rewards for correct reasoning. The teacher’s role here is indirect – it might generate candidates or simply have taught the student enough to evaluate its own outputs.

In our context, GEMMA 3 highlights that *isolating teacher responses* (using them in a dedicated distillation stage) provides a strong foundation for a student, and that integrating such a stage with RL can yield a model that benefits from both imitation and reward-driven refinement. We will leverage these insights when combining with GRPO and MiniLLM methods.

## 4. Feasibility of a Hybrid GRPO + Distillation Approach

Given the above, we propose to **integrate GEMMA’s teacher-guided distillation and MiniLLM’s losses into the GRPO framework**. The goal is a hybrid training procedure for knowledge distillation that combines:
- **Teacher behavioral cloning** (as in Gemma’s distillation or MiniLLM’s reverse KL),
- **Reinforcement learning via GRPO** (ensuring strong performance on specific objectives, e.g. reasoning or correctness),
- **Pre-training regularization** ($L_{PT}$ to retain the student’s original language strengths).

Such an approach is **feasible** and potentially highly beneficial. Let’s break down how these components interplay and why they complement each other:

**a. Using Teacher Responses as “Reward” in GRPO:** In a pure knowledge distillation setting, the “reward” for the student is essentially how well it matches the teacher. We can incorporate this into GRPO by defining the reward $r$ for a generated output *based on the teacher’s output or distribution*. For example:
- We can set $r = \text{sim}(y_{\text{student}}, y_{\text{teacher}})$, where $y_{\text{teacher}}$ is the teacher’s reference answer and $\text{sim}$ is some similarity metric (e.g. BLEU, ROUGE, or simply negative edit distance). This would directly reward the student for matching the teacher’s answer. A perfect match gets a high reward, a dissimilar answer gets a low (or negative) reward. This is effectively *sequence-level imitation reward*.
- Alternatively, use the **teacher’s log-likelihood**: $r = \log p_{\text{teacher}}(y_{\text{student}}|x)$. This is exactly the signal used in MiniLLM’s reverse KL. Under GRPO, we could sample a *group of outputs from the student* for prompt $x$, compute each output’s reward = teacher log-prob, and baseline them relatively. This means the student is not required to match the teacher *exactly token-for-token*, but any output that the teacher would also likely produce (assigns high probability to) will be rewarded. This is a softer measure that allows the student to perhaps use different wording as long as the content aligns with the teacher’s preferences. It embeds the *teacher’s entire distribution* as the reward function.

Either way, **the teacher becomes the source of the reward signal**. This fits naturally with GRPO: instead of a human-trained reward model, we use the teacher as a reward model. By doing group generation and relative comparison (GRPO), the student will learn to *outperform its own average* with respect to teacher similarity. Over time, it should converge toward imitating the teacher’s top outputs.

One must ensure that the teacher’s signal is informative enough. If the teacher is very strong, $r$ will be higher for outputs that are correct and well-formed, and lower for incorrect ones. If the teacher is weak in a certain area (e.g., teacher’s Faroese is broken), then $r$ might inadvertently prefer flawed outputs. We address this below with additional terms.

**b. Integrating $L_{PT}$ (Pre-training loss):** We will maintain a running auxiliary loss on the student’s original pre-training data (or any relevant Faroese corpus, as we’ll discuss in Section 6). This is straightforward to add in a hybrid framework: at each update step, we compute gradients from two sources – (i) the **RL loss (GRPO)** on the prompts of interest, and (ii) the **LM loss** on a batch of pre-training text. The combined update $\Delta\theta$ is a weighted sum of both gradients ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)). This means the student is *simultaneously learning from the teacher and remembering its base language*. Feasibly, this is implemented by intermixing training data or having two separate data loaders and summing losses:
   ```python
   # Pseudocode snippet for combined loss
   loss_total = loss_grpo + lambda_pt * loss_pretrain
   loss_total.backward()
   optimizer.step()
   ```
   including appropriate rescaling ($\lambda_{pt}$) to ensure $L_{PT}$ doesn’t dominate or vanish. The **value of this integration** is clear: it prevents the student from unlearning Faroese vocabulary or general fluency while it chases the teacher’s output patterns ([](https://arxiv.org/pdf/2306.08543#:~:text=match%20at%20L2025%20on%20the,NLP%20tasks%20while%20keeping%20the)). Particularly because our scenario involves a smaller student with strong Faroese knowledge and a teacher with possibly weaker Faroese, $L_{PT}$ will act as a **safety net** to preserve the student’s language-specific strengths.

**c. Incorporating Teacher’s *content* and *feedback* (GEMMA’s idea):** We can incorporate Gemma’s approach by *first doing a supervised distillation pass*, then RL. However, an even tighter integration is possible: we can treat the teacher’s actual output $y_{\text{teacher}}$ as an *expert demonstration* and use it in two ways:
  1. **Behavior cloning loss:** Add a supervised loss term $L_{\text{BC}} = -\log q_\theta(y_{\text{teacher}}|x)$ (basically forward KL on teacher’s output). This is akin to standard KD. It ensures the student can reproduce the teacher’s exact answer when needed.
  2. **Policy gradient reward:** Also use the teacher’s answer to compute a reward for student samples, as discussed. For instance, measure the overlap between student output and the teacher’s answer (this could even be done token-wise: give a token-level reward for each token the student has in common with the teacher’s answer at the correct position, etc., though sequence-level might suffice).

  In essence, the hybrid could optimize a **mixed objective**: 
  $$ L = -\mathbb{E}_{y \sim q_\theta}[r_{\text{teacher}}(y)] + \alpha \, L_{\text{BC}} + \beta \, L_{PT} \,. $$
  Here $r_{\text{teacher}}(y)$ is the teacher-derived reward (e.g. log-prob or similarity), and $L_{BC}$ is the supervised distillation loss on the teacher’s chosen response. This covers *both* the case where we explicitly train on the teacher’s best answer (like Gemma’s distillation or BOND’s best-of-N) **and** the case where we allow exploration of other outputs with guidance from the teacher’s scoring.

  This is feasible within GRPO by expanding the “reward model”: we can define a composite reward that accounts for multiple factors:
   - e.g. $R_{\text{total}} = R_{\text{teacher}}(y) + R_{\text{other}}(y)$.
  For instance, we might include an **additional reward for outputting in Faroese correctly**. If the teacher is not fluent in Faroese, it might produce some incorrect grammar – the student, however, might have learned proper grammar from $L_{PT}$ data. We could introduce a simple rule-based reward $R_{\text{lang}}$: +1 if the student output is entirely in Faroese vocabulary and passes a Faroese spell-check, and -1 if it includes many English words or gibberish. This would incentivize the student to use its Faroese skills and not just copy the teacher’s potentially broken Faroese. Such a reward could be seen as analogous to how DeepSeek added format and consistency rewards to ensure outputs followed a desired structure ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,between%20%E2%80%98%E2%80%99%20and%20%E2%80%98%E2%80%99%20tags)).

**d. Group Baseline in Hybrid Setup:** Using GRPO means even the teacher-guided reward will be baselined by group averages. This is actually beneficial for distillation: if all $G$ samples from the student are poor compared to the teacher, their rewards might all be low, but GRPO will focus on the *relative* differences. Over time, as the policy improves, the baseline shifts upward, continually challenging the model to get even closer to the teacher. This avoids the need to have an absolute threshold of success – the student just keeps trying to beat its last performance relative to the teacher’s standard. It’s a kind of self-play against the teacher’s shadow.

**e. Feasibility Considerations:** 
- We assume access to the teacher model for generating outputs and computing log-likelihoods. This can be done offline (generate a fixed dataset of teacher Q&A pairs for supervised learning) or online (query the teacher on the fly for each student sample). Online use of the teacher in the loop is more expensive but allows dynamic feedback and true RL (the MiniLLM approach is essentially this, treating teacher as a black-box reward provider).
- The student’s training stability can be managed by tuning the KL term and perhaps periodically resetting the reference policy (as in PPO/GRPO) to avoid drifting too far. Since we have $L_{BC}$ anchoring the student to the teacher’s behavior, huge divergences are unlikely.
- From a value standpoint, this hybrid method could significantly **accelerate training**. Standard RLHF often requires hundreds of thousands of comparison labels or a trained reward model ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=The%20training%20process%20requires%20you,is%2C%20given%20the%20user%20query)). Here, the teacher model *replaces the reward model and the human labeler*. Its knowledge is baked into the reward and the supervised targets. This is particularly useful for low-resource languages like Faroese where human feedback data is scarce; instead, we leverage a high-resource teacher’s knowledge.

In conclusion, integrating GEMMA 3’s teacher response distillation and MiniLLM’s reverse-KL + $L_{PT}$ into GRPO is not only feasible, but it synergistically combines **imitation and reinforcement**:
- The student gets **explicit guidance** (exact answers) from the teacher, ensuring knowledge transfer.
- GRPO adds an **optimization lens** that can incorporate additional goals (like correctness, Faroese fluency) by tweaking reward functions, and improves sample efficiency via relative comparisons.
- $L_{PT}$ guarantees the student doesn’t lose its unique strengths (e.g. vocabulary, baseline fluency).

Next, we detail an **algorithmic blueprint** for this hybrid training and provide pseudocode and a PyTorch-style implementation outline.

## 5. Hybrid Training Algorithm: GRPO + Distillation

Below is an **algorithmic flowchart** and pseudocode for the proposed hybrid method, which we might call **GRPO-KD (Group Relative Policy Optimization for Knowledge Distillation)**. We assume we have:
- A **teacher model** $M_T$ (large LLM).
- A **student model** $M_S$ (smaller LLM) with parameters $\theta$.
- A dataset of prompts $D_{\text{task}}$ (task examples we care about, e.g. questions in Faroese).
- A **pre-training corpus** $D_{PT}$ (raw text, including Faroese content).
- (Optionally) Some automatic metrics or rules for additional rewards (e.g. language correctness).

**Algorithm: Hybrid GRPO-KD Training**  
*(combining teacher distillation, GRPO updates, and pretraining loss)*

1. **Initialize Student:** Start with $M_S$ pre-trained on $D_{PT}$ or other corpora (so it has a strong Faroese vocabulary to begin with). Initialize $\pi_\theta = M_S$. Optionally initialize a **reference policy** $\pi_{\text{ref}} \leftarrow \pi_\theta$ for KL regularization.

2. **Supervised Distillation Warm-up (optional):** For each prompt $x$ in a warm-up subset of $D_{\text{task}}$, obtain the teacher’s answer $y_T = M_T(x)$. Train $M_S$ to maximize $\log p_\theta(y_T|x)$ (supervised learning). This can be one epoch over data or interleaved in the RL loop. *Rationale:* gives $M_S$ a good starting point close to teacher, improving initial rewards.

3. **Reinforcement Loop:** Repeat for a number of iterations or until convergence:
   1. **Sample Task Batch:** Sample a batch of prompts $\{x_i\}_{i=1}^N$ from $D_{\text{task}}$.
   2. **Generate Student Outputs:** For each prompt $x_i$, sample a *group* of $G$ outputs from the student: $\{y_{i,1}, ..., y_{i,G}\} \sim \pi_\theta(\cdot|x_i)$. (We could use nucleus or random sampling to get diverse candidates.)
   3. **Teacher Evaluation:** For each output $y_{i,j}$, have the teacher model compute a score. Two modes:
      - **Log-Likelihood:** $r_{i,j}^{(T)} = \log p_{M_T}(y_{i,j}|x_i)$ (the teacher’s log-probability of the student’s output).
      - **Comparison to Teacher Answer:** Alternatively, have teacher generate its preferred answer $y_i^*$ for $x_i$ (if not done in warm-up), then define $r_{i,j}^{(T)} = \text{Sim}(y_{i,j},\, y_i^*)$ where Sim is e.g. ROUGE-L or exact match if it’s a QA task. This requires the teacher’s reference answer as ground truth.
      
      We can use either. The log-likelihood approach uses the teacher as a **reward model** (it scores any attempt), whereas the reference answer approach treats the teacher’s single answer as ground truth (closer to supervised). In practice, log-likelihood is smoother and was shown effective in MiniLLM ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)), so we lean towards that.
   4. **Additional Rewards:** Compute any additional rewards $r_{i,j}^{(aux)}$ for each output. For example:
      - Language reward: $+1$ if output is in Faroese with correct characters/words, $-1$ if not.
      - Format or length rewards if needed (not crucial unless we have format constraints).
      These help encode requirements the teacher might not cover.
   5. **Combine Rewards:** $r_{i,j} = r_{i,j}^{(T)} + r_{i,j}^{(aux)}$. (We may normalize these rewards across the group or batch for stability.)
   6. **Compute Advantages:** For each prompt $i$, compute the group baseline $b_i = \frac{1}{G}\sum_{j=1}^G r_{i,j}$ (the mean reward of the group for prompt $i$). Then advantage $A_{i,j} = r_{i,j} - b_i$. Optionally, normalize $A_{i,j}$ by the std dev of rewards in the group (as DeepSeek did) ([Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/#:~:text=1,5)).
   7. **Policy Gradient Step (GRPO):** Compute the policy loss:
      $$ L_{\text{policy}} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^G \frac{\pi_\theta(y_{i,j}|x_i)}{\sum_{k=1}^G \pi_\theta(y_{i,k}|x_i)} A_{i,j} \,. $$
      In practice, this is implemented by treating each $y_{i,j}$ as an on-policy sample from $\pi_\theta$ and using the REINFORCE gradient $-\nabla \log \pi_\theta(y_{i,j}|x_i) \cdot A_{i,j}$. (The fraction shown is conceptually how one might sample without replacement; but since we already have the samples, we just weight by $A$.) We also add a KL penalty to the loss: $L_{KL} = \beta \, \text{KL}(\pi_\theta(\cdot|x_i) \parallel \pi_{\text{ref}}(\cdot|x_i))$ averaged over $i$. This ensures the student doesn’t diverge too fast from the reference (which could be the initial model or periodically updated model).
   8. **Pre-training Loss:** Sample a batch of texts from $D_{PT}$. Compute $L_{PT} = -\frac{1}{|D_{PT,batch}|}\sum_{d \in \text{batch}} \log p_\theta(d)$, the student’s NLL on this text.
   9. **Total Loss and Update:** Combine losses: 
      $$ L_{\text{total}} = L_{\text{policy}} + L_{KL} + \lambda_{PT} L_{PT} + \lambda_{BC} L_{BC}, $$
      where $L_{BC}$ is a supervised loss on teacher reference outputs if we include that (from step 3, if teacher answers $y_i^*$ are available). $\lambda_{PT}$ and $\lambda_{BC}$ are hyperparameters to weight the regularizers. We then do a gradient descent step on $L_{\text{total}}$ to update $\theta$. 
      
      If using mini-batch advantage estimates, we might do multiple mini-batch updates per sample (like PPO epochs). GRPO’s formulation often allows a few updates with the computed advantages before regenerating new samples.
   10. **Update Reference (optional):** After each iteration or epoch, you may set $\pi_{\text{ref}} \leftarrow \pi_\theta$ (or use a moving average) to slowly move the reference point along with the student, keeping KL in check.

4. **Convergence:** Repeat until the student’s performance converges or improves to the desired level. We monitor metrics like the average teacher reward or task success rate. We may anneal $\lambda_{PT}$ over time (heavy regularization early on, then less). We also stop if the KL divergence between student and initial model becomes too high – indicating it might be learning oddities.

This algorithm essentially **blends supervised distillation and RL**. Notably, if $\lambda_{BC}$ is high and we remove the sampling of other outputs, it reduces to plain teacher-forcing on $y_T^*$. If $\lambda_{BC}=0$ and we rely purely on the reward from teacher probabilities, it becomes the MiniLLM reverse-KL RL. With GRPO’s group advantage, we get stability and potentially better exploration: the student tries various outputs and learns even from sub-optimal ones by comparison. 

Let’s illustrate this with a **PyTorch-style pseudocode** for one training step:

```python
# Pseudocode PyTorch training loop for hybrid GRPO-KD
import torch
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
lambda_PT = 0.1
lambda_BC = 0.5

for batch in task_loader:  # batch of prompts
    prompts = batch['prompts']  # list of prompt texts
    # 1. Teacher reference answers (for BC loss, optional)
    with torch.no_grad():
        teacher_outs = teacher_model.generate(prompts)  # greedy or high-quality generations
    
    # 2. Generate group of student outputs per prompt
    student_outputs = []
    for prompt in prompts:
        outs = [student_model.generate(prompt, max_length=MAX_LEN, do_sample=True) 
                for _ in range(G)]
        student_outputs.append(outs)
    
    # 3. Compute rewards for each output
    all_log_probs = []
    for prompt, outs in zip(prompts, student_outputs):
        # Teacher log-prob reward: compute log p_Teacher(y | prompt) for each output y
        # We'll flatten outputs and compute log-probs in a single batch for efficiency
        enc = teacher_tokenizer([prompt]*len(outs), outs, return_tensors='pt', padding=True)
        # teacher_model outputs logit scores; compute log-prob of each provided output token
        with torch.no_grad():
            teacher_logits = teacher_model(**enc).logits  # shape: [batch_size, seq_len, vocab_size]
        # Use enc.labels (the token IDs of outputs) to gather log-probs
        log_probs = torch.gather(torch.log_softmax(teacher_logits, dim=-1), -1, enc.labels.unsqueeze(-1))
        # Sum log-probs per sequence (taking care of padding)
        seq_log_probs = log_probs.squeeze(-1).sum(dim=1)  # sum over seq_len
        all_log_probs.append(seq_log_probs)  # list of tensors for each prompt
    
    # 4. Compute auxiliary rewards (e.g., language check)
    aux_rewards = []
    for outs in student_outputs:
        rw = torch.zeros(len(outs))
        for j, out_text in enumerate(outs):
            if is_faroese(out_text):
                rw[j] += 0.1  # small bonus for Faroese output
            else:
                rw[j] -= 0.1  # penalty if not
        aux_rewards.append(rw)
    
    # 5. Combine rewards and compute advantages
    policy_loss = 0.0
    kl_loss = 0.0
    total_samples = 0
    # Flatten outputs for loss computation
    student_log_probs = []  # will store log πθ(y|x) for each output
    advantages = []
    for i, outs in enumerate(student_outputs):
        R = all_log_probs[i] + aux_rewards[i]  # combined reward tensor for this prompt's outputs
        # Baseline: mean reward
        baseline = R.mean()
        adv = R - baseline
        # Normalize advantage (optional)
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-8)
        advantages.append(adv)
        total_samples += len(outs)
        # Compute student log-probs of its outputs (for policy gradient and KL)
        enc = student_tokenizer([prompts[i]]*len(outs), outs, return_tensors='pt', padding=True)
        student_logits = student_model(**enc).logits  # forward pass
        logp = torch.gather(torch.log_softmax(student_logits, dim=-1), -1, enc.labels.unsqueeze(-1))
        seq_logp = logp.squeeze(-1).sum(dim=1)  # log πθ(y|x) for each output
        student_log_probs.append(seq_logp)
        # KL loss: compare student and reference (initial) logits distribution per prompt (if needed)
        if use_kl_penalty:
            with torch.no_grad():
                ref_logits = reference_model(**enc).logits
            # Compute KL divergence between student and reference token distributions
            # (for simplicity, we approximate sequence KL as sum of token KLs)
            student_dist = torch.log_softmax(student_logits, dim=-1)
            ref_dist = torch.softmax(ref_logits, dim=-1)  # reference probabilities
            token_kl = (torch.softmax(student_logits, dim=-1) * (student_dist - torch.log(ref_dist+1e-8))).sum(dim=-1)
            kl_loss += token_kl.mean().item()  # accumulate KL (as scalar) - or keep as tensor for grad
    
    # 6. Calculate policy gradient loss using advantages
    policy_loss = 0.0
    # We have student_log_probs as list per prompt; advantages as list per prompt
    for seq_logp, adv in zip(student_log_probs, advantages):
        # REINFORCE: negative log-prob * advantage (note: advantage is detached from graph to avoid affecting gradient)
        policy_loss += -(seq_logp * adv.detach()).mean()
    policy_loss = policy_loss / len(student_outputs)  # average over prompts
    
    # 7. Supervised BC loss on teacher outputs (optional)
    bc_loss = 0.0
    if lambda_BC > 0:
        enc_bc = student_tokenizer(prompts, teacher_outs, return_tensors='pt', padding=True)
        logits = student_model(**enc_bc).logits
        # Compute cross-entropy loss of student logits vs teacher_outs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc_bc.labels[..., 1:].contiguous()
        bc_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                                    shift_labels.view(-1), ignore_index=student_tokenizer.pad_token_id)
    
    # 8. Pre-training loss on separate batch of PT data
    pt_batch = next(pretrain_iter)  # get a batch of raw text
    enc_pt = student_tokenizer(pt_batch, return_tensors='pt', padding=True, truncation=True)
    logits_pt = student_model(**enc_pt).logits
    shift_logits_pt = logits_pt[..., :-1, :].contiguous()
    shift_labels_pt = enc_pt.input_ids[..., 1:].contiguous()
    pt_loss = torch.nn.functional.cross_entropy(shift_logits_pt.view(-1, shift_logits_pt.size(-1)), 
                                                shift_labels_pt.view(-1), ignore_index=student_tokenizer.pad_token_id)
    
    # 9. Combine losses
    total_loss = policy_loss + (beta * kl_loss if use_kl_penalty else 0) \
                 + lambda_BC * bc_loss + lambda_PT * pt_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

*Explanation:* This code integrates all pieces: generating outputs, computing teacher-based rewards (`all_log_probs` holds $\log p_T$ for each output), adding a Faroese language reward (`aux_rewards`), computing advantages per prompt, then computing the policy gradient loss by weighting student log-probs with those advantages. We also show how a supervised loss (`bc_loss`) on the teacher’s outputs and a pre-training loss (`pt_loss`) are computed and added. The `kl_loss` portion (commented conceptually) would ensure the student doesn’t deviate too much from the reference (could be initial weights or periodically updated weights). 

**Flowchart of Training Process:** Pseudo-graphically, each iteration does:
1. **(Teacher & Student output generation)**: ![Step1](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJsAAABkCAYAAADxK+aAAAABGklEQVR4nO3BMQEAAAzAsOZf9DbwAAAI/wMAAAAAQIEAAAAAAAAgF/+AgQAAICAAAAAAAAAAAAAgIEAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAACBAQAAAIAAAAAAAAAAAAAAIAAAAD8mgiSXG5ucAAAAAElFTkSuQmCC) Teacher model (blue) and Student model (orange) both take the prompt. The teacher either outputs a sample answer or is used to score the student’s outputs. The student produces multiple tentative answers.
2. **(Reward Computation)**: The teacher’s output or scoring mechanism evaluates each student answer (giving a reward/score $r$), and additional checks (language) provide auxiliary scores. We aggregate those into a reward for each answer.
3. **(Compute Advantages)**: We calculate the mean reward per prompt and the advantage for each answer relative to that mean.
4. **(Policy Update)**: We adjust the student model’s weights so that answers with above-average reward are more likely (advantage > 0 => increase log-prob) and below-average are less likely (advantage < 0 => decrease log-prob). Meanwhile, we also apply the supervised imitation loss on the teacher’s best answer (ensuring the student can reproduce it) and a general language modeling loss (ensuring it doesn’t forget general Faroese or other knowledge).
5. **(Repeat)**: Iterate this process, gradually improving the student.

**Comparison to Standard Methods:** This hybrid can be seen as combining the strengths of each method:
- Like **MiniLLM**, it uses the teacher’s *distribution* as a target (via the log-prob reward) ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)), which is a principled way to do KD for LLMs.
- Like **GRPO**, it uses *relative advantages* ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=3,KL%20term%20within%20the%20reward)) to stabilize training and avoid needing a separate value network.
- Like **Gemma 3**, it explicitly leverages the teacher’s actual responses (via the BC loss) and then goes beyond via RL optimization of secondary objectives ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=For%20post,4%20components)).

This approach should be at least as sample-efficient as MiniLLM’s (which already showed strong results with relatively few RL steps by using the teacher’s own knowledge to guide it ([](https://arxiv.org/pdf/2306.08543#:~:text=covers%20a%20large%20range%20of,with%20neglectable%20loss%20of%20diversity))). By adding GRPO’s grouping, we might get even better performance on tasks where multiple attempts can be compared (e.g., the model can internally generate a few different phrasings and learn which the teacher likes best).

The pseudocode above would run on each batch; in practice, one would incorporate training loops, validation checks, model saving, etc. But it demonstrates the **integration of teacher reward and $L_{PT}$ into a GRPO-style update**.

## 6. Case Study: Distilling a Faroese-Focused LLM

Now we evaluate how this hybrid approach can help distill a smaller model that excels in **Faroese**, given:
- The **student** has strong Faroese vocabulary and some language proficiency (from pre-training on Faroese or related languages; effectively a low-resource language specialist).
- The **teacher** is a much larger model with superior general reasoning, world knowledge, and possibly multi-lingual capabilities, but not specifically tuned for Faroese (hence might have weaker Faroese lexical or grammatical mastery).

**Why this scenario?** Faroese is a low-resource language (~50k native speakers) and large models often underperform in such languages due to limited training data ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=Recent%20NLP%20advancements%2C%20driven%20by,are%20crucial%20for%20assessing%20and)). A smaller model pre-trained or fine-tuned on Faroese text might actually produce more fluent Faroese output. However, that smaller model won’t match a big model’s performance in complex reasoning or broad knowledge. By distilling a large model’s **knowledge and reasoning** into the small model *while preserving the small model’s language-specific strengths*, we aim to get the best of both: a model that *thinks* like the big model but *speaks* like a native Faroese speaker.

**How the Hybrid Approach Helps:**

- **Retention of Faroese Vocabulary:** The $L_{PT}$ (or continued MLM training) on Faroese texts ensures that during RL, the student doesn’t “forget” rare Faroese words or the ability to form correct Faroese sentences ([](https://arxiv.org/pdf/2306.08543#:~:text=match%20at%20L2025%20on%20the,NLP%20tasks%20while%20keeping%20the)). As the student learns from the teacher, some teacher words or structures might leak in (e.g., the teacher might use an English word if it doesn’t know the Faroese term). The pre-training loss will continuously pull the student back toward genuine Faroese usage by training on Faroese corpus. Essentially, it acts as a **constraint that the output remains in the target language**. We could even augment $L_{PT}$ with a **Faroese corpus exclusively**, to really emphasize that any deviation from Faroese incurs a penalty. This way, even if the teacher isn’t perfect in Faroese, the student will prefer its own internal Faroese knowledge for actual wording.

- **Teacher’s Reasoning and Knowledge Transfer:** The teacher, being very large, will often produce the correct answer or a very useful chain-of-thought for a given prompt (even in Faroese, it might get the content right but grammar wrong, or it might respond in English if it’s unsure in Faroese). Our method leverages this in multiple ways:
  - If the teacher outputs a correct answer (in whatever language), the supervised $L_{BC}$ can train the student on the translated or equivalent Faroese answer. For example, suppose the prompt and teacher’s answer are:

    *Prompt (Faroese):* *“Hvat eitur høvuðsstaðurin í Brasil?”* (What is the capital of Brazil?)  
    *Teacher’s answer:* *“The capital of Brazil is Brasília.”* (in English)

    The student might see this and our algorithm can translate or require the answer in Faroese. Perhaps we’d prefer the teacher to answer in Faroese directly; if the teacher can be prompted to respond in Faroese, great. If not, we could feed an English answer into a Faroese translation model or even have the student try to produce the answer in Faroese and use the teacher for checking correctness. In any case, the content “Brasília” is the key knowledge from the teacher. The student already knows in Faroese the country is “Brasil” and how to form "*Høvuðsstaðurin er Brasília.*". So, the training might incorporate a step where the student is rewarded for saying "Brasília" in the output (because teacher’s content matches that).
    
    Over time, through many QA examples, the student **accumulates factual knowledge and reasoning patterns** from the teacher (it learns that “capital of X” questions are answered with “The capital of X is Y”), while speaking in Faroese.
    
  - The chain-of-thought or reasoning aspect: If the teacher tends to do multi-step reasoning (e.g., for math problems or complex questions), we can encourage the student to do the same. DeepSeek’s approach of including special reasoning tags ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=,between%20%E2%80%98%E2%80%99%20and%20%E2%80%98%E2%80%99%20tags)) could be adopted. For instance, we could format prompts so that the student is asked to think step by step (and perhaps the teacher’s reward model checks the steps). The large teacher likely has such capabilities (e.g., GPT-4 can do chain-of-thought). The student, via reward, will pick up these behaviors. In Faroese, that might mean the student learns to produce a reasoning in Faroese. If teacher can’t produce Faroese reasoning, we might use a bilingual approach: the teacher’s English reasoning can be translated and used as reference. This is admittedly complex, but feasible with additional translation models or by leveraging closely related languages (like Danish, which Faroese speakers often also know – a Danish reasoning could be more easily translated to Faroese).
  
- **Bridging the Language Gap:** A core challenge is the teacher’s weaker Faroese vocabulary. The hybrid approach addresses this by **separating knowledge from language** in training signals:
  - The reward $r_{i,j}^{(T)} = \log p_T(y|x)$ doesn’t care what language $y$ is in per se, it cares whether $y$ is something the teacher finds likely *given its training*. If the teacher is multilingual to some extent, it might still assign decent probability to a correct Faroese answer (especially if the question was in Faroese – many large models will continue in the same language). If the teacher completely fails to assign meaningful probabilities (say it has almost no Faroese understanding), another strategy is needed: we could prompt the teacher in English (translated prompt) to get an English answer, and separately handle translation. This becomes a two-teacher scenario (one teacher for content, another for language, or a translation system stepping in). However, assuming the teacher is at least moderately trained on Faroese (most big models have seen some Faroese, e.g., Gemma3 and LLaMA2 are trained on hundreds of languages ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=Gemma%203%20introduces%20multimodality%2C%20supporting,tuned%20versions)), likely including Faroese to a tiny degree), the teacher log-likelihood might still rank a correct Faroese answer above an incorrect one. The $L_{PT}$ on Faroese ensures the student will actually produce the Faroese text even if the teacher’s probability is not maximal for it initially. Over iterations, the teacher’s log-prob reward might start low for perfectly fluent Faroese, but as the student sticks to Faroese (due to $L_{PT}$) and only improves in content accuracy, the reward will increase because the *content* (which the teacher truly cares about for probability) is aligning, and the teacher might start to assign higher probability to the correct content even if phrased in Faroese (especially if it recognizes names or numbers).
  - We also can incorporate a **bilingual evaluation**: e.g., for each output, translate the student’s Faroese answer to English and have the teacher evaluate that (or vice versa). If the teacher says “Brasília” is capital, a Faroese “Høvuðsstaðurin er Brasília” when translated to English is “The capital is Brasília”, which the teacher would score highly. This way, we circumvent the teacher’s limited Faroese by evaluating in English. This could be an advanced component of the reward model (and would require a Faroese-to-English translator, possibly even the student’s own bilingual ability or a separate model). This is an extension beyond the current scope but worth noting as feasible.

**Expected Outcome:** With this training, we expect the student model to significantly improve on tasks in Faroese that require reasoning or broad knowledge. For example:
- **Question Answering (QA):** The student should approach the teacher’s accuracy on Faroese QA tasks. In *FoQA* (Faroese QA dataset) ([Paper page - FoQA: A Faroese Question-Answering Dataset](https://huggingface.co/papers/2502.07642#:~:text=We%20present%20FoQA%2C%20a%20Faroese,released%20in%20three%20versions%3A%20a)), GPT-4-turbo achieved the highest F1 (around 75) while a Faroese BERT (FoBERT) only got F1 ~36 ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,5)) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=FoBERT,5)). A distilled student might reach somewhere in between or even close to GPT-4’s level on that dataset. The Gemma-3 8B model (instruction-tuned) got F1 ~73.6 on FoQA, nearly matching GPT-4 ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,5)). If our student is ~7B, this is plausible. It would vastly outperform older small models that didn’t have teacher guidance.
- **General Dialogue or Knowledge:** The student would be able to converse in Faroese with the factual accuracy of the teacher. For instance, if asked a historical question in Faroese, the student would recall facts like the teacher does, but articulate the answer in fluent Faroese. Even subtle culture or idioms – if the teacher knows them or if included in $D_{PT}$ – could be preserved.

Crucially, this method uses no explicit Faroese human annotations beyond what’s in $D_{PT}$ and the initial prompts. It’s a way to bootstrap a high-quality Faroese model **without needing a Faroese RLHF dataset** – instead, we leverage the English (or multi-lingual) RLHF that was presumably done on the teacher.

**Potential Pitfalls and Mitigations:**
- If the teacher provides incorrect answers in some cases (especially if it is weak in Faroese and misunderstands the question), the student might learn those errors. To mitigate this, one could incorporate multiple teachers or a verification step (for instance, use an English teacher and a back-translation check). Or use the reward model approach: if the teacher isn’t sure, the reward might be low for all outputs, and GRPO will then mostly rely on relative difference. If all outputs are wrong, relative advantage might be small and the gradient small – meaning the student won’t update much on that prompt. So it won’t strongly internalize wrong info unless it occasionally by chance gets it right and the teacher confirms.
- If the student’s Faroese is far superior to the teacher’s, the teacher’s log-likelihood might not sufficiently differentiate a grammatically correct vs incorrect Faroese answer (it might just find both low-probability). In that case, having an **auxiliary language reward** (like a simple grammar model or even a rule-based check) is helpful as we included. Also, as noted, using translations for evaluation is an option.

Overall, the approach is a powerful way to build a *Faroese expert model* by transferring knowledge from a general model and using the student’s inherent Faroese strength to present that knowledge properly.

## 7. Candidate Teacher and Student Models

For a concrete implementation, we should choose appropriate teacher and student models. Below we list some candidates and their properties:

| **Model**                | **Size**    | **Training**                                 | **Strengths**                           | **Weaknesses**                    | **Role**    |
|--------------------------|------------|----------------------------------------------|----------------------------------------|-----------------------------------|-------------|
| **GPT-4 / GPT-3.5**      | ~>100B (est)| Trained on vast data (many languages)         | Excellent reasoning, broad knowledge; has seen some Faroese ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=Our%20evaluation%20of%20the%20FoQA,these%20models%20handle%20Faroese%20specifically)) | Closed source, cannot fine-tune; unknown Faroese proficiency (likely moderate) | Teacher via API (for eval or data generation) |
| **LLaMA-2 70B**          | 70B        | Multilingual pretraining (English-heavy, some Nordic) + fine-tuned versions | Strong general LLM, open model; good reasoning for open-source | Faroese data in pretraining is minimal, might make mistakes in Faroese | Teacher (if RLHF-tuned variant available) or high-end student |
| **Gemma-3 27B (IT)**     | 27B        | Pretrained on 14T tokens, 140+ languages; distilled from a larger model ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=,trained%20checkpoints)) | State-of-art open model; supports many languages, long context; RLHF optimized ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=In%20this%20section%2C%20we%20report,of%20the%20aforementioned%20models%20have)) | 27B still large to finetune; Faroese is likely included but small portion ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=Distillation,sampled%20logits%2C%20and%20renormalized)) | Teacher (for distillation) or high-quality student (if we use an even bigger teacher) |
| **GPT-SW3 6.7B**         | 6.7B       | Trained on Nordic languages (SV, DA, NO, IS; reportedly not on Faroese directly) ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=OpenAI%E2%80%99s%20GPT,score%20considering%20translation%20quality%20and)) | Designed for Nordic family; can understand some Faroese via similarity transfer ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=GPT,score%20considering%20translation%20quality%20and)); open source | Weaker reasoning than larger models; not specifically Faroese-trained (no Faroese corpus) | Possible student (after fine-tuning on Faroese) |
| **Mistral 7B**           | 7B         | Trained on a large multilingual web corpus; strong base model   | Good general ability for 7B; open source (Apache 2) | Not instruction-tuned out of box; unknown Faroese coverage | Student (after Faroese fine-tune + instruction tune) |
| **Bloom / Bloomz 7.1B**  | 7.1B       | Multilingual (130 languages) pretrain; Bloomz is instruction-tuned | Coverage of many languages; possibly some Faroese (though primarily higher-resource ones) | 7B scale, and Faroese might still be very minimal in training data | Student (with further Faroese adaptation) |
| **FoBERT** (Faroese BERT)| 110M       | Trained on Faroese texts (Sosialurin news, etc.) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=FoBERT,5)) | Excellent Faroese understanding (for its size) | Not a generative model, and far smaller; can’t reason or generate long text | N/A (could be used to initialize embeddings or as a reference for Faroese vocab) |
| **LLaMA-2 13B**          | 13B        | Same as 70B but smaller scale                | Middle ground – some reasoning ability, easier to fine-tune | Faroese knowledge still minimal without adaptation | Could be either: a smaller teacher or a larger student |
| **OpenAI GPT-3 (davinci)**| 175B (davinci)| English heavy training, some multilingual    | Strong general performance             | Closed, older (no RLHF for Faroese specifically) | Teacher (if open alternatives not sufficient) |

For our case, a likely pairing is:
- **Teacher:** An instruction-tuned model of the largest available kind that can be run or accessed. Since GPT-4 is closed, we might use **LLaMA-2 70B Chat** or **Gemma-3 27B** (Instruction-tuned version). Both are strong and support many languages. Gemma-3 27B is especially appealing because it has undergone a distillation and RL process already ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=,trained%20checkpoints)), and it supports *140+ languages* ([
            
            Introducing Gemma 3: The Developer Guide
            
            
            - Google Developers Blog
            
        ](https://developers.googleblog.com/en/introducing-gemma3/#:~:text=Gemma%203%20introduces%20multimodality%2C%20supporting,tuned%20versions)) – Faroese is likely among those (even if only through transfer). In fact, Gemma-3 being “top open compact model” indicates it could serve as a high-quality teacher. Another option is **GPT-3.5-turbo (OpenAI)** via API for generating teacher answers and scores in Faroese – GPT-3.5 has been evaluated on Faroese tasks like FoQA and performs well (F1 in high 60s) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,6)).
- **Student:** A 7B-scale model that we can fine-tune. **Mistral-7B** is a great base because of its strong performance among 7B models and open license. We would further pre-train or at least fine-tune Mistral on any Faroese text we have (to give it the vocabulary and fluency – essentially creating a “Mistral-Faroese”). Alternatively, **GPT-SW3 6.7B** from AI Sweden is already focused on Nordic languages; one could fine-tune it on Faroese data (noting that GPT-SW3 wasn’t initially trained on Faroese, but adding Faroese data would adapt its Scandinavian knowledge to Faroese, which is closely related to Icelandic and Danish). **LLaMA-2 7B** can also be used, with a Faroese adaptation. We might call the resulting model something like “Farø-LLaMA-7B”.

**Why these sizes?** A 7B student is small enough to be efficient for inference (can run on a single GPU with 16-32GB memory) which is practical for Faroese users, and large enough to hold complex reasoning patterns (with help from the teacher). The teacher at 27-70B ensures a large gap in capability that justifies distillation.

**Additional Considerations:** We should ensure the **tokenizer** of the student covers Faroese characters and words. Faroese uses some special characters like **ð, ø, á, í, ó, ú, ý** (common in Icelandic/Danish too) and **æ**. Most multilingual models include these. If not, we’d extend the tokenizer. E.g., LLaMA’s tokenizer might not have “ø” as a single token, but BPE can still form it. This is a minor detail but important for fidelity.

**Model Pairing Recommendations:**
- *Option 1:* **Teacher: Gemma-3 27B Chat**, **Student: Mistral 7B (Faroese-adapted)**. Use Gemma-3’s responses as ground truth and reward signals. Mistral 7B is fine-tuned on Faroese news/Wikipedia first (to create **Mistral-FA**). Then apply our hybrid RL training.
- *Option 2:* **Teacher: LLaMA-2 70B Chat**, **Student: LLaMA-2 7B** (Faroese fine-tuned). This keeps architecture similar (which can sometimes simplify distillation because tokenization and some representations align), though it’s not necessary they be same family. LLaMA-2 70B’s Faroese might be mediocre, but its general knowledge is great. The 7B, after being fine-tuned on Faroese text (and possibly loaded with the vocabulary), would learn from 70B. One might also use the intermediate **LLaMA-2 13B** as a stepping stone or co-teacher.
- *Option 3:* **Teacher: GPT-4 via API**, **Student: Mistral 7B or GPT-SW3 6.7B**. We could query GPT-4 for a large set of Faroese prompts (e.g., ask it to answer in Faroese – GPT-4 is actually quite good at producing Faroese if instructed, as it has strong multilingual capabilities). Use those as supervised data and also for reward (GPT-4 can score outputs via the API if asked to judge correctness). This would give top-notch guidance, but cost and API limits are considerations. Still, for critical applications, GPT-4 as a teacher might yield the best student.

We note that in the FoQA paper, a **LLaMA-3.1-8B** model (likely an experimental newer LLaMA version) reached nearly the same performance as GPT-4 on Faroese QA ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,6)), suggesting that with the right fine-tuning, even 8B can excel. Our approach basically describes *how to achieve such fine-tuning via distillation*.

## 8. Faroese Language Models and Evaluation Resources

Finally, to measure and guide our distillation, we should be aware of available **Faroese datasets and benchmarks**. Fortunately, interest in Faroese NLP has grown, and a few resources exist:

- **FoQA (Faroese Question Answering dataset):** Introduced in 2025 ([Paper page - FoQA: A Faroese Question-Answering Dataset](https://huggingface.co/papers/2502.07642#:~:text=We%20present%20FoQA%2C%20a%20Faroese,released%20in%20three%20versions%3A%20a)), it contains 2,000 extractive QA pairs from Faroese Wikipedia, plus additional generated samples ([Paper page - FoQA: A Faroese Question-Answering Dataset](https://huggingface.co/papers/2502.07642#:~:text=We%20present%20FoQA%2C%20a%20Faroese,released%20in%20three%20versions%3A%20a)). It’s a benchmark for reading comprehension in Faroese. Baselines on FoQA include GPT-4-turbo (F1 ≈75.2) and GPT-4o-mini (a smaller GPT-4, F1 ≈75.2) as the top, LLaMA-3.1-8B (F1 73.6), GPT-SW3-6.7B (63.4), Mistral-7B (62.4), and Faroese BERT (36.0) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,5)) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=FoBERT,5)). This dataset is excellent for evaluating how well our student learned factual QA in Faroese. Improvement from ~36 F1 to 60–70 F1 would be a huge leap, demonstrating effective knowledge transfer.
- **Faroese NER (FoNE) and other datasets:** Snæbjarnarson et al. (2023) created a Faroese Named Entity Recognition corpus by annotating the *Sosialurin* newspaper texts ([[PDF] Transfer to a Low-Resource Language via Close Relatives](https://aclanthology.org/2023.nodalida-1.74.pdf#:~:text=Relatives%20aclanthology,NER%20%28Icelandic%29%2C%20NorNE%20%28Norwegian%29%2C%20and)). The Faroese NER dataset (often called **FoNE**) can be used to test language understanding at the token level. They also mention a Faroese STS (Semantic Textual Similarity) dataset and a new web corpus for Faroese ([Daily Papers - Hugging Face](https://huggingface.co/papers?q=Faroese#:~:text=ignored,STS%29%2C%20and)). These were part of a study on cross-lingual transfer via close Scandinavian relatives ([Daily Papers - Hugging Face](https://huggingface.co/papers?q=Faroese#:~:text=ignored,STS%29%2C%20and)). For instance, they likely released a Faroese STS and maybe a small QA or translation dataset.
- **EuroParl and JW300:** There are parallel corpora involving Faroese. The **ITU Faroese Pairs** dataset ([Daily Papers - Hugging Face](https://huggingface.co/papers?q=Faroese#:~:text=The%20ITU%20Faroese%20Pairs%20Dataset)) contains Faroese-Danish sentence pairs (useful for translation tasks or bilingual training). **JW300 Faroese** (Jehovah’s Witnesses publications) and the **Bible** (already used in training some models ([](https://aclanthology.org/2023.nodalida-1.74.pdf#:~:text=V%C3%A9steinn%20Sn%C3%A6bjarnarson%2C%20Haukur%20Barri%20S%C3%ADmonarson%2C,%C2%B4)) ([](https://aclanthology.org/2023.nodalida-1.74.pdf#:~:text=transfer,based%20adaptation.%20In))) are also available parallel texts. While these are not evaluation benchmarks, they provide training data.
- **Flores-200:** Meta’s FLORES-200 dataset includes Faroese for machine translation (Faeroese-English and vice versa). It has a test set of 1012 sentences for Faroese<->English translation. It’s useful to evaluate if the model can do translation or comprehend content bi-directionally. The GPT-SW3 paper used FLORES200 Faroese to English as a test of zero-shot translation, with GPT-SW3 vs GPT-3.5/GPT-4 ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=evaluate%20GPT,performance%20with%20respect%20to%20Open)).
- **EuroEval Benchmark:** The *EuroEval* project is a comprehensive European language model benchmark suite ([EuroEval/EuroEval: The robust European language model benchmark.](https://github.com/EuroEval/EuroEval#:~:text=benchmark,creating%20an%20account%20on%20GitHub)). It explicitly includes Faroese as one target language (as indicated by adding FoQA as the Faroese reading comprehension benchmark, per their changelog ([EuroEval/CHANGELOG.md at main - GitHub](https://github.com/ScandEval/ScandEval/blob/main/CHANGELOG.md#:~:text=EuroEval%2FCHANGELOG.md%20at%20main%20,Now))). EuroEval likely tests multiple tasks (QA, NER, translation, maybe summarization) across European languages, especially focusing on low-resource ones. It would allow us to compare our distilled model against others on Faroese tasks in a standardized way. For example, we might get an overall score of how well the model does in Faroese QA, chat, etc., compared to say GPT-4 or other open models.
- **Human Evaluation:** For a truly thorough evaluation, one might involve Faroese speakers to rate the model’s output for correctness and fluency. Given the small community, this might be feasible on a smaller scale (for critical applications like chatbots or educational tools in Faroese).

**Existing Faroese Models:** Apart from the general ones we discussed:
- **FoBERT** (110M) and **ScandiBERT** (multi-lingual Nordic BERT, ~270M) were mentioned in FoQA paper ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=FoBERT,5)). FoBERT is a monolingual Faroese BERT model (good for understanding tasks, not generation). ScandiBERT was trained on Scandinavian languages excluding Faroese (“no-fo”) and another version presumably including Faroese. These are useful for building things like a Faroese spell-checker or NER, but not for generative tasks.
- **GPT-SW3** by AI Sweden is actually a series: 6.7B, 20B, and a larger one (~40B if I recall). They explicitly aimed at Nordic languages (Swedish primarily, but also Danish, Norwegian, Icelandic; Faroese was left out of training data likely due to scarcity, but it can leverage similarity) ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=OpenAI%E2%80%99s%20GPT,score%20considering%20translation%20quality%20and)). There’s also an instruction-tuned version of GPT-SW3 (not sure if released). GPT-SW3’s performance on tricky Faroese sentences was qualitatively compared with GPT-3.5 and GPT-4 ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=semantic%20similarity%20score%20%28SBERT%29,pipeline%20thus%20created%20and%20use)) – they found that a Nordic-focused model can handle some linguistic nuances better than even GPT-4 at times (particularly “trick sentences” in Faroese) ([Evaluating the Potential of Language-family-specific Generative Models for Low-resource Data Augmentation: A Faroese Case Study](https://aclanthology.org/2024.lrec-main.576.pdf#:~:text=performance%20with%20respect%20to%20Open,Introduction)). This underscores the value of having a model specialized to the language – reinforcing our strategy.
- **Meta’s wikitext-based models:** There might not be a dedicated Faroese GPT, but given that Faroese has a Wikipedia with ~20K articles, one could train a 1.3B model just on Faroese Wikipedia and texts. However, that alone yields a weak model on complex tasks (similar to GPT-2 level). That’s why our distillation from a large model is key.

In summary, for evaluation we will use:
- **Direct Knowledge Tasks:** FoQA (QA), possibly a set of Faroese trivia or fact questions.
- **Reasoning Tasks:** We could translate some common reasoning benchmarks (math word problems, logic puzzles) into Faroese to see if the distilled student can solve them. There is mention of *“Natural Questions in Icelandic”* and efforts to create such data for Faroese ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=Early%20question%20answering%20datasets%20like,not%20specifically%20measure%20performance%20differences)) – not sure if available, but we could similarly create a small set for testing reasoning.
- **Language Tasks:** Faroese NER, or even just check perplexity on Faroese text vs teacher’s perplexity (the student should have lower perplexity on Faroese than the teacher does, indicating better language model fit – GPT-4 might have higher perplexity on Faroese than our student will).
- **EuroEval:** If available, use its leaderboard to place our model among others for Faroese.

By covering these, we ensure the distilled model is evaluated for **both components**: *language fidelity* (is the Faroese output correct and fluent?) and *task performance* (does it carry over the teacher’s capabilities?). A key sign of success would be if our 7B model outperforms all previous <=7B models on Faroese tasks by a wide margin, approaching the performance of models 10x its size. That would confirm the value of the hybrid GRPO-distillation approach.

----

**References:**

1. Shao et al., *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"* (2024) – introduced GRPO. *[See especially:]* GRPO uses no critic, baselines on group scores ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=modifies%20the%20traditional%20Proximal%20Policy,to%20improve%20models%20on%20helpfulness)) ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=3,KL%20term%20within%20the%20reward)), enabling efficient RL training ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=Traditional%20RL%20methods%20like%20Proximal,to%20reasoning%20tasks%20in%20LLMs)) ([The Math Behind DeepSeek: A Deep Dive into Group Relative Policy Optimization (GRPO) | by Sahin Ahmed, Data Scientist | Medium](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba#:~:text=%2A%20Critic,to%20assess%20how%20well%20a%E2%80%A6)).
2. Gu et al., *"MiniLLM: Knowledge Distillation of Large Language Models"* (2023) – proposed reverse KL distillation. *[Notably:]* Reverse KL makes student focus on teacher’s major modes, avoiding long-tail errors ([](https://arxiv.org/pdf/2306.08543#:~:text=,To%20further%20stabilize%20and%20accelerate)). Added pre-training loss $L_{PT}$ to preserve performance on NLP benchmarks ([](https://arxiv.org/pdf/2306.08543#:~:text=include%20a%20language%20modeling%20loss,using%20a%20combination%20of%20gradients)).
3. Anil et al., *"Gemma 3 Technical Report"* (2025) – detailed Gemma 3’s training. *[Highlights:]* Gemma3 models trained with distillation from a larger teacher and RL fine-tuning (BOND, WARM, WARP) ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=Distillation,sampled%20logits%2C%20and%20renormalized)) ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=knowledge%20distillation%C2%A0%28Hinton%20et%C2%A0al,2024a)). Distillation implemented via sampling top 256 logits from teacher for student’s cross-entropy ([Gemma 3 Technical Report](https://arxiv.org/html/2503.19786v1#:~:text=Distillation,sampled%20logits%2C%20and%20renormalized)).
4. Philschmid (2025), *“How DeepSeek R1 was trained”* – blog summary. *[Used for:]* Clear explanation of GRPO steps ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=1,This%20is%20different)) and multi-stage training (SFT→RL) to fix issues ([Bite: How Deepseek R1 was trained](https://www.philschmid.de/deepseek-r1#:~:text=Image%3A%20r1)).
5. FoQA: Simonsen et al. (2025), *“FoQA: A Faroese Question-Answering Dataset”*. *[Data points:]* First Faroese QA dataset, used GPT-4 to generate questions ([Paper page - FoQA: A Faroese Question-Answering Dataset](https://huggingface.co/papers/2502.07642#:~:text=We%20present%20FoQA%2C%20a%20Faroese,released%20in%20three%20versions%3A%20a)). Evaluation showed GPT-4 and LLaMA 8B ~75 F1, Mistral7B ~62, FoBERT ~36 ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=GPT,5)) ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=FoBERT,5)), and that big decoder models far outperform Faroese-specialized BERT on QA ([[2502.07642] FoQA: A Faroese Question-Answering Dataset](https://ar5iv.org/pdf/2502.07642#:~:text=We%20found%20that%20encoder%20models,A)).
6. Faroese NLP resources: Snæbjarnarson et al. (2023), *“Transfer to a Low-Resource Language via Close Relatives: Faroese”*. *[Relevance:]* Released Faroese NER (FoNE) ([[PDF] Transfer to a Low-Resource Language via Close Relatives](https://aclanthology.org/2023.nodalida-1.74.pdf#:~:text=Relatives%20aclanthology,NER%20%28Icelandic%29%2C%20NorNE%20%28Norwegian%29%2C%20and)) and showed leveraging Scandinavian corpora improves Faroese models ([Daily Papers - Hugging Face](https://huggingface.co/papers?q=Faroese#:~:text=ignored,STS%29%2C%20and)). 

