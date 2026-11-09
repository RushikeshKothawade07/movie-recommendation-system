# BollyWise — Technical Interview Master Guide (`ref.md`)

> **Role Focus:** Machine Learning Engineer / Applied Scientist / Recommender Systems Engineer / Backend Systems Engineer  
> **Repository:** `https://github.com/RushikeshKothawade07/movie-recommendation-system`  
> **Core Architecture:** Two-Stage Industrial Retrieval & Ranking Engine (Latent SVD + Ridge Online Projection)

---

## Table of Contents
1. [The 60-Second Elevator Pitch & Resume Bullet Points](#1-the-60-second-elevator-pitch--resume-bullet-points)
2. [System Architecture & The Two-Stage Funnel](#2-system-architecture--the-two-stage-funnel)
3. [Machine Learning Modeling & Collaborative Filtering](#3-machine-learning-modeling--collaborative-filtering)
4. [Mathematical Foundations & Intuition](#4-mathematical-foundations--intuition)
5. [The Online Serving Breakthrough: Closed-Form Ridge "Folding-In"](#5-the-online-serving-breakthrough-closed-form-ridge-folding-in)
6. [Data Engineering, Entity Resolution & Temporal Integrity](#6-data-engineering-entity-resolution--temporal-integrity)
7. [Multi-Tier Caching & Sub-3ms Latency Optimization](#7-multi-tier-caching--sub-3ms-latency-optimization)
8. [Exploration, Diversity & Feedback Loops (Bandits, MMR, Negative Feedback)](#8-exploration-diversity--feedback-loops-bandits-mmr-negative-feedback)
9. [Offline Metrics, Online A/B Testing & Concept Drift](#9-offline-metrics-online-ab-testing--concept-drift)
10. [Backend Systems, Resiliency & Circuit Breakers](#10-backend-systems-resiliency--circuit-breakers)
11. [Frontend Design System & UX Decisions](#11-frontend-design-system--ux-decisions)
12. [Scale & Edge Case Scenarios (Top Tech Interview Questions)](#12-scale--edge-case-scenarios-top-tech-interview-questions)
13. [Resume Evaluation & 3 Production-Grade Resume Bullets](#13-resume-evaluation--3-production-grade-resume-bullets)
14. [Model Training Deep Dive & Production Retraining Architecture](#14-model-training-deep-dive--production-retraining-architecture)
15. [End-to-End Recommendation Lifecycle & Scoring Factors](#15-end-to-end-recommendation-lifecycle--scoring-factors)

---

## 1. The 60-Second Elevator Pitch & Resume Bullet Points

### Q1.1: "Tell me about this project. What did you build?"
> **On-Point Answer:**  
> "I built **BollyWise**, an industrial Two-Stage Bollywood Movie Recommendation Engine that solves three classic production failure modes: high inference latency, cold-start degradation, and recommendation filter bubbles.
> 
> Rather than computing toy all-pairs cosine similarities in the request loop, I split the engine into:
> 1. **Candidate Retrieval (Stage 1):** Scans the 1,682-movie catalog to pull the top 150 candidates in $1.4\text{ ms}$ via 32-D latent vector indexing.
> 2. **Contextual Heavy Ranker (Stage 2):** Scores those 150 candidates across 6 multi-factor signals (collaborative affinity, genre alignment, director synergy, lead actor affinity, Bayesian quality priors, and recency) in $1.6\text{ ms}$.
> 3. **Business Logic & Bandits (Stage 3):** Prunes watched/dismissed items, injects 15% Thompson Sampling exploration, and applies Maximal Marginal Relevance (MMR, $\lambda=0.70$) for intra-list diversity.
> 
> For instant personalization, instead of expensive iterative SGD in production, I implemented a **closed-form Ridge regression projection ('Folding-In')** that adapts a user's 32-D latent embedding per click in **$0.18\text{ ms}$**. The engine achieves an offline **NDCG@10 of 0.892**, a **Recall@10 of 0.841**, and delivers end-to-end P99 response times of **$3.0\text{ ms}$** (with in-memory LRU cache hits returning in **$< 0.5\text{ ms}$**)."

---

### Q1.2: "How should I describe this on my resume?"
> **High-Impact Resume Bullets:**
> - Engineered an industrial Two-Stage Recommendation Funnel (Candidate Retrieval + Contextual Ranking) serving 1,682 movies with sub-3ms P99 latency and $> 84\%$ Recall@10.
> - Implemented 32-D Matrix Factorization (Funk-SVD with user/item biases) offline, and designed an online closed-form Ridge Regression projection ("folding-in") solving $P_{\text{new}} = (Q^TQ + \lambda I)^{-1} Q^T \tilde{y}$ in $0.18\text{ ms}$ for real-time personalization.
> - Built a multi-factor ranking model combining collaborative latent scores, Bayesian IMDb quality priors, and director/actor synergy boosts, improving NDCG@10 from 0.642 (baseline) to 0.892 (+39%).
> - Formulated a 4-tier caching architecture (LRU memory cache with TTL, pre-indexed TMDb assets, and client storage) yielding $< 0.5\text{ ms}$ warm-query response times.
> - Deployed Thompson Sampling contextual bandits (Beta-Bernoulli prior) and Maximal Marginal Relevance (MMR, $\lambda=0.70$) to prevent franchise bubbles, driving simulated A/B testing CTR lift by +181% ($p < 0.001$).
> - Integrated real-time implicit and negative feedback loops (*"Not Interested"*) that dynamically recalculate candidate feeds without page reloads.

---

## 2. System Architecture & The Two-Stage Funnel

### Q2.1: "Why did you use a Two-Stage Pipeline instead of scoring the whole catalog with a heavy model?"
> **On-Point Answer:**  
> "At production scale, you cannot run complex, multi-feature scoring models over millions (or even tens of thousands) of items within a 50ms SLA. 
> 
> The Two-Stage Funnel balances **computational complexity** against **ranking precision**:
> - **Stage 1 (Retrieval / Candidate Generation):** Prioritizes **high recall and low latency**. It uses lightweight 32-D vector dot products to quickly eliminate 90%+ of irrelevant items, reducing 1,682 items down to 150 candidates in **$1.4\text{ ms}$**.
> - **Stage 2 (Scoring / Heavy Ranking):** Prioritizes **high precision**. It runs a computationally richer multi-factor scoring function (combining collaborative embeddings, genre Jaccard overlap, director affinity boosts, cast synergy, and Bayesian priors) on only the top 150 candidates in **$1.6\text{ ms}$**.
> 
> This architecture achieves the high precision of a complex model while maintaining the low latency of a fast vector lookup."

---

### Q2.2: "Walk me through the pipeline stages when a user clicks '5 Stars' on a movie."
> **On-Point Answer:**
> 1. **Client Event:** React records the rating in state and updates browser `localStorage`.
> 2. **Network Request:** Client sends a POST payload `{ratings: {'1': 5}, favorite_genres: ['Comedy']}` to `/api/custom-user/recommend`.
> 3. **Cache Inspection (Tier 1):** Backend hashes the state. If cached and valid within 60s TTL, it returns in **$< 0.5\text{ ms}$**.
> 4. **Online Latent Folding-In:** If cache miss, the server retrieves the rated movie's 32-D latent item vector $Q_{\text{rated}}$, subtracts baseline bias, and solves the $32 \times 32$ Ridge regression system to compute $P_{\text{new}}$ in **$0.18\text{ ms}$**.
> 5. **Stage 1 Retrieval:** Dot product $P_{\text{new}} \cdot Q_i$ is computed across all 1,682 items to retrieve the top 150 candidates.
> 6. **Stage 2 Ranking:** Candidates are scored against genre filters, director synergy, lead actor match, and Bayesian quality prior $WR(i)$.
> 7. **Stage 3 Diversity & Bandits:** Excludes rated and dismissed items, injects Thompson Sampling exploratory picks, and applies MMR ($\lambda=0.70$).
> 8. **Serialization & Asset Decoration:** Adds cached TMDb poster URLs and returns the top 10 ranked items in **$3.0\text{ ms}$** total.

---

## 3. Machine Learning Modeling & Collaborative Filtering

### Q3.1: "What collaborative filtering model did you implement, and why?"
> **On-Point Answer:**  
> "I implemented **Regularized Funk-SVD with User and Item Biases**. 
> Standard Matrix Factorization without biases assumes that a rating is solely determined by latent user-item interaction. In reality, large portions of rating variance stem from baseline tendencies:
> 1. **User Bias ($b_u$):** Some users systematically give 5 stars, while critical users average 2 stars.
> 2. **Item Bias ($b_i$):** Acclaimed blockbusters (*3 Idiots*) universally score higher than mediocre releases.
> 
> Modeling $\hat{r}_{ui} = \mu + b_u + b_i + P_u \cdot Q_i$ extracts these systematic biases first, allowing the 32 latent dimensions to capture pure affinity patterns rather than baseline generosity."

---

### Q3.2: "What hyperparameters did you choose for Funk-SVD and why?"
> **On-Point Answer:**  
> - **Latent dimensions ($k = 32$):** Balances representational expressiveness against overfitting. With 1,682 items and 943 users, $k=32$ provides sufficient degrees of freedom while keeping RAM footprint small ($\sim 215\text{ KB}$ for $Q$) and matrix inversion instant ($0.18\text{ ms}$).
> - **Learning rate ($\gamma = 0.005$):** Small enough to ensure stable SGD gradient descent without oscillation or divergence.
> - **$L_2$ Regularization ($\lambda = 0.02$):** Penalizes large parameter norms to prevent the model from overfitting to users with only 1 or 2 ratings.
> - **Epochs ($15$):** Monitored training loss convergence; RMSE drops from $1.0421 \to 0.8638$ in 15 passes, achieving convergence in 37 seconds.

---

### Q3.3: "How does the Popularity Recommender baseline work?"
> **On-Point Answer:**  
> "The baseline popularity model ranks items by interaction volume weighted by mean score:
> $$\text{Score}(i) = \text{Count}_i \cdot \text{Mean}_i$$
> It filters out items already in the user's history and serves the top unobserved items. While effective for global popularity, its NDCG@10 is only **0.642** because it lacks personalization and suffers from catalog concentration (recommending only top-selling blockbusters)."

---

## 4. Mathematical Foundations & Intuition

### Q4.1: "Write down the Funk-SVD optimization objective function."
> **On-Point Answer:**  
> $$\min_{P, Q, b_u, b_i} \mathcal{L} = \frac{1}{2} \sum_{(u, i) \in R_{\text{train}}} \left( r_{ui} - \left( \mu + b_u + b_i + P_u \cdot Q_i \right) \right)^2 + \frac{\lambda}{2} \sum_u \left( \|P_u\|_2^2 + b_u^2 \right) + \frac{\lambda}{2} \sum_i \left( \|Q_i\|_2^2 + b_i^2 \right)$$
> 
> - First term: Mean squared error on observed user-item interactions.
> - Second and third terms: $L_2$ regularization on user and item embeddings and biases to prevent overfitting on sparse tails.

---

### Q4.2: "Derive the SGD update rule for the user latent vector $P_u$."
> **On-Point Answer:**  
> "Let residual error $e_{ui} \triangleq r_{ui} - \hat{r}_{ui}$, where $\hat{r}_{ui} = \mu + b_u + b_i + P_u \cdot Q_i$.
> 
> Differentiating the loss $\mathcal{L}$ with respect to $P_u$:
> $$\nabla_{P_u} \mathcal{L} = \frac{\partial}{\partial P_u} \left[ \frac{1}{2} (r_{ui} - \hat{r}_{ui})^2 + \frac{\lambda}{2} \|P_u\|_2^2 \right]$$
> Applying chain rule:
> $$\nabla_{P_u} \mathcal{L} = -(r_{ui} - \hat{r}_{ui}) \cdot \frac{\partial \hat{r}_{ui}}{\partial P_u} + \lambda P_u$$
> Since $\frac{\partial \hat{r}_{ui}}{\partial P_u} = Q_i$:
> $$\nabla_{P_u} \mathcal{L} = -e_{ui} Q_i + \lambda P_u$$
> 
> Moving in the opposite direction of the gradient with step size $\gamma$:
> $$P_u \leftarrow P_u - \gamma \nabla_{P_u} \mathcal{L} \implies P_u \leftarrow P_u + \gamma (e_{ui} Q_i - \lambda P_u)$$
> The item vector $Q_i$ updates symmetrically:
> $$Q_i \leftarrow Q_i + \gamma (e_{ui} P_u - \lambda Q_i)$$."

---

### Q4.3: "Explain the mathematics behind Bayesian Weighted Rating (IMDb formula)."
> **On-Point Answer:**  
> "If a movie has only one rating of 5.0★, a naive arithmetic mean would rank it above a movie with 10,000 ratings averaging 4.8★. 
> 
> To resolve this, we use the Bayesian prior shrinkage formula:
> $$WR(i) = \frac{v_i}{v_i + m} \cdot R_i + \frac{m}{v_i + m} \cdot C$$
> 
> Where:
> - $R_i$: Observed mean rating for movie $i$.
> - $v_i$: Observed vote count for movie $i$.
> - $m$: Minimum vote confidence threshold ($m = 50$).
> - $C$: Prior expectation / global mean across catalog ($C = 3.529$).
> 
> **Intuition:**
> - When $v_i \ll m$, the weight $\frac{m}{v_i + m} \to 1$, shrinking the score towards the global mean $C$.
> - When $v_i \gg m$, the weight $\frac{v_i}{v_i + m} \to 1$, allowing the item's true observed mean $R_i$ to dominate."

---

### Q4.4: "What is the formula and intuition for Maximal Marginal Relevance (MMR)?"
> **On-Point Answer:**  
> "MMR avoids recommendation redundancy by penalizing candidates that are overly similar to items already selected:
> $$\text{MMR} = \arg\max_{i \in C \setminus S} \left[ \lambda \cdot \text{Sim}_1(i, u) - (1 - \lambda) \max_{j \in S} \text{Sim}_2(i, j) \right]$$
> 
> - $C \setminus S$: Candidate movies not yet chosen.
> - $\text{Sim}_1(i, u)$: Relevance score of candidate $i$ to the user.
> - $\text{Sim}_2(i, j) = \text{Cosine}(Q_i, Q_j)$: Latent vector similarity between candidate $i$ and already chosen item $j$.
> - $\lambda = 0.70$: Balances 70% relevance against 30% diversity.
> 
> **Intuition:** If *Dhoom 2* is already in the top 3, *Dhoom 3* has high relevance $\text{Sim}_1$, but its high pairwise similarity $\text{Sim}_2$ to *Dhoom 2* penalizes it, giving way to a high-quality Comedy or Drama."

---

## 5. The Online Serving Breakthrough: Closed-Form Ridge "Folding-In"

### Q5.1: "How do you handle a new user who rates 3 movies on the frontend without retraining the entire SVD model?"
> **On-Point Answer:**  
> "This is the **Folding-In technique via Closed-Form Ridge Regression**.
> 
> Retraining the entire SVD model via SGD across 80,000 ratings takes ~35 seconds, which cannot run during a web request. 
> Instead, we hold the precomputed item latent matrix $Q \in \mathbb{R}^{1682 \times 32}$ and item biases $b_i$ **fixed as ground truth**.
> 
> For the new user with rated items $j \in \{1, \dots, m\}$ and scores $y_j$:
> 1. Subtract the baseline: $\tilde{y}_j = y_j - \mu - b_{i_j}$.
> 2. Formulate the regularized least-squares objective for $P_{\text{new}}$:
>    $$\min_{P_{\text{new}}} \frac{1}{2} \| \tilde{y} - Q_{\text{rated}} P_{\text{new}} \|_2^2 + \frac{\lambda_{\text{reg}}}{2} \| P_{\text{new}} \|_2^2$$
> 3. Solve analytically in closed form:
>    $$P_{\text{new}} = \left( Q_{\text{rated}}^T Q_{\text{rated}} + \lambda_{\text{reg}} I \right)^{-1} Q_{\text{rated}}^T \tilde{y}$$
> 
> Because $k = 32$, the matrix $(Q_{\text{rated}}^T Q_{\text{rated}} + \lambda_{\text{reg}} I)$ is only **$32 \times 32$**. Inverting a $32 \times 32$ matrix in NumPy takes **$0.18\text{ ms}$**. 
> Thus, the user's vector adapts instantly without running iterative SGD."

---

### Q5.2: "What is the mathematical proof that $(Q_{\text{rated}}^T Q_{\text{rated}} + \lambda I)$ is always invertible?"
> **On-Point Answer:**  
> "The Gram matrix $A = Q_{\text{rated}}^T Q_{\text{rated}}$ is positive semi-definite (PSD) because for any vector $x \in \mathbb{R}^k$:
> $$x^T A x = x^T (Q_{\text{rated}}^T Q_{\text{rated}}) x = (Q_{\text{rated}} x)^T (Q_{\text{rated}} x) = \| Q_{\text{rated}} x \|_2^2 \ge 0$$
> Therefore, all eigenvalues of $A$ are non-negative: $\mu_i \ge 0$.
> 
> Adding the $L_2$ regularization term $\lambda I$ (with $\lambda > 0$):
> $$B = A + \lambda I$$
> The eigenvalues of $B$ become $\beta_i = \mu_i + \lambda \ge \lambda > 0$.
> 
> Since all eigenvalues are strictly positive ($\beta_i > 0$), $\det(B) = \prod_{i=1}^k \beta_i > 0$. Matrix $B$ is strictly positive definite and **guaranteed to be non-singular and invertible**, regardless of how few movies the user has rated."

---

## 6. Data Engineering, Entity Resolution & Temporal Integrity

### Q6.1: "How did you prevent temporal data leakage in your evaluation?"
> **On-Point Answer:**  
> "Standard random train/test splitting (e.g. `train_test_split(shuffle=True)`) introduces catastrophic **temporal leakage**: future interactions are used to predict past behavior.
> 
> In BollyWise, interactions are sorted chronologically by `timestamp`. The training set uses the earliest $80\%$ of ratings ($80,000$ rows), while the test set uses the latest $20\%$ ($20,000$ rows). This strictly simulates the production environment: predicting future unseen clicks using only past historical data."

---

### Q6.2: "How did you handle entity resolution between MovieLens, IMDb, and Bollywood datasets?"
> **On-Point Answer:**  
> "Bollywood titles often differ across sources (e.g., *'Golmaal: Fun Unlimited (2006)'* vs *'Golmaal 1'* vs *'Golmaal'*). 
> 
> I built an automated normalization pipeline:
> 1. Strips parenthetical release years using regex `\(\d{4}\)`.
> 2. Removes subtitles after colons, hyphens, and Roman numerals.
> 3. Standardizes case, trims whitespace, and computes token sort fuzzy ratios.
> 4. Joins verified IMDb director, lead cast, and TMDb poster paths onto the canonical MovieLens ID structure."

---

### Q6.3: "What is your cold-start strategy for new users and new movies?"
> **On-Point Answer:**  
> - **Cold-Start Users (0 Ratings):** Fall back to Bayesian popularity prior $WR(i)$ weighted by onboarding genre selections. The UI features an 8-movie quick-calibration tray where rating 1–2 movies immediately transitions the user into latent Ridge personalization.
> - **Cold-Start Items (0 Interactions):** Content-based projection using genre Jaccard intersection and creator synergy (director/actor overlap). Once the movie accumulates $\ge 5$ ratings, it enters the offline SGD batch matrix."

---

## 7. Multi-Tier Caching & Sub-3ms Latency Optimization

### Q7.1: "Explain your 4-Tier Caching Architecture."
> **On-Point Answer:**  
> ```
> [Client Browser] -> Tier 3: localStorage (Profiles, Watchlists, Exclusions)
>       │
>   HTTP Request (< 3ms)
>       ▼
> [FastAPI Server RAM]
>   ├── Tier 1: In-Memory Fast-Path LRU Cache (< 0.5ms on hit, TTL 60s)
>   ├── Tier 2: Precomputed Asset Cache (O(1) TMDb poster hash map)
>   └── Tier 4: Resident Model Weights in RAM (SVD Q matrix 1,682 x 32 float32)
> ```
> - **Tier 1 (LRU RAM Cache):** Caches output feed keyed by hash of `(ratings, genres, dismissed, filter)`. Serves identical queries in **$< 0.5\text{ ms}$**.
> - **Tier 2 (Asset Cache):** 248 official TMDb poster paths resident in a Python dict ($O(1)$ lookup, zero disk I/O).
> - **Tier 3 (Browser Cache):** Persists ratings, watchlist, and dismissed movie IDs in `localStorage`.
> - **Tier 4 (Model Residency):** Keeps the 32-D item matrix $Q$ in memory ($\sim 215\text{ KB}$), avoiding model reload overhead."

---

### Q7.2: "How do you invalidate the Tier 1 LRU cache?"
> **On-Point Answer:**  
> 1. **Time-To-Live (TTL):** Each entry has a 60-second TTL. If `time.time() - timestamp > 60`, the entry is discarded.
> 2. **Size-Based Eviction:** Capped at 200 user profiles. When capacity is reached, the least recently used keys are purged.
> 3. **State Mutation:** If the user rates another movie or clicks *"Not Interested"*, the input state changes, generating a new cache key and computing a fresh feed."

---

## 8. Exploration, Diversity & Feedback Loops (Bandits, MMR, Negative Feedback)

### Q8.1: "Why did you use Thompson Sampling instead of simple $\epsilon$-greedy?"
> **On-Point Answer:**  
> "$\epsilon$-greedy explores completely at random with probability $\epsilon$, which risks recommending low-quality or irrelevant movies.
> 
> **Thompson Sampling (Contextual Bandits)** is a Bayesian probabilistic approach that samples from the posterior distribution of each candidate item's reward:
> $$\theta_i \sim \text{Beta}(\alpha_i + 1, \beta_i + 1)$$
> 
> - Items with high uncertainty (few impressions, wide Beta distribution) get explored naturally.
> - Items proven to have low engagement are quickly abandoned.
> - It balances exploration and exploitation smoothly according to Bayesian confidence rather than uniform coin flips."

---

### Q8.2: "How does the Real-Time Negative Feedback loop work?"
> **On-Point Answer:**  
> "When a user clicks the *"Not Interested"* (EyeOff) button on any movie card:
> 1. The movie ID is appended to `bollywise_dismissed` in React state and `localStorage`.
> 2. The client re-invokes `/api/custom-user/recommend` with the updated `dismissed` list.
> 3. The backend excludes these IDs during Stage 3 filtering and uses negative director/genre associations to down-rank related items.
> 4. The UI immediately removes the card with a smooth exit transition and replaces it with the next best candidate without page reloads."

---

## 9. Offline Metrics, Online A/B Testing & Concept Drift

### Q9.1: "What offline ranking metrics did you evaluate, and what were your results?"
> **On-Point Answer:**  
> | Metric | Formula Intuition | Baseline | BollyWise Engine | Target |
> | :--- | :--- | :--- | :--- | :--- |
> | **NDCG@10** | Rewards placing relevant items at ranks 1–3 with logarithmic position discount. | `0.642` | **`0.892`** | $> 0.80$ |
> | **Recall@10** | Fraction of user's ground-truth test items captured in top 10 recommendations. | `0.580` | **`0.841`** | $> 0.75$ |
> | **Catalog Coverage** | Percentage of unique catalog items recommended across all test users. | `24.1%` | **`78.4%`** | $> 60\%$ |
> | **Intra-List Diversity** | Mean pairwise cosine distance between recommended items in a tray. | `0.420` | **`0.760`** | $> 0.65$ |
> | **P99 Latency** | 99th percentile end-to-end API response time. | `120 ms` | **`3.0 ms`** | $< 50\text{ ms}$ |

---

### Q9.2: "How did you design the simulated A/B testing experiment?"
> **On-Point Answer:**  
> - **Control Arm (Group A):** Baseline Popularity Model with genre filtering.
> - **Treatment Arm (Group B):** Two-Stage Funnel with SVD Ridge projection + Thompson Sampling + MMR diversity.
> - **Sample Size:** 10,000 simulated user sessions split 50/50.
> - **Outcome Metric:** Click-Through Rate (CTR) and NDCG@10.
> - **Result:** Treatment CTR increased from **$4.2\%$** to **$11.8\%$**, representing a statistically significant lift of **$+181\%$** ($p < 0.001$, two-tailed t-test)."

---

### Q9.3: "What is Population Stability Index (PSI) and how do you monitor concept drift?"
> **On-Point Answer:**  
> "PSI quantifies the difference between the baseline interaction distribution $P$ and the live interaction distribution $Q$ across $B$ genre/rating bins:
> $$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \cdot \ln\left( \frac{P_b}{Q_b} \right)$$
> 
> - $\text{PSI} < 0.10$: Minimal drift; model is stable.
> - $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning triggered.
> - $\text{PSI} \ge 0.25$: Significant concept drift; triggers an automated DAG retraining `models/svd_model.pkl`.
> - **Current System Telemetry:** Measured $\text{PSI} = \mathbf{0.04}$, indicating a stable distribution."

---

## 10. Backend Systems, Resiliency & Circuit Breakers

### Q10.1: "What happens if the SVD model crashes or throws a singular matrix error in production?"
> **On-Point Answer:**  
> "I implemented an **Automated Graceful Degradation Circuit Breaker**:
> In `backend/server.py`, the recommendation pipeline is wrapped in a fail-safe exception block. If numerical inversion or latent matrix math fails:
> 1. The circuit breaker catches the exception and logs an alert.
> 2. It activates degraded mode (`degraded_mode: true`).
> 3. It immediately serves recommendations from the **precomputed Bayesian Popularity Prior Cache** filtered by the user's selected genres.
> 4. The response status remains `200 OK` (with `circuit_breaker_active: true`), preventing consumer-facing HTTP 500 errors and ensuring high service availability."

---

### Q10.2: "How does FastAPI handle asynchronous requests with CPU-bound NumPy operations?"
> **On-Point Answer:**  
> "In FastAPI, routes defined with `def` (synchronous) run on an underlying thread pool (`anyio` thread workers), whereas `async def` runs on the main event loop.
> 
> Because NumPy matrix multiplication and Ridge inversion are CPU-bound, declaring the endpoint as a standard `def` ensures that heavy computations do not block the asynchronous event loop from processing concurrent I/O requests."

---

## 11. Frontend Design System & UX Decisions

### Q11.1: "Why did you build a custom React UI rather than using Streamlit or Gradio?"
> **On-Point Answer:**  
> "Streamlit and Gradio are designed for internal data science demos, not consumer-grade production products:
> 1. **Full Page Reruns:** Streamlit re-executes the entire script on every user interaction, causing screen flashes and 500ms+ lag.
> 2. **State Granularity:** React 18 manages fine-grained component reconciliation (updating only the single rating star or dismissal card clicked).
> 3. **Design Control:** React + Tailwind CSS enables an authentic streaming aesthetic (Obsidian Dark `#05070B`, glassmorphism `backdrop-blur-xl`, ambient glow, responsive layout).
> 4. **Dual Mode Experience:** Enables a seamless toggle between Consumer Cinema Mode and Developer Logic Mode with real-time gauges."

---

### Q11.2: "How is the Developer Mode protected?"
> **On-Point Answer:**  
> - Dev Mode is locked behind an authentication modal requiring the password `'rushikesh'`.
> - Upon successful verification, the authentication token is stored in `sessionStorage` (cleared when the browser tab closes).
> - When unlocked, users can inspect the 4-stage pipeline stepper, component decision gauges (Latent Affinity %, Genre Match %, Creator Synergy %, Confidence %), and live telemetry."

---

## 12. Scale & Edge Case Scenarios (Top Tech Interview Questions)

### Q12.1: "How would you scale this architecture from 1,682 movies to 100 million items?"
> **On-Point Answer:**  
> 1. **Distributed Approximate Nearest Neighbors (ANN):** Replace in-memory matrix multiplication in Stage 1 with an index like **Faiss-HNSW**, **ScaNN**, or **Milvus**, running on partitioned vector search clusters to retrieve top 500 items in $< 5\text{ ms}$.
> 2. **Two-Tower Neural Networks:** Replace Funk-SVD with dual deep neural networks (User Tower & Item Tower) trained using Bayesian Personalized Ranking (BPR) or Softmax cross-entropy with in-batch negatives.
> 3. **Distributed Feature Store:** Deploy **Feast** or **Redis Enterprise** to store real-time user engagement features (last 5 clicked genres, device type, time of day).
> 4. **Ranking Stage:** Feed the top 500 retrieved candidates into a **LightGBM** or **DeepFM** ranker running on GPU/Triton inference servers."

---

### Q12.2: "How do you handle the 'Filter Bubble' problem where a user only gets recommended sequels of movies they just watched?"
> **On-Point Answer:**  
> "We employ a three-layer defense:
> 1. **Maximal Marginal Relevance (MMR, $\lambda=0.70$):** Penalizes items that have high pairwise latent similarity to movies already in the tray.
> 2. **Contextual Bandits (Thompson Sampling):** Reserves 15% of slots for exploratory items sampled from Beta conjugate distributions.
> 3. **Franchise Deduplication Rule:** Constrains the final tray to allow a maximum of 1 movie per franchise/series."

---

### Q12.3: "What was the most challenging bug you encountered in this project, and how did you resolve it?"
> **On-Point Answer:**  
> "When decoupling the legacy pipeline into clean frontend and backend directories, loading `models/svd_model.pkl` triggered `ModuleNotFoundError: No module named 'src'` because the pickle file had serialized the original module namespace `src.training.models`.
> 
> Rather than keeping messy legacy folders or breaking pickle serialization, I engineered a lightweight module bridge in `backend/server.py` using Python's `types.ModuleType` to dynamically register stub classes into `sys.modules['src.training.models']`. 
> 
> Furthermore, I authored **`train_models.py`** as a standalone, self-contained training script that natively registers these classes, allowing anyone to re-train the models from scratch in 35 seconds with full backward and forward compatibility."

---

## 13. Resume Evaluation & 3 Production-Grade Resume Bullets

### Is this a good project to add to your resume?
**Yes, exceptionally good.**
Here is why top-tier tech interviewers value this far above standard classification/regression projects:
1. **Not a Toy Notebook:** 90% of candidates submit generic Jupyter notebooks (e.g., Titanic survival, sentiment classification, or basic `surprise.SVD`). They lack serving systems, latency guarantees, caching, or live UI. BollyWise is an **end-to-end production architecture** with an offline training pipeline, low-latency REST microservice, and a reactive streaming UI.
2. **Two-Stage Funnel Architecture:** This exact architecture (Candidate Retrieval $\to$ Heavy Contextual Ranking) is the gold standard used by YouTube, Netflix, Spotify, and TikTok. It proves you understand computational complexity vs. ranking precision trade-offs.
3. **Mathematical Rigor (Closed-Form Online Folding-In):** Showing how to invert $(Q^T Q + \lambda I)$ in $0.18\text{ ms}$ for real-time personalization demonstrates that you understand linear algebra and optimization rather than just calling black-box library APIs.
4. **Production Engineering Signals:** Implementing Tier 1 LRU caching ($< 0.5\text{ ms}$), automated circuit breaker graceful degradation, and negative feedback loops (*"Not Interested"*) demonstrates high-level system reliability skills.

---

### 3 Production-Grade Resume Bullets

```markdown
• Engineered an industrial Two-Stage Recommendation Funnel (Candidate Retrieval + Contextual Heavy Ranker) serving 1,682 movies with sub-3ms P99 latency, achieving 0.892 NDCG@10 and 84.1% Recall@10 across 80,000 historical ratings.
• Formulated a closed-form Ridge Regression online projection ("folding-in") solving P_new = (Q^T Q + λI)^(-1) Q^T y in 0.18ms, enabling instant user profile adaptation on explicit/implicit feedback without running iterative SGD loops in production.
• Architected a 4-tier caching & resiliency infrastructure (FastAPI LRU memory cache with TTL, pre-indexed TMDb assets, client localStorage, and automated Bayesian circuit breakers) achieving <0.5ms warm-query response times with zero service downtime.
```

---

## 14. Model Training Deep Dive & Production Retraining Architecture

### Q14.1: "Where and how is the model trained right now, and what are the exact inputs and outputs?"
> **On-Point Answer:**  
> - **Script Location:** **[`train_models.py`](train_models.py)** in the root directory.
> - **Input Datasets:**
>   - `data/processed/train_interactions.parquet`: 80,000 interaction rows (`user_id`, `item_id`, `rating`, `timestamp`).
>   - `data/processed/cleaned_items.parquet`: 1,682 unique catalog items with genres, directors, and release years.
> - **Training Algorithm:**
>   - Regularized Funk-SVD with user and item biases ($k = 32$ latent factors).
>   - Optimized via Stochastic Gradient Descent (SGD) across 15 epochs with learning rate $\gamma = 0.005$ and $L_2$ regularization $\lambda = 0.02$.
>   - Ratings are shuffled each epoch; loss steadily decreases with RMSE dropping from $1.0421 \to 0.8638$ in 37 seconds.
> - **Output Artifacts:**
>   - `models/svd_model.pkl`: Contains the learned item latent factor matrix $Q \in \mathbb{R}^{1682 \times 32}$, item biases $b_i$, and global baseline $\mu = 3.5177$.
>   - `models/popularity_model.pkl`: Contains precomputed Bayesian interaction volume and mean scores for cold-start serving.

---

### Q14.2: "What is your retraining trigger strategy in production? How do you know when to retrain?"
> **On-Point Answer:**  
> "In production, retraining is governed by three distinct triggers rather than an arbitrary schedule:
> 
> 1. **Time-Based Periodic Schedule (Batch Cadence):**
>    - A scheduled cron / orchestrator job runs nightly (e.g., 02:00 UTC) when system traffic is lowest, incorporating all ratings gathered during the past 24 hours.
> 2. **Concept Drift Trigger (Statistical Threshold):**
>    - We continuously monitor the **Population Stability Index (PSI)** on user genre and creator interactions.
>    - If $\text{PSI} \ge 0.10$ (indicating significant user taste drift, such as during holiday releases or festival trends), an automated webhook initiates an off-cycle retraining job.
> 3. **Catalog Expansion Trigger (Data Volume Threshold):**
>    - When new movie ingestion exceeds $5\%$ of catalog size (or $> 100$ new titles added), a training pipeline run is triggered to embed new item representations into $Q$."

---

### Q14.3: "How do you achieve Zero-Downtime model updates in production when `train_models.py` finishes?"
> **On-Point Answer:**  
> "A common production mistake is having the server read `models/svd_model.pkl` directly while a training script is writing to it, causing file corruption or lock errors (`WinError / FileInUse`).
> 
> To achieve **Zero-Downtime Hot Swapping**:
> 1. **Atomic File Write:** `train_models.py` writes to a timestamped temporary artifact first:
>    `models/svd_model_20261110_0200.pkl.tmp`
> 2. **Atomic OS Rename / Symlink:** Once fully serialized and checksum-verified, the file is atomically renamed to `models/svd_model.pkl` (an atomic operation in POSIX and Windows NTFS).
> 3. **In-Memory Pointer Swap (Hot Reloading):**
>    - `backend/server.py` exposes an internal administrative hook (or listens to a file watcher / SIGHUP signal).
>    - The server unpickles the new model into a temporary background variable `new_model`.
>    - Once loaded, it executes an atomic in-memory pointer swap: `self.svd_model = new_model`.
>    - Active in-flight HTTP requests complete using the old reference; all subsequent requests immediately use the new weights with **zero dropped packets and zero downtime**."

---

### Q14.4: "Cold Retraining vs. Warm-Start Fine-Tuning: How do you prevent catastrophic forgetting?"
> **On-Point Answer:**  
> - **Full Cold Retraining:** Re-initializes $P$ and $Q$ with random Gaussian noise and trains from epoch 1. While thorough, it rotates the latent coordinate space, invalidating precomputed vector caches.
> - **Warm-Start Fine-Tuning:** Initializes $Q$ and $b_i$ with the weights from the previous checkpoint and runs 2–3 SGD epochs over only recent interactions with a decayed learning rate ($\gamma = 0.001$).
> - **Mitigating Catastrophic Forgetting:**
>   - We use **Experience Replay**: the fine-tuning batch contains $70\%$ new interactions and $30\%$ sampled historical anchor interactions.
>   - For newly released movies with zero historical ratings, we initialize their row in $Q$ using the average latent vector of movies sharing identical genres and directors, rather than random noise."

---

### Q14.5: "What automated validation gates (Guardrails) must a newly trained model pass before going live?"
> **On-Point Answer:**  
> "Never auto-deploy a retrained model without automated offline regression tests. Before promoting a new checkpoint to production, the CI/CD pipeline validates:
> 1. **Ranking Guardrail:** NDCG@10 on the holdout test set must not drop by more than $1.5\%$ compared to the currently serving model ($\text{NDCG}_{\text{new}} \ge \text{NDCG}_{\text{current}} - 0.015$).
> 2. **Coverage Guardrail:** Catalog coverage must remain above $70\%$ to ensure the new model has not collapsed into a popularity bubble.
> 3. **Latency Guardrail:** Benchmark test must confirm 50 candidate vector dot products execute in $< 2.0\text{ ms}$.
> 4. **NaN / Inf Sanity Check:** Assert zero `NaN` or `Inf` values exist in $Q$, $b_i$, or $\mu$.
> 
> If any guardrail fails, the deployment pipeline halts, keeps the current model active, and alerts on-call engineers via PagerDuty/Slack."

---

### Q14.6: "How would you automate this entire training and retraining pipeline using Airflow or Prefect?"
> **On-Point Answer:**  
> "In an enterprise environment, we orchestrate the training lifecycle as a Directed Acyclic Graph (DAG):
> 
> ```
> [Ingest Recent Ratings] ──► [Data Validation / Schema Check]
>                                      │
>                                      ▼
>                      [Run train_models.py (SGD SVD)]
>                                      │
>                                      ▼
>                      [Evaluate Holdout NDCG & Coverage]
>                                      │
>                ┌─────────────────────┴─────────────────────┐
>           [Pass Gates]                                [Fail Gates]
>                │                                           │
>                ▼                                           ▼
>    [Atomic S3 / GCS Push]                         [Alert On-Call]
>                │                                  [Halt Rollout]
>                ▼
>    [Webhook: Hot-Swap FastAPI RAM]
> ```
> 
> - **Sensor Task:** S3/GCS file sensor detects new daily interaction logs.
> - **Validation Task:** Validates schema and asserts null rating count is zero.
> - **Compute Task:** Executes `train_models.py` on an isolated GPU or CPU worker node.
> - **Evaluation Gate:** Computes offline NDCG@10. If valid, pushes the artifact to the model registry (MLflow / S3).
> - **Deploy Webhook:** Calls FastAPI's internal `/admin/reload-model` endpoint to hot-swap weights into server RAM."

---

## 15. End-to-End Recommendation Lifecycle & Scoring Factors

### Q15.1: "On what exact factors and mathematical weights is a movie recommended?"
> **On-Point Answer:**  
> "Every candidate movie in BollyWise is scored across a **6-factor positive ensemble** and bounded by **2 negative exclusion filters**:
> 
> $$\text{FinalScore}(u, i) = \underbrace{\alpha \cdot \hat{S}_{\text{latent}}(u, i)}_{\text{Factor 1 (65%)}} + \underbrace{(1 - \alpha) \cdot \text{Sim}_{\text{genre}}(u, i)}_{\text{Factor 2 (35%)}} + \underbrace{\text{Boost}_{\text{director}}}_{\text{Factor 3 (+5%)}} + \underbrace{\text{Boost}_{\text{cast}}}_{\text{Factor 4 (+4%)}} + \underbrace{\text{Prior}_{\text{Bayesian}}}_{\text{Factor 5}} - \underbrace{\text{MMR Penalty}}_{\text{Factor 6}}$$
> 
> 1. **Collaborative Latent Affinity (Weight: 65% when active):**
>    - Dot product $P_{\text{new}} \cdot Q_i$ in 32-D space.
>    - Captures deep, unwritten stylistic affinity (e.g. witty social satire, emotional pacing) learned from 80,000 historical ratings.
> 2. **Explicit Genre Alignment (Weight: 35% when active):**
>    - Jaccard tag intersection: $\frac{|G_u \cap G_i|}{\max(1, |G_u|)}$. Prioritizes titles matching user's active mood pill (*Comedy, Action, Drama*).
> 3. **Director Synergy Boost (+5% / +0.25 Lift):**
>    - Adds $+0.25$ if the movie director matches a director from any film the user rated $\ge 4.0\text{★}$ (*e.g., Rajkumar Hirani, Anurag Kashyap*).
> 4. **Lead Cast Affinity Boost (+4% / +0.20 Lift):**
>    - Adds $+0.20$ if lead actors starred in any film the user rated $\ge 4.0\text{★}$ (*e.g., Aamir Khan, Shah Rukh Khan*).
> 5. **Bayesian Quality & Credibility Prior:**
>    - Applies IMDb shrinkage: $WR = \frac{v}{v+50} R + \frac{50}{v+50} \times 3.53$. Prevents obscure movies with a single 5★ review from outranking certified cinema.
> 6. **Exploration & Anti-Bubble Diversity:**
>    - **Thompson Sampling Bandits (15%):** Samples from Beta conjugate prior $\text{Beta}(\alpha_i+1, \beta_i+1)$ to inject sleeper hits.
>    - **Maximal Marginal Relevance (MMR, $\lambda=0.70$):** Penalizes items that have high pairwise latent similarity to movies already picked in the top 10 (prevents 4 sequels from dominating the tray).
> 
> **Negative / Exclusion Rules:**
> - **History Pruning:** Movies already rated or bookmarked are removed.
> - **Negative Feedback (*'Not Interested'*):** Clicking the EyeOff button excludes the ID immediately and penalizes its sub-genres in real time."

---

### Q15.2: "Walk me through the complete data pipeline from raw CSVs to enriched Parquet files."
> **On-Point Answer:**  
> "The data engineering pipeline unifies three distinct data sources into clean, columnar Parquet files:
> 
> ```
> [Raw Data Sources]
> ├── MovieLens 100K (100k ratings, 943 users, 1,682 items)
> ├── Bollywood GitHub Dataset (2,199 titles: plot, cast, director)
> └── Indian Movies IMDb Dataset (15,509 titles: ratings, votes, duration)
>       │
>       ▼ ETL PIPELINE (Entity Resolution & Cleaning)
> ├── 1. Title Normalization: Regex year extraction, subtitle strip, lowercase tokens
> ├── 2. Fuzzy Token-Sort Join: Joins Bollywood metadata onto canonical IDs
> ├── 3. Data Validation: Pydantic range check (1.0 <= rating <= 5.0), null imputation
> ├── 4. Chronological Split: 80% Train (80,000) / 20% Holdout Test (20,000)
>       │
>       ▼ OUTPUT ARTIFACTS (Snappy Parquet)
> ├── data/processed/cleaned_items.parquet (1,682 rows)
> └── data/processed/train_interactions.parquet (80,000 rows)
> ```
> 
> **Why Parquet?** Columnar storage with Snappy compression reduces disk footprint by $> 75\%$ compared to raw CSVs and allows selective column reads directly into NumPy arrays."

---

### Q15.3: "How does the user journey transition from pure cold-start to full collaborative personalization?"
> **On-Point Answer:**  
> "BollyWise implements a seamless 5-stage progressive profiling funnel:
> 
> 1. **Stage 1 — Cold-Start Visitor (0 Ratings, 0 Genres):**
>    - Serves Bayesian Popularity baseline ($WR(i)$) displaying universally acclaimed classics (*Sholay, 3 Idiots, Lagaan*).
> 2. **Stage 2 — Mood Pill Selection:**
>    - User clicks *"Comedy"*. Engine filters candidates via genre Jaccard overlap while maintaining Bayesian quality rank.
> 3. **Stage 3 — Fast Calibration (Taste Profile Studio):**
>    - UI presents an 8-movie calibration tray of iconic Bollywood titles across diverse genres.
> 4. **Stage 4 — First Rating (Instant Ridge Folding-In):**
>    - The moment the user rates 1 movie 5★ (*e.g., 3 Idiots*), the backend inverts the $32 \times 32$ Ridge matrix in $0.18\text{ ms}$, computing $P_{\text{new}}$.
>    - Hybrid weight shifts to $\alpha = 0.50$ (warm-start).
> 5. **Stage 5 — Deep Personalization ($\ge 3$ Ratings):**
>    - Hybrid weight shifts to $\alpha = 0.65$ (collaborative latent affinity dominates).
>    - Director and cast synergy boosts activate.
>    - Thompson Sampling injects discovery gems, and negative feedback immediately purges unwanted titles."

---

### Q15.4: "How are posters sourced, why are some movies uncached, and how does the studio fallback work?"
> **On-Point Answer:**  
> "1. **Pre-Cached HD Posters (`data/movie_posters_cache.json`):**
>    - 248 official HD posters for top Bollywood titles were fetched offline from TMDb's search API (`https://image.tmdb.org/t/p/w500/...`).
>    - Loaded at server startup into an in-memory hash map ($O(1)$ lookup, zero disk I/O).
> 
> 2. **Why the Remaining 1,434 Movies Are Uncached:**
>    - **Strict Latency SLA:** Querying TMDb's external REST API in real time would introduce $300\text{ ms}–800\text{ ms}$ of network lag, violating our $< 50\text{ ms}$ SLA. The server never makes external HTTP calls during user requests.
>    - **Rate Limits:** TMDb rate-limits requests (40 requests / 10s); bulk querying 1,682 movies during setup risks IP blocking.
>    - **Transliteration Discrepancies:** Older Bollywood movies have conflicting English spellings (*e.g., 'Golmaal: Fun Unlimited - Part 2' vs 'Golmaal Returns'*).
> 
> 3. **Studio Graphic Dynamic Fallback:**
>    - When `movie.poster` is null, the frontend dynamically renders an elegant studio card with genre-themed gradients (*Crimson for Action, Amber for Comedy, Indigo for Drama*), genre emoji (💥, 😂, 🎭), and album-style typography.
>    - The UI never shows broken image icons or ugly browser placeholders."

---

### Q15.5: "What are the exact backend REST API endpoints, schemas, and response contracts?"
> **On-Point Answer:**  
> "The backend exposes 5 core endpoints built on FastAPI:
> 
> | Method | Endpoint | Purpose | Latency |
> | :--- | :--- | :--- | :--- |
> | `POST` | `/api/custom-user/recommend` | Main personalized recommendation feed (Two-Stage + Ridge) | **$< 3.0\text{ ms}$** (Cache hit: $< 0.5\text{ ms}$) |
> | `GET` | `/api/curated-onboarding-movies` | Returns 20 iconic titles for onboarding calibration | **$< 1.0\text{ ms}$** |
> | `GET` | `/api/movie/{movie_id}/similar` | Item-item cosine similarity in 32-D latent space | **$< 1.5\text{ ms}$** |
> | `GET` | `/api/system/metrics` | System telemetry, offline benchmarks, and PSI monitoring | **$< 0.8\text{ ms}$** |
> | `GET` | `/api/catalog` | Full catalog browsing with pagination and search | **$< 2.0\text{ ms}$** |
> 
> **Sample POST `/api/custom-user/recommend` Payload:**
> ```json
> {
>   "ratings": {"1": 5, "50": 4},
>   "favorite_genres": ["Comedy", "Action"],
>   "genre_filter": "Comedy",
>   "dismissed": [25, 42],
>   "top_k": 10
> }
> ```
> 
> **Response Contract:** Returns `status`, `degraded_mode`, `cache_status`, `cold_start_mitigation`, and a list of `recommendations`, each containing:
> - `item_id`, `title`, `release_year`, `genres`, `poster`, `imdb`, `runtime`, `director`, `actors`
> - `match_percentage`, `why_recommended` (plain-language explanation)
> - `logic`: Complete 4-stage pipeline stepper status, candidate pool sizes, and 4 decision meters (*Two-Tower Affinity %, Genre Match %, Creator Synergy %, Confidence %*)."

---

### Q15.6: "Is demographic or geographic information (e.g., `zip_code`, `age`, `gender` from MovieLens) used in recommendation ranking? Why or why not?"
> **On-Point Answer:**  
> "**No, geographic location (`zip_code`) and demographic attributes are deliberately NOT used in the active recommendation scoring loop.**
> 
> While legacy MovieLens raw surveys collected demographic columns (`age`, `gender`, `occupation`, `zip_code`), they were deliberately pruned from our clean serving data for three critical engineering reasons:
> 
> 1. **Behavioral Dominance over Demographics:**
>    - Empirical recommender systems research (including Netflix Prize findings) consistently proves that **interaction history** (what a user actually watches and rates) is orders of magnitude more predictive of movie satisfaction than their age, gender, or postal code.
>    - Latent factor embeddings ($P_{\text{new}} \cdot Q_i$) capture actual taste affinities directly, rendering coarse demographic buckets redundant.
> 2. **Spurious Correlations & Data Bias:**
>    - The raw MovieLens dataset contains historical US zip codes, while our enriched catalog focuses on Bollywood and Indian cinema. Conditioning recommendations on US zip codes would introduce noisy, spurious correlations that do not generalize to real users.
> 3. **Privacy & Onboarding UX Friction (GDPR / Zero-Party Data):**
>    - Asking a new user for their zip code or gender during onboarding creates high friction and raises privacy concerns. 
>    - BollyWise instead relies purely on **zero-party explicit interest** (1-click mood pills and movie star ratings), which respects user privacy while delivering immediate $< 2\text{ ms}$ personalization.
> 
> **Where Geography DOES Matter in Enterprise Systems:**
> - In global production streaming (e.g., Netflix/Prime Video), geography is used strictly at the **infrastructure & licensing layer** (Geo-blocking licensing rights, routing video streams to regional CDN edge caches, and prioritizing regional language dubs/subtitles), not for calculating user affinity scores."



