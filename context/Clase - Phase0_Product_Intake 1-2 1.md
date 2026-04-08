# Phase 0 — Product Intake & Scoping

**Music Popularity Predictor — AI-Native Software Project**

| Attribute                           | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Document ID**                     | MPP-PH0-INTAKE-v1.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Audience**                        | Project team (Data Science, Backend, Frontend/UI-UX); Academic supervisors                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Purpose**                         | Capture the foundational product decisions that seed every downstream phase of the Music Popularity Predictor application. This document produces a **Product Intake Brief** — a structured snapshot of what the product is, who it is designed for, what value it provides, and what technical constraints it operates within. Every decision here is a v1 that can evolve, but every downstream activity (Service Design, Domain Modeling, Technical Foundation, and Product Build) depends on these seeds existing. |
| **Duration**                        | 1–2 working days (initial project definition phase)                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Output**                          | product-intake-brief-v1.md — consumed by all subsequent phases                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Relationship to existing stages** | Phase 0 (this) → Phase 1 (Service Design / UX Design) → Stage 0 (Technical Foundation) → Phase 2 (Domain Modeling) → Stage 2 (Application Build) → Stage 3 (Testing & Optimization)                                                                                                                                                                                                                                                                                                                                    |

---

## Why This Phase Exists

Phase 1 Service Design (User Archetypes, Service Mapping, Journey Mapping, etc.) requires context to begin: What problem does the product solve? Who are the users (artists, producers, record labels)? What type of product is being built? What data and AI capabilities support it? Is the system being built from scratch or integrating existing components?

These questions are either answered explicitly here, or answered implicitly (and inconsistently) by each team member making different assumptions.

Phase 0 prevents that drift. It captures the **minimum viable context** so every design artifact — user archetypes, journey maps, interaction flows, and feature definitions — starts from the same shared understanding of the Music Popularity Predictor product.

| What happens WITHOUT Phase 0                                               | What happens WITH Phase 0                                             |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Team members assume different target users (artists vs labels vs analysts) | Everyone aligns on the same target users and product purpose          |
| UX design becomes generic and disconnected from the ML model               | UX design reflects the real capabilities of the prediction system     |
| Features are defined without considering the AI pipeline                   | Features align with the prediction model and audio feature extraction |
| Scope expands without clear boundaries                                     | Core capabilities defined before design begins                        |
| Technical decisions ignore data and model constraints                      | Model, data, and architecture constraints known before implementation |

---

## How to Use This Document

1. **At the beginning of the project**, the team defines the core context of the Music Popularity Predictor application (problem, users, and product scope).
2. Each module produces a section of the **Product Intake Brief** for the project.
3. The full brief is reviewed by the project team and becomes the **seed document** for Phase 1 (Service Design and UX definition).
4. The brief is a **living v1** — it can evolve as the project progresses, but updates should be documented and versioned.

---

## Module Overview

| Module    | Name                         | What It Captures                                                                               | Time       |
| --------- | ---------------------------- | ---------------------------------------------------------------------------------------------- | ---------- |
| **PI-01** | Project Context              | Problem in the music industry, motivation for predicting song popularity                       | 30 min     |
| **PI-02** | Product Identity             | Definition of the Music Popularity Predictor application and its main functionality            | 30 min     |
| **PI-03** | Users & Stakeholders         | Target users such as artists, producers, and record labels                                     | 30 min     |
| **PI-04** | Value Proposition            | How the product supports decision-making in music production and promotion                     | 30 min     |
| **PI-05** | Scope & Boundaries           | What features are included (prediction, audio analysis, optimization) and what is out of scope | 45 min     |
| **PI-06** | Constraints & Risks          | Technical limitations, dataset limitations, and dependency on external APIs                    | 30 min     |
| **PI-07** | Success Criteria & Readiness | Metrics to evaluate the project and readiness for the next development phases                  | 15 min     |
|           | **Total**                    |                                                                                                | **~3.5 h** |

---

## Module PI-01: Project Context

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-01                                                                                                                                                                                                                                                                        |
| **Output**         | Context of the Music Popularity Predictor project, the problem it addresses, and the opportunity it explores                                                                                                                                                                 |
| **Why it matters** | Every downstream decision (user archetypes, UX design, feature scope, and AI capabilities) depends on understanding the problem and the project goals. Without this context, the team may design features based on assumptions instead of the real objectives of the system. |

### Artifact Template — Project Context

| Field                      | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Guidance                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **Project ID**             | MPP-2025-01                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Internal project identifier      |
| **Project Name**           | Music Popularity Predictor — _¿Qué tan popular será tu canción?_                                                                                                                                                                                                                                                                                                                                                                                                                       |                                  |
| **Project Type**           | AI-powered analytics web application                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                  |
| **Project Stakeholders**   | Data Science Team · Backend Development · Frontend/UI-UX Design · Academic supervisors (Universidad EAFIT)                                                                                                                                                                                                                                                                                                                                                                             | Main contributors to the project |
| **Project Context**        | Spotify and other streaming platforms receive more than 120,000 new songs per day, making it practically impossible for record labels and independent artists to manually evaluate the potential of each release. Historically, promotion decisions were based on executive intuition or the artist's track record rather than quantitative evidence. This project addresses that gap by building a machine learning model that estimates a song's popularity from its audio features. |                                  |
| **Project Motivation**     | To build a system that supports data-driven decisions in music production by predicting song popularity based on musical characteristics extracted from audio data. The goal is to reduce subjectivity and help identify songs with high potential regardless of the artist's fame.                                                                                                                                                                                                    |                                  |
| **Strategic Objective**    | Develop a web application capable of analyzing song features and estimating its potential popularity using machine learning models trained on the _Spotify 1 Million Tracks_ dataset (2000–2023, ~1 million records, 19 variables).                                                                                                                                                                                                                                                    |                                  |
| **Project Team**           | Data Science: Model development and evaluation · Backend: FastAPI services and audio processing logic · Frontend/UI-UX: React interface and user interaction design                                                                                                                                                                                                                                                                                                                    |                                  |
| **Project Start Date**     | 2025-09-01                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                  |
| **Target Completion Date** | 2025-11-19                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                  |
| **Resources / Tools**      | Python · scikit-learn · LightGBM · XGBoost · CatBoost · FastAPI · React · librosa · ReccoBeats API · _Spotify 1 Million Tracks_ dataset                                                                                                                                                                                                                                                                                                                                                |                                  |

### Quality Checklist

| #   | Check                                                          | [ ] |
| --- | -------------------------------------------------------------- | --- |
| 1   | Project type clearly defined                                   | [x] |
| 2   | Project context and problem statement described                | [x] |
| 3   | Motivation for developing the system explained (≥ 2 sentences) | [x] |
| 4   | Strategic objective stated in a clear and concise sentence     | [x] |
| 5   | Project team and main stakeholders identified                  | [x] |

---

## Module PI-02: Product Identity

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-02                                                                                                                                                                                                                                                                                                                     |
| **Output**         | What the Music Popularity Predictor is as a digital product: its platform, delivery mechanism, and technical structure                                                                                                                                                                                                    |
| **Why it matters** | The product form (web application with AI prediction capabilities) determines the UX design, system architecture, and technical constraints. Phase 1 Service Design (user journeys, interaction flows, and features) depends on understanding how users will interact with the prediction system defined in Module PI-01. |

### Artifact Template — Product Identity Card

| Field                         | Content                                                                                                                                                                                                                                                                                                    | Options / Guidance                                                                                             |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Product Working Name**      | Music Popularity Predictor                                                                                                                                                                                                                                                                                 | Reference name for the project                                                                                 |
| **Product Type**              | Web application                                                                                                                                                                                                                                                                                            |                                                                                                                |
| **Platform Strategy**         | Web-only                                                                                                                                                                                                                                                                                                   | The system is accessed through a browser interface                                                             |
| **Delivery Model**            | Local (localhost) with AWS-ready architecture                                                                                                                                                                                                                                                              | Backend runs on FastAPI; frontend served via React; architecture designed for EC2 + S3 + CloudFront deployment |
| **Access Model**              | Public (academic project environment)                                                                                                                                                                                                                                                                      | Users interact through the web interface without authentication                                                |
| **Product Maturity**          | Greenfield (built from scratch)                                                                                                                                                                                                                                                                            | The system was developed from scratch as an academic project                                                   |
| **Existing Codebase?**        | Yes                                                                                                                                                                                                                                                                                                        | Backend and frontend components already implemented and tested end-to-end in localhost                         |
| **Existing Codebase Details** | Backend: Python + FastAPI · Frontend: React · ML models: LightGBM (production), XGBoost, CatBoost, Random Forest, MLP (evaluated) · Audio processing: librosa + ReccoBeats API                                                                                                                             |                                                                                                                |
| **Key Technical Constraint**  | The system must process audio features in real time (via ReccoBeats API with librosa as fallback) and run ML inference to estimate song popularity. The model must remain loaded in memory to avoid high latency — this rules out serverless functions (AWS Lambda) in favor of a persistent EC2 instance. |                                                                                                                |
| **AI-Native Ambition**        | AI is the core product (AI-first)                                                                                                                                                                                                                                                                          | The main value of the application comes from the ML prediction model                                           |

### User Interaction Flows

The user accesses the React interface and chooses one of three routes:

| Route                    | Description                                                                                                                                           | Backend endpoint                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Process audio**        | Upload an `.mp3` file + optional genre; features extracted via ReccoBeats and librosa; model returns a popularity prediction                          | `POST /extract-features`                    |
| **Search by Spotify ID** | Enter a Spotify track ID; backend resolves it to a ReccoBeats internal ID and returns features + prediction                                           | `GET /search-track` → `GET /track-features` |
| **Optimize parameters**  | Submit a set of base features with locked variables; backend evaluates a value grid and returns the configuration that maximizes predicted popularity | `POST /wizard-optimize`                     |

### Decision Tree — Product Type → Phase 1 Implications

| Product Type Decision     | Impact on Phase 1 Service Design                                                                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| Web application           | User journeys assume browser interaction through a React interface                                        |
| AI-first product          | UX must clearly expose prediction results, feature importance, and model limitations                      |
| Greenfield product        | All system features and interaction flows were designed from scratch                                      |
| Web-only platform         | Interaction design focuses on responsive web interfaces and REST API communication                        |
| Three distinct user flows | UX must accommodate audio upload, ID search, and parameter optimization as separate but coherent journeys |

### Quality Checklist

| #   | Check                                     | [ ] |
| --- | ----------------------------------------- | --- |
| 1   | Product type selected                     | [x] |
| 2   | Platform strategy selected                | [x] |
| 3   | Delivery model selected                   | [x] |
| 4   | Product maturity selected                 | [x] |
| 5   | Existing codebase technologies documented | [x] |
| 6   | AI-native ambition explicitly defined     | [x] |
| 7   | Key technical constraint identified       | [x] |

---

## Module PI-03: Market & Users

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-03                                                                                                                                                                                                                                                                                     |
| **Output**         | Who the Music Popularity Predictor serves, in what industry, and under what usage context. This provides the initial user hypotheses that will seed Phase 1 archetype discovery and UX design.                                                                                            |
| **Why it matters** | Phase 1 Service Design requires an understanding of the product domain and potential users. Defining the music industry context and potential users ensures that archetypes, journeys, and features are grounded in the real problem described in PI-01 and the product defined in PI-02. |

### Artifact Template — Market & User Sketch

| Field                      | Content                                                          | Options / Guidance                                                    |
| -------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Industry Vertical**      | Media / Music Industry                                           |                                                                       |
| **Sub-vertical**           | Music analytics and popularity prediction using machine learning |                                                                       |
| **Market Model**           | B2B2C (music industry professionals and independent artists)     |                                                                       |
| **Market Geography**       | Global                                                           | The model analyzes audio features independently of region or language |
| **Market Size Indicator**  | Not yet estimated                                                |                                                                       |
| **Target Company Size**    | Multiple segments                                                | Independent artists, music producers, and record label analysts       |
| **Target End User Sketch** | See table below                                                  | Pre-research hypothesis — will be validated in Phase 1                |

**Initial User Sketch (pre-archetype hypothesis):**

| #   | Provisional Label  | Role / Title         | Interaction Frequency | Buying vs. Using | Known Pain (hypothesis)                                                                                 |
| --- | ------------------ | -------------------- | --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| 1   | Music Producer     | Producer / Composer  | Weekly                | User             | Difficulty estimating the commercial potential of a song before release, currently relying on intuition |
| 2   | Independent Artist | Music Artist         | Occasional            | User             | Uncertainty about whether a track has strong streaming potential; limited access to analytical tools    |
| 3   | A&R Analyst        | Record Label Analyst | Monthly               | Buyer / User     | Need for objective, data-driven insights to prioritize which songs to promote out of a large catalog    |

_Note: These are NOT archetypes. They are seeds for the UA module. Phase 1 will produce proper evidence-backed archetypes._

### Quality Checklist

| #   | Check                                                        | [ ] |
| --- | ------------------------------------------------------------ | --- |
| 1   | Industry vertical selected                                   | [x] |
| 2   | Sub-vertical specified                                       | [x] |
| 3   | Market model selected                                        | [x] |
| 4   | Geography specified                                          | [x] |
| 5   | ≥ 2 initial user types sketched                              | [x] |
| 6   | At least one buyer ≠ user identified (if B2B/B2B2C)          | [x] |
| 7   | All user sketches marked as hypotheses (not validated facts) | [x] |

---

## Module PI-04: Business Model & Monetization

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-04                                                                                                                                                                                                                                                                   |
| **Output**         | How the product generates value (or saves costs, in an academic/research context). Defines the potential monetization strategy and value metrics for the Music Popularity Predictor system.                                                                             |
| **Why it matters** | The business model influences feature prioritization and product design. Even in an academic project, defining how the system could generate value helps align product features, user journeys, and success metrics with real-world applications in the music industry. |

### Artifact Template — Business Model Card

| Field                             | Content                                                                                                                                                                                                                                                             | Options / Guidance                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Revenue Model**                 | Pre-revenue (academic prototype) · Potential future model: Freemium                                                                                                                                                                                                 | The current version is a functional prototype with no commercial deployment                                          |
| **Pricing Strategy Shape**        | Free + premium upsell (hypothetical future model)                                                                                                                                                                                                                   | Free tier for individual song analysis; premium tier could unlock batch analysis, optimization wizard, or API access |
| **Pricing Reference Point**       | TBD                                                                                                                                                                                                                                                                 | Not defined for the academic phase                                                                                   |
| **Free Tier?**                    | Yes                                                                                                                                                                                                                                                                 | Basic popularity prediction for individual songs via the web interface (all three flows available)                   |
| **Free Tier Scope**               | Single-song analysis through any of the three available routes (audio upload, Spotify ID search, or manual parameter entry)                                                                                                                                         | Only if "Yes" above                                                                                                  |
| **Primary Value Metric**          | Number of song analyses / predictions performed                                                                                                                                                                                                                     | Each call to `/extract-features`, `/track-features`, or `/wizard-optimize` counts as one analysis                    |
| **Revenue Timeline**              | Pre-revenue (academic prototype)                                                                                                                                                                                                                                    | Revenue model is hypothetical; the current system is not commercially deployed                                       |
| **Key Unit Economics Constraint** | ML inference and audio processing must remain computationally efficient. The model must stay loaded in memory on the backend (EC2 or equivalent) to deliver low-latency predictions — serverless execution is not viable given model size and librosa dependencies. |                                                                                                                      |
| **Go-to-Market Model**            | Self-serve (product-led usage)                                                                                                                                                                                                                                      | Users access the tool directly through the web interface without requiring sales or onboarding assistance            |

### Decision Tree — Business Model → Phase 1 Implications

| Business Model Decision | Impact on Phase 1                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Freemium (future)       | Feature prioritization must distinguish core free analysis from advanced optimization capabilities                     |
| Self-serve              | UX must support zero-friction onboarding and immediate song analysis                                                   |
| AI-based analytics      | Journeys must clearly expose prediction results, feature explanations (SHAP-informed UI), and model limitations        |
| Three user flows        | Onboarding journey must help users understand which route fits their context (audio file, Spotify ID, or manual input) |

### Quality Checklist

| #   | Check                                                           | [ ] |
| --- | --------------------------------------------------------------- | --- |
| 1   | Revenue model selected (even if hypothetical for the prototype) | [x] |
| 2   | Pricing shape selected                                          | [x] |
| 3   | Primary value metric identified                                 | [x] |
| 4   | Go-to-market model selected                                     | [x] |
| 5   | Key unit economics constraint documented                        | [x] |
| 6   | If freemium: free tier scope defined                            | [x] |

---

## Module PI-05: Scope & Boundaries

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-05                                                                                                                                                                                                                                                                                                      |
| **Output**         | What is in scope for the Music Popularity Predictor project, what is explicitly out of scope, the transformation type, and the current system context                                                                                                                                                      |
| **Why it matters** | Phase 1 Service Design must understand whether the system is being designed from scratch or based on existing components. Feature prioritization and UX design depend on clearly defined boundaries. Without explicit scope, the project can expand beyond the intended academic and technical objectives. |

### Artifact Template — Scope & Boundaries Card

| Field                            | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Guidance                                    |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Transformation Type**          | Greenfield — New Product                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | See taxonomy below                          |
| **Transformation Description**   | The project develops a new AI-powered web application capable of predicting the popularity of songs using machine learning models and audio feature analysis. The system integrates data processing (librosa + ReccoBeats API), gradient boosting prediction models (LightGBM as the production model), and a React web interface that allows users to analyze songs through three distinct flows: audio upload, Spotify ID search, and parameter optimization.                                                                                                                                                                                                                                                                                                                                                                  |                                             |
| **Current-State Summary**        | No existing product performed this task within the project context. The system was developed from scratch as an academic project (Universidad EAFIT, Proyecto Integrador 2) combining machine learning, FastAPI backend services, and a React-based web interface. The full pipeline — from audio upload to popularity prediction — was implemented and tested end-to-end in a localhost environment.                                                                                                                                                                                                                                                                                                                                                                                                                            |                                             |
| **Existing Systems Inventory**   | N/A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Greenfield product — no prior system to map |
| **In-Scope for This Engagement** | 1. Machine learning model to predict song popularity (trained on _Spotify 1 Million Tracks_, ~333K records after filtering to top 30 genres by average popularity).<br>2. Backend service (FastAPI) to orchestrate audio processing and model inference.<br>3. Audio feature extraction from `.mp3` files via ReccoBeats API + librosa fallback.<br>4. Track feature retrieval by Spotify ID via ReccoBeats API.<br>5. React web interface supporting three user flows (audio upload, ID search, parameter optimization).<br>6. Visualization of predicted popularity score.<br>7. Parameter optimization wizard (`/wizard-optimize`) to simulate feature changes and find the configuration that maximizes predicted popularity.<br>8. Hypothetical AWS deployment architecture (EC2 for backend + S3/CloudFront for frontend).<br>9. GenAI-based audio preview generation from the optimized parameter configuration, available as a post-optimization exploratory feature. |                                          |
| **Explicitly Out of Scope**      | 1. Integration with commercial music streaming platforms for live production deployment.<br>2. Real-time large-scale production infrastructure or high-concurrency handling.<br>3. Playlist recommendation, marketing automation, or social media prediction features.<br>4. Persistent storage or management of user music libraries.<br>5. Incorporation of external signals not present in the dataset (social media data, artist follower counts, playlist placements, marketing spend).                                                                                                                                                                                                                                                                                                                                     |                                             |
| **Scope Change Process**         | Any change to the project scope must be discussed and approved by the project team. Changes should be documented and evaluated based on technical feasibility and alignment with project objectives and the academic timeline.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                             |
| **Phase 1 Tailoring**            | See table below                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                             |

### Transformation Type Taxonomy

| Type                                 | Description                                     | Phase 1 Impact                                                                                   |
| ------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Greenfield — New Product**         | Nothing exists; building from scratch           | SM-AS is lightweight (no current systems); CML is important; full UX and feature design required |
| **Greenfield — New Module**          | Adding a new capability to an existing product  | SM-AS maps the existing product as context; SM-TB adds new module                                |
| **Refactor — Technical**             | Same functionality, new architecture            | JN-AS/JN-TB may remain similar; focus on technical ecosystem                                     |
| **Refactor — Experience**            | Same product, redesigned UX                     | JN-AS/JN-TB drive the redesign                                                                   |
| **Enhancement — AI Augmentation**    | Adding AI capabilities to an existing product   | AI integration becomes the main design focus                                                     |
| **Pivot — Business Model**           | Same market, different monetization or delivery | Business model redesign impacts features and journeys                                            |
| **Migration — Platform**             | Moving from one platform to another             | Platform constraints reshape user journeys                                                       |
| **Integration — Connecting Systems** | Building integrations between systems           | Ecosystem mapping becomes critical                                                               |

### Existing Systems Inventory (for non-greenfield)

| System ID | System Name | Type | Role in Current State | Data It Holds | Integration Available? | Disposition Hypothesis |
| --------- | ----------- | ---- | --------------------- | ------------- | ---------------------- | ---------------------- |
| N/A       | N/A         | N/A  | N/A                   | N/A           | N/A                    | N/A                    |

_Note: This is a greenfield product. No existing systems to inventory._

### Phase 1 Module Tailoring

Based on scope, some Phase 1 modules may be shortened or adjusted:

| Phase 1 Module              | Greenfield New Product            | New Module                                | Technical Refactor                   | Experience Refactor             | AI Augmentation                         |
| --------------------------- | --------------------------------- | ----------------------------------------- | ------------------------------------ | ------------------------------- | --------------------------------------- |
| **UA** (Archetypes)         | Full                              | Full                                      | Abbreviated — reuse existing         | Full                            | Abbreviated — reuse existing            |
| **SM-AS** (Ecosystem As-Is) | Lightweight — no existing systems | Partial — map existing product as context | Full — current architecture critical | Full                            | Full — existing product is the baseline |
| **JN-AS** (Journeys As-Is)  | Skip if no current process        | Full                                      | Abbreviated — journeys unchanged     | Full — journeys are the problem | Full — need AI insertion points         |
| **CML** (Competitive)       | Full                              | Full                                      | Abbreviated — known space            | Full                            | Full — AI competitive differentiation   |
| **SM-TB** (Ecosystem To-Be) | Full                              | Full — focus on integration points        | Full — new architecture design       | Full                            | Full — AI Overlay is primary output     |
| **JN-TB** (Journeys To-Be)  | Full                              | Full                                      | Skip if journeys unchanged           | Full — primary deliverable      | Full — AI-enhanced journeys             |
| **PC** (Product Concept)    | Full                              | Abbreviated — concept exists              | Skip — same product                  | Full                            | Abbreviated                             |
| **FD** (Features)           | Full                              | Full                                      | Full — parity features               | Full                            | Full                                    |
| **FE/FP** (Eval/Priority)   | Full                              | Full                                      | Full                                 | Full                            | Full                                    |

### Quality Checklist

| #   | Check                                                    | [ ] |
| --- | -------------------------------------------------------- | --- |
| 1   | Transformation type selected from taxonomy               | [x] |
| 2   | Current-state summary written (even if "nothing exists") | [x] |
| 3   | If non-greenfield: ≥ 1 existing system inventoried       | N/A |
| 4   | In-scope list has 5–10 specific items                    | [x] |
| 5   | Out-of-scope list has ≥ 3 explicit exclusions            | [x] |
| 6   | Phase 1 tailoring decisions documented                   | [x] |
| 7   | Scope change process defined                             | [x] |

---

## Module PI-06: Constraints & Risks

### Definition

| Attribute          | Value                                                                                                                                                                                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-06                                                                                                                                                                                                                                                                                   |
| **Output**         | Hard constraints (non-negotiable) and risks (manageable) that shape every subsequent decision — technical, academic/project constraints, timeline, and resource limitations                                                                                                             |
| **Why it matters** | Identifying constraints early ensures that system architecture, model selection, and UX design remain feasible within the project's technical and academic context. Discovering constraints later in development could require redesigning models, infrastructure, or product features. |

### Artifact Template — Constraints & Risks Register

**Hard Constraints (non-negotiable — must be designed around):**

| Constraint ID | Category       | Constraint                                                                                              | Impact on Design                                                                                                     | Source                              |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| HC-01         | Technical      | The system must process audio features in real time and run ML inference for song popularity prediction | Backend must support the full pipeline: ReccoBeats API call + librosa fallback + model inference in a single request | Project technical requirements      |
| HC-02         | Technical      | The model must remain loaded in memory to deliver low-latency predictions                               | Serverless execution (e.g., AWS Lambda) is not viable; a persistent runtime (EC2 or equivalent) is required          | Model size and librosa dependencies |
| HC-03         | Technical      | The application must be implemented using Python (backend/ML) and React (frontend)                      | All ML training, inference, and audio processing must remain within the Python ecosystem                             | Project development stack           |
| HC-04         | Technical      | Dependency on ReccoBeats API for audio feature extraction from `.mp3` files                             | A librosa-based local fallback must exist for when the external API is unavailable or returns an error               | External API reliability            |
| HC-05         | Organizational | The project is an academic deliverable (Proyecto Integrador 2, Universidad EAFIT)                       | Development scope is bounded by the academic semester; not all desirable features could be implemented               | Academic project requirements       |
| HC-06         | Budget         | Development must rely exclusively on open-source tools and publicly available datasets                  | No paid APIs beyond ReccoBeats as a test integration; no cloud infrastructure costs during the academic phase        | Project resource constraints        |

**Constraint Categories:**

| Category           | Examples                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Technical**      | Real-time audio processing pipeline, persistent model loading, Python ecosystem, ReccoBeats API dependency, librosa fallback |
| **Regulatory**     | No specific regulatory requirements for this academic project                                                                |
| **Organizational** | Academic project scope bounded by semester timeline; team divided into DS, Backend, and Frontend roles                       |
| **Timeline**       | Project completion by 19/11/2025 aligned with EAFIT academic calendar                                                        |
| **Budget**         | Open-source libraries only; _Spotify 1 Million Tracks_ public dataset; no cloud infrastructure costs in academic phase       |

**Risks (manageable — need mitigation):**

| Risk ID | Risk                                                                                                                                     | Category            | Likelihood               | Impact | Mitigation                                                                                                                                       | Owner             |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| R-01    | Model predictive capacity is inherently limited (R² ≈ 0.39); ~60% of popularity variance is driven by factors not present in the dataset | Technical           | H (already materialized) | H      | Communicate clearly in the UI that the prediction is a support tool, not a guarantee; include an explicit limitations note on the results screen | Data Science Team |
| R-02    | Dataset features do not capture key popularity drivers: marketing spend, playlist placements, artist fame, social media virality         | Technical           | H (already evidenced)    | H      | Document limitations in the product; suggest future data enrichment (social signals, artist metadata) as a roadmap item                          | Data Science Team |
| R-03    | ReccoBeats API unavailability or error responses during audio processing                                                                 | Technical           | M                        | M      | librosa-based fallback already implemented; backend returns a clear error message if both paths fail                                             | Backend Team      |
| R-04    | High retraining time (XGBoost: ~14,000s; LightGBM: ~3,000s) makes model updates expensive                                                | Technical           | M                        | M      | LightGBM selected as the production model for its balance of performance and efficiency; retrain only when dataset changes significantly         | Data Science Team |
| R-05    | Model bias toward dominant genres; minority genres may receive less reliable predictions                                                 | Ethical / Technical | M                        | M      | Dataset filtered to top 30 genres by average popularity; SHAP values confirm genre as a significant predictor; surface this limitation in the UI | Data Science Team |
| R-06    | Integration issues between React frontend, FastAPI backend, and ML model in a deployed environment                                       | Technical           | M                        | M      | Modular API architecture; all three flows tested end-to-end locally against real endpoints (no mocks)                                            | Backend Team      |
| R-07    | Limited development time restricts implementation of advanced features (e.g., batch analysis, user history)                              | Resource            | M                        | L      | Core three flows prioritized and delivered; advanced features deferred to a future roadmap                                                       | Project Team      |
| R-08 | Generated audio previews may be perceived as exact production recommendations when they are only synthetic approximations of the optimized feature profile | Product / Ethical | M | M | Clearly disclose in the UI that the preview is illustrative, not a full composition or guaranteed hit formula | Project Team |

### Quality Checklist

| #   | Check                                                                                                               | [ ] |
| --- | ------------------------------------------------------------------------------------------------------------------- | --- |
| 1   | ≥ 3 hard constraints documented                                                                                     | [x] |
| 2   | At least one constraint per category (Technical, Regulatory, Organizational, Timeline, Budget) — or explicit "None" | [x] |
| 3   | ≥ 5 risks identified                                                                                                | [x] |
| 4   | Every High-Impact risk has an actionable mitigation                                                                 | [x] |
| 5   | Every risk has an owner                                                                                             | [x] |
| 6   | Phase 1 blockers explicitly flagged                                                                                 | [x] |

---

## Module PI-07: Success Criteria & Readiness

### Definition

| Attribute          | Value                                                                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Module ID**      | PI-07                                                                                                                                                                                                                                  |
| **Output**         | How we know this project succeeded (measurable criteria) and what is needed from the project team to start Phase 1                                                                                                                     |
| **Why it matters** | Without defined success criteria, the project has no clear finish line. Without a readiness checklist, Phase 1 may start without the required inputs (dataset, development environment, project roles) and stall early in the process. |

### Artifact Template — Success Criteria

| Criteria ID | Type      | Success Criterion                                                                                                      | Measurement                                                     | Target                                                          | Measurement Point           |
| ----------- | --------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| SC-01       | Business  | Demonstrate the value of a data-driven approach for predicting song popularity in a streaming context                  | Evaluation of the functional prototype by academic stakeholders | Functional prototype delivered and presented                    | End of project (19/11/2025) |
| SC-02       | Product   | Users can analyze a song (via audio upload, Spotify ID, or manual parameters) and obtain a predicted popularity score  | Successful execution of all three analysis flows without errors | 100% of core workflow operational across the three routes       | Final prototype validation  |
| SC-03       | Technical | Machine learning model successfully predicts song popularity using audio features, with documented performance metrics | Test set evaluation: MSE, RMSE, MAE, R²                         | LightGBM: MSE = 152.29 · RMSE = 12.34 · MAE = 9.72 · R² = 0.389 | Model evaluation phase      |
| SC-04       | Timeline  | Complete development and deliver the prototype within the academic project timeline                                    | Comparison with project deadlines                               | Delivery by 19/11/2025                                          | End of engagement           |

**Guidance — minimum success criteria:**

| Type          | Example                                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Business**  | Demonstrate how data analysis can support decision-making in the music industry, reducing reliance on intuition alone                          |
| **Product**   | Web application allows users to analyze songs through three flows and obtain a predicted popularity score                                      |
| **Technical** | LightGBM model processes song features and returns predictions with R² ≈ 0.39 on the test set; SHAP values computed for model interpretability |
| **Timeline**  | Final prototype delivered before 19/11/2025                                                                                                    |

### Artifact Template — Client Readiness Checklist

| #   | Readiness Item                                                                | Required For                  | Status | Due By | Notes                                                                             |
| --- | ----------------------------------------------------------------------------- | ----------------------------- | ------ | ------ | --------------------------------------------------------------------------------- |
| 1   | Project team roles defined (data science, backend, frontend/UI-UX)            | Phase 1 activities            | [x]    | Day 1  | 4-person team: Integrante 1, Integrante 2, Integrante 3, Integrante 4             |
| 2   | _Spotify 1 Million Tracks_ dataset downloaded and accessible                  | Model training and evaluation | [x]    | Day 1  | ~1M records, 19 variables, 2000–2023; filtered to 333K records for modeling       |
| 3   | Development environment configured (Python, ML libraries, FastAPI, React)     | Model and backend development | [x]    | Day 2  | LightGBM, scikit-learn, librosa confirmed; ReccoBeats API access tested           |
| 4   | Initial project scope agreed within the team                                  | Phase 1 Service Design        | [x]    | Day 2  | Three user flows defined: audio upload, Spotify ID search, parameter optimization |
| 5   | Access to development tools and repositories                                  | Technical implementation      | [x]    | Day 3  |                                                                                   |
| 6   | Project timeline and milestones confirmed                                     | Project coordination          | [x]    | Day 1  | Academic calendar: delivery 19/11/2025                                            |
| 7   | Definition of system architecture components (frontend, backend, ML pipeline) | Stage 0 Foundation            | [x]    | Week 1 | React + FastAPI + LightGBM + librosa + ReccoBeats API                             |

### Quality Checklist

| #   | Check                                                       | [ ]                       |
| --- | ----------------------------------------------------------- | ------------------------- |
| 1   | ≥ 1 business success criterion defined with a number        | [x]                       |
| 2   | ≥ 1 product success criterion defined                       | [x]                       |
| 3   | ≥ 1 technical success criterion defined                     | [x]                       |
| 4   | Timeline target set                                         | [x]                       |
| 5   | Project readiness checklist reviewed with the team          | [x]                       |
| 6   | All initial readiness items confirmed before Phase 1 starts | [x]                       |
| 7   | Gaps in readiness items documented with mitigation plan     | N/A — all items confirmed |

---

## Assembled Artifact: Product Intake Brief

When all 7 modules are complete, the SA assembles them into a single document:

```
Product Intake Brief — Music Popularity Predictor
Project: MPP-2025-01
Team: Data Science, Backend, Frontend/UI-UX
Date: 14/03/2026
Version: 1.0
Status: Approved

1. Engagement Context            {PI-01 artifact}
2. Product Identity              {PI-02 artifact}
3. Market & Users                {PI-03 artifact, including user sketch}
4. Business Model                {PI-04 artifact}
5. Scope & Boundaries            {PI-05 artifact, including transformation type and Phase 1 tailoring}
6. Constraints & Risks           {PI-06 artifact}
7. Success Criteria & Readiness  {PI-07 artifact, including readiness checklist status}
```

## Sign-off

| Role              | Name | Date | Signature |
| ----------------- | ---- | ---- | --------- |
| Project Lead      |      |      |           |
| Data Science Lead |      |      |           |
| UI/UX Lead        |      |      |           |

---

Version History:

| Version | Date       | Author | Changes                                                                                            |
| ------- | ---------- | ------ | -------------------------------------------------------------------------------------------------- |
| 1.0     | 14/03/2026 | HPWSB  | Content updated to reflect actual project implementation (ProyectoIntegrador2 — Universidad EAFIT) |

---

## Phase 0 Completion Gate

| #   | Gate Item                                                                            | [ ] |
| --- | ------------------------------------------------------------------------------------ | --- |
| 1   | All 7 modules completed (no blank fields except explicitly "N/A" or "TBD with date") | [x] |
| 2   | Transformation type selected and Phase 1 tailoring documented                        | [x] |
| 3   | ≥ 3 hard constraints documented                                                      | [x] |
| 4   | ≥ 5 risks identified with mitigations                                                | [x] |
| 5   | Success criteria include ≥ 1 measurable business outcome                             | [x] |
| 6   | Project readiness: all initial items confirmed                                       | [x] |
| 7   | Product Intake Brief assembled and reviewed by the project team                      | [x] |
| 8   | Brief stored as `product-intake-brief-v1.md` in `/docs/phase0/`                      | [ ] |

**Phase 1 Service Design CANNOT begin until this gate passes.**

---

## How Phase 0 Feeds Phase 1

| Phase 0 Output                  | Consumed By Phase 1 Module                              | Specifically Used In        |
| ------------------------------- | ------------------------------------------------------- | --------------------------- |
| Industry + Sub-vertical (PI-03) | UA (prompt variable `{INDUSTRY}`)                       | UA-DRAFT prompt             |
| Product description (PI-02)     | UA (prompt variable `{PRODUCT_DOMAIN}`)                 | UA-DRAFT prompt             |
| User sketch (PI-03)             | UA (raw inputs for archetype generation)                | UA-2 step                   |
| Market model (PI-03)            | CML (competitive scope)                                 | CML-DISCOVER prompt         |
| Existing systems (PI-05)        | SM-AS (starting inventory)                              | SM-AS-1 through SM-AS-4     |
| Transformation type (PI-05)     | SM-TB (disposition decisions)                           | SM-TB-1 through SM-TB-4     |
| Constraints (PI-06)             | SM-TB (hard constraints shape to-be design)             | SM-TB-3, SM-TB-4            |
| AI ambition (PI-02)             | SM-TB (AI overlay scope)                                | SM-TB-4                     |
| Business model (PI-04)          | FD (feature types: onboarding, analysis features, etc.) | FD-1, FD-3                  |
| Business model (PI-04)          | FP (P0 must include core product capabilities)          | FP-4, FP-5                  |
| Success criteria (PI-07)        | PC (product concept success metrics)                    | PC-1                        |
| Go-to-market (PI-04)            | JN-TB (user onboarding journey design)                  | JN-TB-1                     |
| Phase 1 tailoring (PI-05)       | All modules                                             | SA adjusts depth per module |

---

_MPP-PH0-INTAKE-v1.0 | Music Popularity Predictor Project | Universidad EAFIT_
