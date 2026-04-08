# Phase 1 — Service Design: Execution Playbook

**Music Popularity Predictor — _¿Qué tan popular será tu canción?_**

| Attribute | Value |
|-----------|-------|
| **Document ID** | MPP-PH1-SD-v1.0 |
| **Project** | MPP-2025-01 — Music Popularity Predictor |
| **Audience** | Project team (Data Science, Backend, Frontend/UI-UX); Academic supervisors |
| **Phase dependency** | Requires Phase 0 completion gate passed (MPP-PH0-INTAKE-v1.0) |
| **Date** | 14/03/2026 |
| **Version** | 1.0 |

---

## Module 1 : UA — User Archetypes

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | UA |
| **What you are producing** | 3 data-backed behavioral profiles of the users who will interact with the Music Popularity Predictor. Each profile is scored, traceable to evidence from the project context and the _Spotify 1 Million Tracks_ dataset, and prioritized. |
| **Time-box** | 1.5 days |
| **You need before starting** | Phase 0 complete (MPP-PH0-INTAKE-v1.0); user sketch from PI-03; project context from PI-01 |
| **Who you need** | Project team (for raw input + validation); Data Science lead (for model capabilities and limitations) |
| **What consumes your output** | SM-AS (roles derive from archetypes), JN-AS (journeys map archetype behavior), all downstream modules |

---

### Archetype Card UA-01

| Field | Content |
|-------|---------|
| **Archetype ID** | UA-01 |
| **Name** | The Data-Driven Producer |
| **Role & Context** | Music producer or composer; works independently or within a small studio; mid-to-high tech proficiency; regularly uses DAWs and digital distribution tools; produces multiple tracks per month |
| **Primary Goal** | Evaluate the commercial potential of a finished track before committing promotional budget to its release |
| **Secondary Goals** | (1) Understand which specific audio features are holding back the popularity score; (2) Experiment with parameter adjustments via the optimization wizard to simulate what changes could improve predicted popularity; (3) Compare the popularity potential of alternative versions of a track |
| **Pain Points (scored 1–5)** | (5) Has no objective tool to estimate commercial performance — decisions are based on gut feeling or peer feedback, which is inconsistent; (4) Cannot identify which specific audio characteristics differentiate a commercially successful track from an unsuccessful one; (3) Wastes promotional spend on releases that do not resonate, without understanding the technical reasons why |
| **Frustration Triggers** | Receiving a popularity score with no explanation of which features drove it; long wait times between uploading audio and receiving results; results that seem inconsistent across similar songs |
| **Context of Use** | Post-production stage, typically in a home studio; uses a laptop browser; sessions are focused and short (5–15 min); uses the tool before finalizing release decisions |
| **Trust Expectations** | The prediction must include a clear explanation of contributing factors (SHAP-informed); the tool must acknowledge that it is a support instrument, not a guarantee — receiving a confident wrong prediction with no caveat is unforgivable |
| **Success Metrics** | Can identify the top 3 audio features impacting the predicted popularity score in under 5 minutes; uses the optimization wizard at least once per session to simulate a parameter change |
| **Data Sources** | PI-01 (problem statement: intuition-based decisions), PI-03 (user sketch: Music Producer), PI-06 (R-01: model limitations documented), PDF Section 3.3.5 (SHAP values: duration and danceability are top predictors) |
| **Priority** | Primary |

---

### Archetype Card UA-02

| Field | Content |
|-------|---------|
| **Archetype ID** | UA-02 |
| **Name** | The Independent Artist Exploring Streaming |
| **Role & Context** | Independent musician; self-releases tracks on Spotify and similar platforms; low-to-mid tech proficiency; limited access to analytics or industry advisory services; releases 2–6 tracks per year |
| **Primary Goal** | Understand whether a song has realistic streaming potential before investing personal resources in promotion |
| **Secondary Goals** | (1) Quickly look up how a specific Spotify track scores in the predictor model via Spotify ID; (2) Learn what genre and audio feature combination tends to perform better on streaming platforms; (3) Reduce financial risk by de-selecting tracks with low predicted popularity before paying for promotion |
| **Pain Points (scored 1–5)** | (5) Lacks access to analytical insights that major labels use — relies entirely on social media feedback and personal judgment; (4) Cannot interpret raw audio feature data (danceability, energy, valence) and connect it to streaming outcomes; (3) Uncertain whether using a Spotify track ID is sufficient to obtain a prediction without uploading an audio file |
| **Frustration Triggers** | A complex interface that requires technical knowledge to operate; unclear error messages when a Spotify ID is not found; results presented without plain-language explanation |
| **Context of Use** | At home, on a laptop or desktop browser; irregular usage (per release cycle); short sessions (3–10 min); arrives with a specific song or Spotify ID in mind |
| **Trust Expectations** | The tool must be usable without a technical background; explanations must avoid data science jargon; processing errors must be communicated clearly and not silently fail |
| **Success Metrics** | Completes a full song analysis (via audio upload or Spotify ID) without external help; understands the top factors influencing the score after reading the results screen |
| **Data Sources** | PI-03 (user sketch: Independent Artist), PI-04 (free tier: individual song analysis via all three flows), PI-02 (interaction flows: audio upload + Spotify ID search), PDF Section 1 (motivation: reducing subjectivity, promoting equity for lesser-known artists) |
| **Priority** | Primary |

---

### Archetype Card UA-03

| Field | Content |
|-------|---------|
| **Archetype ID** | UA-03 |
| **Name** | The A&R Catalog Analyst |
| **Role & Context** | Analyst at a record label or music publishing company; evaluates incoming song submissions and catalog performance; high tech proficiency; uses spreadsheets, internal dashboards, and streaming analytics platforms; reviews multiple tracks per session |
| **Primary Goal** | Apply an objective, data-driven filter to prioritize which songs in a submission pipeline warrant promotional investment |
| **Secondary Goals** | (1) Retrieve audio feature profiles for specific tracks by Spotify ID to compare against the prediction model; (2) Use the parameter optimization wizard to identify what audio adjustments would maximize predicted popularity of a candidate track; (3) Reduce the volume of subjective debate about song selection in A&R meetings by anchoring discussion in model-backed scores |
| **Pain Points (scored 1–5)** | (5) Evaluating large volumes of submissions without a consistent, repeatable scoring framework leads to high variance in selection decisions; (4) Existing tools (Spotify for Artists, Chartmetric) provide post-release analytics, not pre-release prediction; (3) Difficulty explaining internally why a song was selected or rejected without quantitative backing |
| **Frustration Triggers** | Inability to process multiple songs in a single session (no batch mode); prediction results that cannot be exported or shared with colleagues; model output that lacks confidence intervals or uncertainty signaling |
| **Context of Use** | Office or remote work environment; desktop browser; longer sessions (15–30 min) reviewing several tracks sequentially; uses results to prepare internal reports or recommendation memos |
| **Trust Expectations** | Prediction scores must be reproducible — running the same song twice must yield the same result; model limitations (R² ≈ 0.39; factors like marketing spend not included) must be clearly disclosed so the analyst can contextualize results |
| **Success Metrics** | Reduces time-to-recommendation per track from days (subjective review cycle) to under 30 minutes; uses the tool in at least one A&R meeting as supporting evidence |
| **Data Sources** | PI-03 (user sketch: A&R Analyst), PI-06 (R-01, R-02: model limitations; R-05: genre bias), PDF Section 3.4 (gradient boosting most robust; R² ~0.39), PI-05 (out of scope: batch analysis deferred) |
| **Priority** | Secondary |

---

### Archetype Priority Summary

| Archetype ID | Name | Priority | Justification |
|---|---|---|---|
| UA-01 | The Data-Driven Producer | Primary | Most direct alignment with all three user flows; highest interaction frequency; core beneficiary of SHAP-informed results and optimization wizard |
| UA-02 | The Independent Artist Exploring Streaming | Primary | Largest potential user segment; validates the equity/accessibility value proposition (PI-01); drives self-serve UX requirements |
| UA-03 | The A&R Catalog Analyst | Secondary | Higher-value use case but deferred scope (no batch analysis in v1); validates the tool's business case for future commercial development |

### Quality Checklist

| # | Check | [ ] |
|---|---|---|
| 1 | 3–5 archetypes produced | [x] |
| 2 | Every attribute traceable to Phase 0 or PDF source | [x] |
| 3 | Pain points scored 1–5 | [x] |
| 4 | At least 1 frustration trigger and 1 trust expectation per archetype | [x] |
| 5 | At least 1 buyer different from user identified | [x] — UA-03 is the buyer/recommender; UA-01 and UA-02 are direct users |
| 6 | Priority assigned to all archetypes | [x] |

---

## Module 2 : SM-AS — Service Ecosystem Map As-Is

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | SM-AS |
| **What you are producing** | A four-layer map of the current human ecosystem around music release decisions: who participates (L1: stakeholder roles), how they interact (L2: channels), what moves between them (L3: information flows), and what tools they currently use (L4: resources). This is the pre-product state — how artists, producers, and labels currently make popularity decisions WITHOUT the Music Popularity Predictor. |
| **Time-box** | 1.5–2 days |
| **You need before starting** | Validated archetype cards (UA-01 through UA-03) |
| **Who you need** | Data Science lead (domain knowledge), full project team (validation) |
| **What consumes your output** | JN-AS (journeys reference roles/channels/resources), SM-TB (transforms this map) |

---

### L1 — Stakeholder Roles

| Role ID | Role Name | Type | Organization | Archetype Mapping | Accountability | Authority Level | Frequency | Likely Pain Points |
|---|---|---|---|---|---|---|---|---|
| RL-01 | Music Producer / Composer | Core | Independent studio or label | UA-01 | Creates and finalizes tracks; decides which version to release | Decision-maker for own productions | Weekly | No quantitative tool to validate release decisions; relies on intuition |
| RL-02 | Independent Artist | Core | Self-managed | UA-02 | Owns the release decision; also manages promotion | Full owner | Occasional (per release) | No access to professional analytics; limited budget for tools |
| RL-03 | A&R Analyst | Core | Record label or publisher | UA-03 | Evaluates submissions; recommends which tracks to develop or release | Recommender (approval by senior A&R) | Daily to weekly | Large submission volumes; inconsistent evaluation criteria |
| RL-04 | Senior A&R Director | Adjacent | Record label | — | Final approval on signing or promotional investment decisions | Final decision-maker | Monthly | Needs a defensible rationale for investment decisions |
| RL-05 | Music Distributor | Upstream | Distribution platform (e.g., DistroKid, TuneCore) | — | Handles submission to Spotify and other platforms | Service provider | Per-release | [INFERRED — needs validation] |
| RL-06 | Streaming Platform Algorithm | Downstream | Spotify / Apple Music | — | Determines playlist placement and organic reach after release | Automated | Continuous | Opaque algorithm; artists cannot predict or influence placement directly |
| RL-07 | Digital Marketing Manager | Adjacent | Label or independent agency | — | Plans and executes promotion campaigns for scheduled releases | Executor of strategy | Weekly | Needs an early signal on song potential to justify budget allocation |

_Note: RL-06 is included as a downstream actor representing the real-world popularity driver that the model approximates. It is not a person but an organizational force shaping outcomes._

---

### L2 — Interaction Channels

| Channel ID | From Role | To Role | Channel Type | Channel Medium | Direction | Frequency | Formality | Friction Level (1–5) | Friction Description |
|---|---|---|---|---|---|---|---|---|---|
| CH-01 | RL-01 (Producer) | RL-03 (A&R Analyst) | External communication | Email + file sharing (WeTransfer, Drive) | Unidirectional (submission) | Per track | Semi-formal | 4 | No standardized submission format; A&R response time is unpredictable; no acknowledgment of receipt is standard |
| CH-02 | RL-02 (Independent Artist) | RL-05 (Distributor) | Service portal | Distribution platform web interface | Unidirectional | Per release | Formal | 2 | Mostly automated; friction is low at submission but high post-release if errors occur |
| CH-03 | RL-03 (A&R Analyst) | RL-04 (Senior A&R Director) | Internal communication | Email + internal presentation | Unidirectional (recommendation) | Monthly | Formal | 3 | Recommendations lack quantitative backing; subjective framing leads to internal disagreement |
| CH-04 | RL-01 (Producer) | RL-07 (Marketing Manager) | Internal communication | Direct conversation / messaging | Bidirectional | Per release | Informal | 3 | Timing misalignment; marketing needs early signals that producers cannot provide objectively |
| CH-05 | RL-06 (Platform Algorithm) | RL-01 / RL-02 / RL-03 | Automated feed | Streaming analytics dashboards (Spotify for Artists, Chartmetric) | Unidirectional (post-release) | Continuous | Automated | 4 | Data is post-release only; does not support pre-release prediction; no explanation of why a song performed as it did |

_High-friction touchpoints: CH-01 (score 4), CH-05 (score 4)._

_Missing channel: No channel currently exists for pre-release popularity estimation between producers/artists and analysts — this is the gap the Music Popularity Predictor fills._

---

### L3 — Information Flows

| Flow ID | Flow Name | Type | From Role | To Role | Channel (ref) | Object | Format | Transformation Required? | Frequency | Pain Point (ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| FL-01 | Track Submission Package | Information | RL-01 | RL-03 | CH-01 | Audio file + metadata (artist, genre, BPM, mood description) | .mp3 / .wav + text email | Yes — A&R must manually listen and score; no standardized feature extraction [MANUAL TRANSFORM — AI candidate] | Per track | UA-03 pain (5): no consistent scoring framework |
| FL-02 | A&R Evaluation Memo | Decision | RL-03 | RL-04 | CH-03 | Subjective recommendation document | Word / Slides | No — content is subjective; not data-backed | Monthly | UA-03 pain (3): difficulty justifying selection decisions internally |
| FL-03 | Release Performance Data | Information | RL-06 | RL-01 / RL-02 / RL-03 | CH-05 | Streams, saves, skip rate, playlist adds | Dashboard (Spotify for Artists) | No — but arrives post-release; cannot influence pre-release decisions | Continuous | UA-01 pain (5): no pre-release signal available |
| FL-04 | Promotional Brief | Information | RL-01 / RL-03 | RL-07 | CH-04 | Release date, target audience, priority tracks | Informal email or meeting | No | Per release | UA-01 pain (3): promotional decisions made without popularity data |

---

### L4 — Resources & Tools

| Resource ID | Resource Name | Type | Used By | Purpose | Replaceability | Known Limitations | Pain Point (ref) | SPOF? |
|---|---|---|---|---|---|---|---|---|
| RES-01 | Spotify for Artists | Software system | RL-01, RL-02, RL-03 | Post-release streaming analytics | Medium | Post-release only; no pre-release prediction capability | UA-01 (5), UA-02 (5) | No |
| RES-02 | Chartmetric / Soundcharts | Software system | RL-03 | Catalog and trend analytics | High — [REPLACEMENT TARGET] | Requires subscription; no audio feature-based prediction | UA-03 (4) | No |
| RES-03 | Subjective Listening Experience | Human expertise | RL-01, RL-03 | Primary method of evaluating track quality and potential | Low — irreplaceable for artistic judgment | Highly variable across individuals; not reproducible; slow at scale | UA-01 (5), UA-03 (5) | Yes |
| RES-04 | Email + File Sharing Tools | Communication infra | RL-01, RL-02, RL-03, RL-07 | Track submission and internal communication | High — [REPLACEMENT TARGET] | No standardization; no acknowledgment workflow; no feature extraction | FL-01 friction | No |
| RES-05 | DAW (Digital Audio Workstation) | Software system | RL-01, RL-02 | Music production and export | Low | Outputs audio files but provides no popularity signal | — | No |
| RES-06 | Industry Experience / Intuition | Human expertise | RL-03, RL-04 | Decision-making heuristic for track selection | Low — irreplaceable for cultural judgment | Subjective; varies by individual; introduces bias toward known artists and dominant genres | UA-03 (5), PI-06 R-05 | Yes |

---

## Module 3 : JN-AS — Journeys As-Is

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | JN-AS |
| **What you are producing** | Step-by-step maps of how each primary archetype currently accomplishes the job of evaluating song popularity potential WITHOUT the Music Popularity Predictor. Captures actions, channels/resources used, emotional states, timing, and failure paths. |
| **Time-box** | 1–1.5 days |
| **You need before starting** | Archetype cards (UA-01 through UA-03), Ecosystem Map As-Is (SM-AS — all four layers) |
| **Who you need** | Full project team (validation), Data Science lead (domain accuracy) |
| **What consumes your output** | JN-TB (redesigned journeys), FD (feature extraction) |

---

### Journey JN-AS-01: Producer Evaluating Track Before Release

**Archetype:** UA-01 — The Data-Driven Producer
**Trigger:** Producer has finished a track mix and must decide whether to invest in promotion before release date.

#### Happy Path

| Step # | User Action | Channel / Resource (ref) | System Response | Emotional State | Time Est. | Pain Point? | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | Producer exports the final .mp3 from the DAW | RES-05 (DAW) | File saved to local disk | Neutral | 5 min | No | New audio file created |
| 2 | Producer shares the file with a trusted peer for informal feedback | RES-04 (Email + file sharing) | Peer receives file; no structured format | Uncertain | 1–2 days (wait) | Yes — no structured intake | Informal opinion received |
| 3 | Producer listens to reference tracks on Spotify for Artists to compare | RES-01 (Spotify for Artists) | Displays post-release streams of reference tracks | Frustrated | 30 min | Yes — data is post-release only | None |
| 4 | Producer makes a subjective judgment call on release readiness | RES-06 (Industry experience) | No system response — internal decision | Anxious | 1 hour | Yes — decision is not data-backed | Go/no-go for release |
| 5 | If confident: producer sends track to distributor for release | CH-02 via RL-05 | Distributor confirms submission | Relieved (but uncertain) | 2–3 days | No | Release submitted |

#### Failure Paths

| Failure ID | At Step | Scenario | Impact | Workaround | Time Cost |
|---|---|---|---|---|---|
| FP-AS-01-A | Step 2 | Peer feedback is contradictory or unavailable | Producer has no basis for decision | Waits longer or decides alone | +2–5 days |
| FP-AS-01-B | Step 3 | Reference tracks are from different genres; comparison is not meaningful | False confidence or paralysis | Searches for closer references manually | +1–2 hours |
| FP-AS-01-C | Step 4 | Producer overestimates potential and invests in promotion that yields low returns | Financial loss | None — consequence discovered post-release | Post-release |

#### Summary Metrics

| Metric | Value |
|---|---|
| Total time (happy path) | 2–4 days |
| Steps with high pain | Steps 2, 3, 4 |
| Key bottleneck | Waiting for informal peer feedback (Step 2) |
| Decision confidence | Low — subjective throughout |
| Data used | None that predicts popularity; only post-release analytics available |

---

### Journey JN-AS-02: Independent Artist Assessing Song Before Promotion

**Archetype:** UA-02 — The Independent Artist Exploring Streaming
**Trigger:** Artist has a finished track and a limited budget; needs to decide whether to pay for promotional campaigns before release.

#### Happy Path

| Step # | User Action | Channel / Resource (ref) | System Response | Emotional State | Time Est. | Pain Point? | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | Artist listens to the finished track repeatedly and self-assesses its quality | RES-05 (DAW playback) | No external signal | Optimistic but uncertain | 1–2 hours | No | None |
| 2 | Artist posts a snippet on social media to gauge audience reaction | RES-04 (social media — informal channel) | Informal likes/comments | Anxious — waiting for validation | 1–3 days | Yes — not predictive of streams | Informal feedback received |
| 3 | Artist searches Spotify for tracks with a similar sound to assess how they performed | RES-01 (Spotify for Artists — limited public data) | Stream counts not fully visible for all tracks | Frustrated — data is incomplete | 30–60 min | Yes — reference data is opaque | None |
| 4 | Artist decides whether to invest in promotion based on self-assessment and social signals | RES-06 (personal intuition) | None | Resigned — decision made without confidence | 1 hour | Yes | Promotion decision taken |

#### Failure Paths

| Failure ID | At Step | Scenario | Impact | Workaround | Time Cost |
|---|---|---|---|---|---|
| FP-AS-02-A | Step 2 | Social media algorithm limits organic reach; snippet receives no engagement | Artist misinterprets silence as low potential | Tries again on a different platform | +1–2 days |
| FP-AS-02-B | Step 3 | Reference tracks not comparable — different artist size or release context | Misleading self-assessment | None effective | +30 min wasted |

#### Summary Metrics

| Metric | Value |
|---|---|
| Total time (happy path) | 2–5 days |
| Steps with high pain | Steps 2, 3, 4 |
| Key bottleneck | Waiting for social media feedback as a proxy for quality |
| Decision confidence | Very low — entirely social-signal dependent |
| Data used | No audio-feature data; only social engagement proxies |

---

### Journey JN-AS-03: A&R Analyst Evaluating Submission Pipeline

**Archetype:** UA-03 — The A&R Catalog Analyst
**Trigger:** A batch of new track submissions has arrived; analyst must filter and recommend which to escalate to the Senior A&R Director.

#### Happy Path

| Step # | User Action | Channel / Resource (ref) | System Response | Emotional State | Time Est. | Pain Point? | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | Analyst receives submission package via email (audio file + artist bio) | RES-04 (CH-01) | Email received; no structured intake | Neutral | 10 min per track | No | Queue of files to review |
| 2 | Analyst listens to each track using a media player | RES-03 (Subjective listening) | None | Focused | 3–5 min per track | Yes — slow at scale | Informal mental note |
| 3 | Analyst cross-checks artist's streaming metrics on Chartmetric | RES-02 (Chartmetric) | Historical streams and follower data shown | Analytical | 10 min per artist | Yes — no pre-release signal | Follower/stream history noted |
| 4 | Analyst scores tracks informally in a spreadsheet | RES-04 (manual spreadsheet) | None — manual input | Frustrated | 15 min per track | Yes — no standardized criteria | Spreadsheet updated |
| 5 | Analyst prepares recommendation memo for Senior A&R Director | RES-04 (CH-03) | None | Anxious — subjective output | 2–3 hours | Yes — lacks quantitative backing | Memo delivered |

#### Failure Paths

| Failure ID | At Step | Scenario | Impact | Workaround | Time Cost |
|---|---|---|---|---|---|
| FP-AS-03-A | Step 2 | Large batch (20+ submissions) makes thorough listening impractical | Analyst applies even less rigor to each track | Skims or delegates to junior staff | Quality degraded |
| FP-AS-03-B | Step 5 | Senior A&R Director challenges the recommendation without data | Analyst cannot defend the selection objectively | Relies on genre trends or artist fame as proxies | +1–2 days of rework |

#### Summary Metrics

| Metric | Value |
|---|---|
| Total time per track | 40–60 min |
| Total time for 10-track batch | 7–10 hours |
| Steps with high pain | Steps 2, 3, 4, 5 |
| Key bottleneck | Subjective listening as the only evaluation method |
| Decision confidence | Low — no reproducible scoring framework |

---

## Module 4 : JN-TB — Journeys To-Be

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | JN-TB |
| **What you are producing** | Redesigned journeys using the Music Popularity Predictor. Every changed step quantifies improvement vs. as-is. Failure paths show graceful degradation (e.g., ReccoBeats API unavailability, model limitations disclosed). |
| **Time-box** | 1–1.5 days |
| **You need before starting** | JN-AS, SM-AS (as-is ecosystem as contrast), PI-02 (three interaction routes), PI-06 (risks and constraints) |
| **Who you need** | Full project team; Backend lead (failure paths and API behavior); Data Science lead (model limitations for trust messaging) |

---

### Journey JN-TB-01: Producer Evaluating Track via Audio Upload

**Archetype:** UA-01 — The Data-Driven Producer
**Trigger:** Producer exports a finished .mp3 and opens the Music Popularity Predictor to assess its commercial potential before committing to promotion.

#### Redesigned Happy Path

| Step # | User Action | Channel / Resource | System Response | Emotional State | Time Est. | Improvement vs. As-Is | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | Producer opens the web application and selects "Process Audio" | React web interface | Landing page displays three flow options clearly | Confident — clear entry point | 1 min | Eliminates 2-day wait for informal peer feedback (JN-AS-01, Step 2) | None |
| 2 | Producer uploads the .mp3 file and optionally selects genre from a dropdown | React — POST /extract-features | File accepted; processing indicator shown | Focused | 1–2 min | Eliminates manual reference track search (JN-AS-01, Step 3) | Audio file sent to backend |
| 3 | Backend sends audio to ReccoBeats API; librosa fallback activates if API is unavailable | FastAPI — ReccoBeats API / librosa | Features extracted (danceability, energy, tempo, duration, etc.) | [Backend process] | 10–30 sec | Automated feature extraction replaces subjective listening | 13 audio features computed |
| 4 | LightGBM model returns a predicted popularity score (0–100) | FastAPI — LightGBM model | Popularity score displayed prominently on results screen | Relieved — has a signal | < 5 sec | Replaces subjective judgment call (JN-AS-01, Step 4) | Popularity prediction generated |
| 5 | Results screen displays SHAP-informed feature importance: top 3 factors driving the score | React results display | Top contributors shown (e.g., "Duration is reducing your score"; "Danceability is increasing your score") | Informed — understands the "why" | Immediate | New capability — did not exist in as-is | Feature importance surfaced |
| 6 | Producer optionally navigates to Optimization Wizard to simulate parameter adjustments | React — POST /wizard-optimize | Wizard accepts locked and free parameters; returns configuration that maximizes predicted popularity | Engaged — takes action | 3–5 min | New capability: replaces trial-and-error re-mixing | Optimized parameter set returned |
| 7 | Producer opens the AI Preview window to generate an audio sample from the optimized configuration | React — GenAI Preview Window | Window displays optimized parameters and a CTA to generate synthetic audio | Curious — wants to hear the recommendation | 1 min | New capability: transforms abstract optimization results into an audible experience | Preview request prepared |
| 8 | Producer listens to the generated preview and makes an informed release/revision decision | React audio player + GenAI backend | Playable audio preview returned with replay/regenerate options | Confident — decision backed by data and sound | 2–4 min | New capability: bridges model output and creative validation | Go/no-go decision with supporting data and synthetic preview |

#### Failure Paths

| Failure ID | At Step | Scenario | System Behavior | User Experience | Time Cost |
|---|---|---|---|---|---|
| FP-TB-01-A | Step 3 | ReccoBeats API is unavailable | Backend activates librosa fallback automatically; processing continues | User sees "Analysis in progress" — no error unless both paths fail | +10–20 sec |
| FP-TB-01-B | Step 3 | Both ReccoBeats and librosa fail | Backend returns: "Audio processing failed. Please try again or use the Spotify ID route." | User redirected to alternative flow | ~2 min restart |
| FP-TB-01-C | Step 4 | User interprets the score as a guarantee | Results screen includes a disclosed limitation notice: "This score is an estimate based on audio features only. Factors such as marketing, artist exposure, and playlist placement are not included. R² = 0.39." | User calibrates expectations appropriately | None — mitigated by UI |

#### Improvement Delta

| Metric | As-Is (JN-AS-01) | To-Be (JN-TB-01) | Delta |
|---|---|---|---|
| Total time to decision | 2–4 days | < 15 minutes | -99% |
| Decision confidence | Low — subjective | Moderate — data-backed with acknowledged limitations | Qualitative improvement |
| Feature explanation available | No | Yes — SHAP-informed top 3 factors | New capability |
| Optimization suggestions | No | Yes — wizard returns max-popularity parameter set | New capability |
| Ability to hear optimized recommendation | No | Yes — AI-generated preview from optimized parameters | New capability |

---

### Journey JN-TB-02: Independent Artist Searching by Spotify ID

**Archetype:** UA-02 — The Independent Artist Exploring Streaming
**Trigger:** Artist wants to check the predicted popularity of a track already on Spotify without needing to upload an audio file.

#### Redesigned Happy Path

| Step # | User Action | Channel / Resource | System Response | Emotional State | Time Est. | Improvement vs. As-Is | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | Artist opens the web application and selects "Search by Spotify ID" | React web interface | Spotify ID input field displayed with guidance text | Confident — clear entry point | 1 min | Eliminates need for manual social media benchmarking (JN-AS-02, Steps 2–3) | None |
| 2 | Artist enters the Spotify track ID (copied from Spotify app or web player) | React — GET /search-track | Backend resolves the Spotify ID to a ReccoBeats internal ID | Focused | 30 sec | Requires no file upload — lowers technical barrier for UA-02 | ID resolved |
| 3 | Backend retrieves audio features from ReccoBeats API by internal ID | FastAPI — GET /track-features | 13 audio features returned and completed if missing | [Backend process] | 10–20 sec | Automated feature retrieval replaces manual Spotify search (JN-AS-02, Step 3) | Feature set assembled |
| 4 | LightGBM model predicts popularity score | FastAPI — LightGBM model | Score returned to React frontend | Relieved — has objective signal | < 5 sec | Replaces social media likes as popularity proxy (JN-AS-02, Step 2) | Score generated |
| 5 | Results screen displays score + top contributing features in plain language | React results display | Feature explanations shown without jargon (e.g., "High danceability is positively impacting this track's predicted popularity") | Informed — no technical background required | Immediate | New capability — plain-language explanation designed for UA-02 | None |

#### Failure Paths

| Failure ID | At Step | Scenario | System Behavior | User Experience | Time Cost |
|---|---|---|---|---|---|
| FP-TB-02-A | Step 2 | Spotify ID is invalid or not found in ReccoBeats | Backend returns: "Track ID not found. Please verify the Spotify ID or try uploading the audio file." | User redirected to Upload flow | ~2 min |
| FP-TB-02-B | Step 3 | ReccoBeats API returns empty features | Backend returns an error with a clear message and suggests the Upload route | User not left in a dead-end state | ~2 min |

#### Improvement Delta

| Metric | As-Is (JN-AS-02) | To-Be (JN-TB-02) | Delta |
|---|---|---|---|
| Total time to assessment | 2–5 days | < 5 minutes | -99% |
| Technical skill required | Low (social media) but non-predictive | Low (copy Spotify ID) and predictive | Barrier maintained; quality improved |
| External validation needed | Yes (social media) | No | Dependency eliminated |

---

### Journey JN-TB-03: A&R Analyst Using the Optimization Wizard

**Archetype:** UA-03 — The A&R Catalog Analyst
**Trigger:** A&R analyst has retrieved a track's audio features via Spotify ID and wants to evaluate what parameter changes would maximize predicted popularity before drafting a recommendation.

#### Redesigned Happy Path

| Step # | User Action | Channel / Resource | System Response | Emotional State | Time Est. | Improvement vs. As-Is | Data Changed |
|---|---|---|---|---|---|---|---|
| 1 | After completing a Spotify ID lookup (JN-TB-02), analyst opens the Optimization Wizard | React — wizard panel | Wizard displays current feature values with lock toggles for each parameter | Focused | 1 min | Adds a capability that did not exist in as-is workflow | Feature set loaded |
| 2 | Analyst locks features that are non-negotiable for the track (e.g., genre, key) | React wizard interface | Locked features highlighted; free parameters available for optimization | Engaged | 2–3 min | Analyst controls constraints — no equivalent in as-is | Locked/free features set |
| 3 | Analyst submits optimization request | React — POST /wizard-optimize | Backend evaluates a grid of candidate values for free parameters and identifies the popularity-maximizing configuration | Anticipatory | 10–30 sec | Replaces manual "what if" mental modeling | Optimized parameter set computed |
| 4 | Results display optimized configuration vs. original, with delta score and optional AI preview generation | React results screen | Side-by-side comparison: original score vs. optimized score; changed parameters highlighted; option to generate synthetic preview | Informed | Immediate | New capability — quantified suggestion for production team plus audible preview | Recommendation artifact generated |
| 5 | Analyst optionally generates and listens to an AI preview based on the optimized configuration | React audio player + GenAI backend | Synthetic preview returned for internal evaluation or presentation support | Confident — has quantitative and audible backing | 2–4 min | New capability: strengthens internal communication with a demonstrable output | Memo support enhanced |
| 6 | Analyst uses the output as supporting evidence in the A&R recommendation memo | External (report writing) | None — user exports/copies results manually | Confident — has quantitative backing | 15–20 min | Memo now includes reproducible, data-backed analysis plus optional preview evidence | Memo quality improved |

#### Improvement Delta

| Metric | As-Is (JN-AS-03) | To-Be (JN-TB-03) | Delta |
|---|---|---|---|
| Time per track evaluation | 40–60 min | 10–15 min | -75% |
| Scoring reproducibility | None — subjective | Yes — same inputs yield same score | Fully reproducible |
| Optimization suggestions | None | Yes — wizard returns max-popularity configuration | New capability |
| Defensibility of recommendation | Low — subjective memo | Moderate — score + feature importance + optimization delta | Qualitative improvement |

---

## Module 5 : PC — Product Conceptualization

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | PC |
| **What you are producing** | 1-page product concept: what the Music Popularity Predictor IS, for whom, why it is different, and how success is measured. North star for all feature work in FD. |
| **Time-box** | 0.5 days |
| **You need before starting** | All UA archetypes + all JN-TB journeys + SM-AS ecosystem map |
| **Who you need** | Full project team alignment |

---

### Product Concept Card

| Section | Content |
|---------|---------|
| **Product Name** | Music Popularity Predictor — _¿Qué tan popular será tu canción?_ |
| **One-Liner** | Predict your song's Spotify popularity from audio features before you release it. |
| **Problem Statement** | Music producers, independent artists, and A&R analysts currently rely on intuition, peer feedback, and post-release data to evaluate a song's commercial potential — methods that are slow, inconsistent, and inaccessible to smaller creators. Over 120,000 songs are uploaded to Spotify daily, yet no accessible pre-release prediction tool exists for non-enterprise users. The Music Popularity Predictor fills this gap by estimating popularity from audio features before a track is released, giving creators an objective signal to inform their decisions. |
| **Target Users** | Primary: UA-01 (The Data-Driven Producer), UA-02 (The Independent Artist Exploring Streaming); Secondary: UA-03 (The A&R Catalog Analyst) |
| **Core Value Proposition** | The only accessible, audio-feature-based pre-release popularity predictor that (a) works from either an .mp3 upload or a Spotify ID, (b) explains the score through SHAP-informed feature importance in plain language, (c) offers an optimization wizard to simulate which audio changes would maximize predicted popularity, and (d) allows users to generate an AI audio preview from the optimized parameter set so they can hear a synthetic approximation of the recommended song profile — all without requiring analytics expertise or a platform subscription. |
| **Key Journey Improvements** | (1) Decision time for producers reduced from 2–4 days to under 15 minutes (JN-TB-01 vs. JN-AS-01); (2) Time per track evaluation for A&R analysts reduced by ~75%, from 40–60 min to 10–15 min (JN-TB-03 vs. JN-AS-03); (3) Independent artists gain a data-backed signal replacing social media likes as the sole popularity proxy, with zero technical skill required to operate (JN-TB-02 vs. JN-AS-02) |
| **AI-Native Angle** | The core of the product is a LightGBM gradient boosting model trained on the _Spotify 1 Million Tracks_ dataset (~333K records, 30 genres, 2000–2023), achieving R² = 0.389 on the test set. SHAP values expose feature contributions per prediction. The optimization wizard iterates over a candidate grid to return the configuration that maximizes the model's predicted score. In addition, a GenAI audio synthesis module can transform the optimized parameter set into a short synthetic audio preview, allowing users to audibly explore the musical profile associated with the recommended configuration. ReccoBeats API + librosa provide the audio feature extraction pipeline. |
| **Success Metrics** | (1) 100% of three core user flows (audio upload, Spotify ID search, parameter optimization) operational end-to-end without errors; (2) Model: LightGBM MSE = 152.29, RMSE = 12.34, MAE = 9.72, R² = 0.389; (3) UA-01 and UA-02 can complete a full song analysis in under 15 minutes; (4) Results screen includes disclosed model limitations (R² ≈ 0.39; excluded factors: marketing, artist fame, playlist placement); (5) Prototype delivered and demonstrated by 19/11/2025 |
| **Scope Boundaries** | This product does NOT: (1) integrate with live Spotify production systems for real-time or automated catalog analysis; (2) provide batch analysis of multiple tracks in a single session (deferred to roadmap); (3) incorporate external signals absent from the dataset (social media, artist follower counts, marketing spend, playlist placements); (4) store or manage a user's song history or analysis results across sessions; (5) predict popularity for genres outside the top 30 by average popularity in the training dataset |
| **Risks & Assumptions** | (1) The model's R² ≈ 0.39 means ~61% of popularity variance is driven by factors not in the dataset (R-01, R-02) — the product must communicate this clearly to avoid misuse as a guarantee; (2) The product assumes ReccoBeats API availability; librosa fallback mitigates but does not eliminate this risk (R-03, HC-04); (3) Genre bias in the model (trained on top 30 genres only) may produce less reliable results for minority genres — this must be disclosed in the UI (R-05) |

---

## Module 6 : FD — Feature Definition

### Definition

| Attribute | Value |
|-----------|-------|
| **Module ID** | FD |
| **What you are producing** | A structured feature backlog: each feature is a discrete, buildable unit of user-visible value with user story, acceptance criteria, AI involvement, complexity estimate, and full traceability to journeys and archetypes. |
| **Time-box** | 1 day |
| **You need before starting** | PC (product concept and scope), JN-TB-01 through JN-TB-03 (journeys to decompose), SM-AS AI resources (LightGBM, FastAPI, ReccoBeats, librosa) |
| **Who you need** | Backend lead (technical fields + complexity), Data Science lead (AI involvement fields), Frontend lead (UX fields) |
| **What consumes your output** | FE (features get scored), FP (scored features get prioritized) |

---

### Feature Card FT-001

| Field | Content |
|---|---|
| **Feature ID** | FT-001 |
| **Feature Name** | Audio File Upload and Feature Extraction |
| **Description** | Allows the user to upload an .mp3 file and an optional genre selection through the web interface. The backend sends the file to the ReccoBeats API for audio feature extraction; if the API is unavailable, librosa performs local extraction as a fallback. The extracted features are returned to the frontend for display and downstream prediction. |
| **Archetype(s) Served** | UA-01 (Primary), UA-02 (Secondary) |
| **Journey(s) Impacted** | JN-TB-01, Steps 1–3 |
| **User Story** | As a music producer (UA-01), I want to upload my finished .mp3 and have its audio features extracted automatically so that I can obtain a popularity prediction without manually computing any technical values. |
| **Acceptance Criteria** | 1. Given a valid .mp3 file, when the user uploads it via the interface, then the backend returns all 13 required audio features within 60 seconds. 2. Given a ReccoBeats API error, when the upload is processed, then librosa fallback activates automatically and the user receives features without seeing an error. 3. Given an invalid or corrupted file, when the user uploads it, then the interface displays a clear error message with instructions to retry or use the Spotify ID route. |
| **AI Involvement** | AI-Assisted (audio feature extraction feeds the AI prediction in FT-003) |
| **AI Model Tier** | N/A for this feature (feature extraction, not prediction) |
| **Data Dependencies** | .mp3 audio file (user-provided); ReccoBeats API audio-features endpoint; librosa library |
| **Integration Dependencies** | ReccoBeats API (POST /analysis/audio-features); librosa (local fallback) |
| **Estimated Complexity** | M (3–5 days) |
| **Scope Boundary** | Does not support batch upload of multiple files; does not store audio files after processing; does not support formats other than .mp3 |

---

### Feature Card FT-002

| Field | Content |
|---|---|
| **Feature ID** | FT-002 |
| **Feature Name** | Spotify Track ID Lookup and Feature Retrieval |
| **Description** | Allows the user to enter one or more Spotify track IDs. The backend resolves each Spotify ID to a ReccoBeats internal ID, retrieves the track's audio features, and completes any missing fields (e.g., time_signature, duration_ms) with a secondary API call. The features are returned to the frontend ready for prediction. |
| **Archetype(s) Served** | UA-02 (Primary), UA-03 (Primary) |
| **Journey(s) Impacted** | JN-TB-02, Steps 1–3; JN-TB-03, Step 1 |
| **User Story** | As an independent artist (UA-02), I want to enter a Spotify track ID so that I can get a popularity prediction for any track on Spotify without needing to download and upload its audio file. |
| **Acceptance Criteria** | 1. Given a valid Spotify track ID, when the user submits it, then the backend returns a complete feature set within 30 seconds. 2. Given an invalid or unrecognized Spotify ID, when the user submits it, then the interface displays a clear error message and suggests the audio upload route as an alternative. 3. Given a feature set with missing fields from ReccoBeats, when features are returned, then the backend completes missing fields via a secondary call before returning the result to the frontend. |
| **AI Involvement** | AI-Assisted (feature retrieval feeds the AI prediction in FT-003) |
| **AI Model Tier** | N/A for this feature |
| **Data Dependencies** | Spotify track ID (user-provided); ReccoBeats API (search and track-features endpoints) |
| **Integration Dependencies** | ReccoBeats API (GET /search-track; GET /track/{id}/audio-features; GET /track/{id}) |
| **Estimated Complexity** | M (3–5 days) |
| **Scope Boundary** | Does not support batch ID entry for multiple tracks simultaneously (out of scope per PI-05); does not retrieve lyrics, metadata, or artist information |

---

### Feature Card FT-003

| Field | Content |
|---|---|
| **Feature ID** | FT-003 |
| **Feature Name** | Popularity Score Prediction |
| **Description** | Once audio features are available (from FT-001 or FT-002), the LightGBM model receives the feature vector (13 numeric features + one-hot encoded genre), applies StandardScaler and OneHotEncoder transformations, and returns a predicted popularity score on a 0–100 scale. The score is displayed prominently on the results screen. |
| **Archetype(s) Served** | UA-01, UA-02, UA-03 |
| **Journey(s) Impacted** | JN-TB-01 Step 4; JN-TB-02 Step 4; JN-TB-03 (via JN-TB-02 Steps 1–3) |
| **User Story** | As a music producer (UA-01), I want the application to predict a popularity score for my song so that I have an objective signal to compare against my intuition before deciding on promotional investment. |
| **Acceptance Criteria** | 1. Given a complete feature vector, when the prediction endpoint is called, then the model returns a score between 0 and 100 in under 5 seconds. 2. Given the same input features submitted twice, when the prediction runs both times, then both results return the same score (deterministic output). 3. Given any prediction result, when the results screen is displayed, then a model limitation notice is visible: "This score is an estimate based on audio features only. R² = 0.389. Factors such as marketing, artist exposure, and playlist placement are not included." |
| **AI Involvement** | AI-Powered |
| **AI Model Tier** | Tier 2 (Balanced) — LightGBM; MSE = 152.29, R² = 0.389 |
| **Data Dependencies** | 13 audio features (from FT-001 or FT-002); genre label; pre-trained LightGBM model artifact; StandardScaler and OneHotEncoder pipeline artifacts |
| **Integration Dependencies** | None external — model runs in-process on FastAPI backend |
| **Estimated Complexity** | S (1–2 days — model already trained; integration work only) |
| **Scope Boundary** | Does not provide a confidence interval around the prediction; does not predict performance on platforms other than Spotify |

---

### Feature Card FT-004

| Field | Content |
|---|---|
| **Feature ID** | FT-004 |
| **Feature Name** | SHAP-Informed Feature Importance Display |
| **Description** | After the prediction score is returned, the results screen displays the top contributing audio features to the score in plain language, derived from SHAP value analysis of the LightGBM model. Each factor is labeled positively or negatively (e.g., "High danceability is increasing your predicted popularity" / "Long duration is reducing your predicted popularity"). |
| **Archetype(s) Served** | UA-01 (Primary), UA-02 (Primary), UA-03 (Secondary) |
| **Journey(s) Impacted** | JN-TB-01 Step 5; JN-TB-02 Step 5 |
| **User Story** | As a music producer (UA-01), I want to see which audio features are most responsible for my song's predicted score so that I know what to adjust if I want to improve its commercial potential. |
| **Acceptance Criteria** | 1. Given a prediction result, when the results screen loads, then the top 3 contributing features are displayed with a positive or negative direction label. 2. Given a feature with a negative contribution, when it is displayed, then the label uses plain language accessible to a non-technical user (no raw SHAP scores shown). 3. Given a prediction for a track with genre "pop," when the results screen displays genre contribution, then the genre label is included among the displayed contributors if it ranks in the top 3 SHAP values. |
| **AI Involvement** | AI-Powered (SHAP values computed from LightGBM model) |
| **AI Model Tier** | Tier 2 (Balanced) — same model as FT-003 |
| **Data Dependencies** | Pre-computed SHAP value distributions (from PDF Section 3.3.5); real-time SHAP inference at prediction time |
| **Integration Dependencies** | SHAP Python library; LightGBM model (same instance as FT-003) |
| **Estimated Complexity** | M (3–5 days) |
| **Scope Boundary** | Displays top 3 features only; does not show raw SHAP values or probability plots; does not allow users to reorder or filter feature importance interactively |

---

### Feature Card FT-005

| Field | Content |
|---|---|
| **Feature ID** | FT-005 |
| **Feature Name** | Parameter Optimization Wizard |
| **Description** | Allows the user to submit a base audio feature set with some features locked (fixed) and others free for optimization. The backend evaluates a grid of candidate values for each free parameter, runs the LightGBM model for each combination, and returns the configuration that maximizes the predicted popularity score alongside the original and achieved scores. |
| **Archetype(s) Served** | UA-01 (Primary), UA-03 (Secondary) |
| **Journey(s) Impacted** | JN-TB-01 Steps 6–7; JN-TB-03 Steps 2–4 |
| **User Story** | As a music producer (UA-01), I want to simulate changes to my song's audio parameters so that I can understand what production adjustments would give it the highest possible predicted popularity score. |
| **Acceptance Criteria** | 1. Given a feature set with at least one free parameter, when the optimization request is submitted, then the backend returns an optimized configuration and the resulting score within 60 seconds. 2. Given locked features, when the optimization runs, then locked feature values remain unchanged in the returned configuration. 3. Given the optimization result, when the results screen displays it, then the original score, optimized score, and changed parameter values are all shown side by side. |
| **AI Involvement** | AI-Powered (LightGBM model called iteratively over candidate grid) |
| **AI Model Tier** | Tier 2 (Balanced) — same model as FT-003 |
| **Data Dependencies** | Feature set from FT-001 or FT-002; locked/free feature selection (user-provided); pre-trained LightGBM model |
| **Integration Dependencies** | None external — runs in-process on FastAPI backend (POST /wizard-optimize) |
| **Estimated Complexity** | M (3–5 days) |
| **Scope Boundary** | Optimizes one song at a time; does not support multi-objective optimization; grid search only — does not use gradient-based optimization |

---

### Feature Card FT-006

| Field | Content |
|---|---|
| **Feature ID** | FT-006 |
| **Feature Name** | GenAI Audio Preview from Optimized Parameters |
| **Description** | Allows the user to generate a short AI audio preview from the optimized feature set returned by the Parameter Optimization Wizard. Once the backend identifies the configuration that maximizes predicted popularity, the user can open a dedicated preview window and request a synthetic audio sample that approximates the recommended musical profile (e.g., tempo, energy, danceability, valence, duration, and genre). |
| **Archetype(s) Served** | UA-01 (Primary), UA-03 (Secondary), UA-02 (Secondary) |
| **Journey(s) Impacted** | JN-TB-01 Step 6–7; JN-TB-03 Step 4–5 |
| **User Story** | As a music producer (UA-01), I want to generate an AI audio preview from the optimized song parameters so that I can hear an approximate sonic interpretation of the recommended configuration before deciding whether to modify my production. |
| **Acceptance Criteria** | 1. Given an optimization result, when the user clicks "Generate AI Preview", then the system opens a preview window with the optimized parameters preloaded. 2. Given a valid optimized parameter set, when the generation request is submitted, then the backend returns a playable synthetic audio preview within an acceptable waiting time. 3. Given a generation failure, when the process cannot complete, then the interface displays a clear error message and preserves the optimized parameter set so the user can retry. 4. Given a generated preview, when the result is displayed, then the user can play, pause, and regenerate the preview without rerunning the full optimization flow. |
| **AI Involvement** | AI-Powered |
| **AI Model Tier** | Tier 3 (Generative AI) — audio synthesis model |
| **Data Dependencies** | Optimized feature set from FT-005; genre label; synthesis prompt template or parameter-to-audio mapping schema |
| **Integration Dependencies** | GenAI audio generation service or model endpoint; FastAPI backend orchestration; frontend audio player component |
| **Estimated Complexity** | L (5–10 days) |
| **Scope Boundary** | Generates a short preview only; does not produce a full commercial-quality mastered track; does not replace music production software; does not guarantee that the generated audio exactly matches the original song structure or artistic intent. |

### Feature Coverage Matrix

| Feature ID | Feature Name | JN-TB-01 Steps | JN-TB-02 Steps | JN-TB-03 Steps |
|---|---|---|---|---|
| FT-001 | Audio File Upload and Feature Extraction | 1–3 | — | — |
| FT-002 | Spotify Track ID Lookup and Feature Retrieval | — | 1–3 | 1 |
| FT-003 | Popularity Score Prediction | 4 | 4 | via FT-002 |
| FT-004 | SHAP-Informed Feature Importance Display | 5 | 5 | via FT-003 |
| FT-005 | Parameter Optimization Wizard | 6–7 | — | 2–4 |

_All journey steps are covered by at least one feature. No uncovered steps identified._

---

## Phase 1 Quality Gate

| # | Gate Item | Status |
|---|---|---|
| 1 | UA: 3–5 archetypes produced and prioritized | [x] — 3 archetypes (UA-01 Primary, UA-02 Primary, UA-03 Secondary) |
| 2 | SM-AS: All 4 layers documented (roles, channels, flows, resources) | [x] |
| 3 | JN-AS: At least 1 journey per primary archetype with failure paths | [x] — JN-AS-01, JN-AS-02, JN-AS-03 |
| 4 | JN-TB: Redesigned journey per flow with improvement delta table | [x] — JN-TB-01, JN-TB-02, JN-TB-03 |
| 5 | PC: Product Concept Card complete with quantified success metrics | [x] |
| 6 | FD: Feature backlog covers all JN-TB journey steps (no gaps) | [x] — FT-001 through FT-005; coverage matrix confirms full coverage |
| 7 | Every feature traceable to at least one archetype and one journey | [x] |
| 8 | AI involvement documented for every feature | [x] |
| 9 | Model limitations (R² ≈ 0.39) reflected in trust expectations, journeys, and product concept | [x] |
| 10 | Out-of-scope items from PI-05 respected (no batch processing, no social signals) | [x] |

---

## Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Project Lead | | | |
| Data Science Lead | | | |
| Backend Lead | | | |
| Frontend / UI-UX Lead | | | |

---

Version History:

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 14/03/2026 | HPWSB | Phase 1 Service Design produced from Phase 0 (MPP-PH0-INTAKE-v1.0) and ProyectoIntegrador2 PDF report |

---

_MPP-PH1-SD-v1.0 | Music Popularity Predictor Project | Universidad EAFIT_
