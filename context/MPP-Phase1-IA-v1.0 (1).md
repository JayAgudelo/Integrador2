# Module 7 : IA — Information Architecture

**Music Popularity Predictor — _¿Qué tan popular será tu canción?_**

| Attribute | Value |
|-----------|-------|
| **Document ID** | MPP-PH1-IA-v1.0 |
| **Project** | MPP-2025-01 — Music Popularity Predictor |
| **Depends on** | MPP-PH0-INTAKE-v1.0 (Phase 0) · MPP-PH1-SD-v1.0 (Phase 1: UA, SM-AS, JN-AS, JN-TB, PC, FD) |
| **Date** | 14/03/2026 |
| **Version** | 1.0 |

---

## Step IA-1 — Navigation Candidates Extracted from Features

Before building the hierarchy, all features and their natural user goals are grouped:

| Feature ID | Feature Name | User Goal | Natural Grouping |
|---|---|---|---|
| FT-001 | Audio File Upload and Feature Extraction | "I have an .mp3 — analyze it" | Analysis Entry — Upload route |
| FT-002 | Spotify Track ID Lookup and Feature Retrieval | "I have a Spotify ID — analyze it" | Analysis Entry — ID route |
| FT-003 | Popularity Score Prediction | "See the predicted score" | Results |
| FT-004 | SHAP-Informed Feature Importance Display | "Understand why the score is what it is" | Results |
| FT-005 | Parameter Optimization Wizard | "Simulate changes to improve the score" | Optimization |

Additional non-feature screens required per template rules: Landing / Home, Onboarding / How It Works, Error, Empty State, Help & Limitations.

---

## IA-2 — Navigation Model

### Navigation Model — Music Popularity Predictor

#### Level 0: Product Shell

- **Global navigation:** always-visible top bar with product name, links to Home and Help
- **Authentication boundary:** none — public access (PI-02: Access Model = Public, academic project environment)
- **Role-based access gates:** none in v1 — all three archetypes access the same interface

---

#### Level 1: Primary Sections

| Section ID | Section Name | Purpose | Primary Archetype(s) | Features Hosted | Label in UI |
|---|---|---|---|---|---|
| NAV-01 | Home | Entry point — presents the three analysis routes and orients first-time users | UA-01, UA-02, UA-03 | — (routing hub, no features; links to NAV-02 and NAV-03) | Home |
| NAV-02 | Analyze a Song | Core analysis flows — user chooses Upload or Spotify ID route to obtain audio features and a prediction | UA-01 (Upload), UA-02 (ID search), UA-03 (ID search) | FT-001, FT-002, FT-003, FT-004 | Analyze |
| NAV-03 | Optimize Parameters | Optimization wizard — user starts from a feature set and simulates adjustments to maximize predicted popularity | UA-01, UA-03 | FT-005 | Optimize |
| NAV-04 | How It Works | Onboarding / educational content — explains what the model does, its limitations (R² ≈ 0.39), and how to interpret results | UA-02 (primary), UA-01, UA-03 | — (content screen) | How It Works |
| NAV-05 | Help & Limitations | Persistent reference — model limitations, data scope, FAQ, link to academic report context | UA-01, UA-02, UA-03 | — (content screen) | Help |

_5 Level 1 sections — within the 4–7 rule for web applications._

---

#### Level 2: Sub-sections within NAV-02 (Analyze a Song)

| Sub-section ID | Parent | Sub-section Name | Purpose | Features Hosted |
|---|---|---|---|---|
| NAV-02.1 | NAV-02 | Upload Audio | Upload an .mp3 file and optional genre to extract features and receive a prediction | FT-001, FT-003, FT-004 |
| NAV-02.2 | NAV-02 | Search by Spotify ID | Enter a Spotify track ID to retrieve features and receive a prediction | FT-002, FT-003, FT-004 |
| NAV-02.3 | NAV-02 | Results | Display the predicted popularity score, feature importance, and navigation options | FT-003, FT-004 |

_Note: FT-003 and FT-004 are structurally hosted in NAV-02.3 (Results). They are triggered from both NAV-02.1 and NAV-02.2 but their primary structural home is the Results sub-section._

---

#### Level 3: Detail Views

| Detail ID | Parent | Detail Name | Trigger | Features Hosted |
|---|---|---|---|---|
| NAV-03.1 | NAV-03 | Wizard Setup | User arrives from Results (post-prediction) or directly from Home via "Optimize" option; feature set pre-loaded or entered manually | FT-005 (input step) |
| NAV-03.2 | NAV-03 | Wizard Results | System returns the optimized parameter set after grid evaluation | FT-005 (output step) |

---

## IA-3 — Screen Inventory

### SCR-001 — Home / Landing

| Field | Content |
|---|---|
| **Screen ID** | SCR-001 |
| **Screen Name** | Home |
| **Navigation Path** | NAV-01 |
| **Screen Type** | Dashboard (routing hub) |
| **Primary Archetype** | UA-01, UA-02, UA-03 |
| **Features on Screen** | None — navigation hub only |
| **Key Data Objects** | None |
| **AI Interactions** | None |
| **Entry Points** | Direct URL; browser back from any screen; product logo click |
| **Exit Points** | → SCR-002 (Upload Audio) via "Upload .mp3" card; → SCR-003 (Spotify ID Search) via "Search by Spotify ID" card; → SCR-005 (Optimizer Setup) via "Optimize Parameters" card; → SCR-007 (How It Works) via "How It Works" link |
| **State Variations** | Default (all three flow options presented); first-time-use (onboarding banner visible, links to SCR-007) |
| **Estimated Complexity** | Low |

---

### SCR-002 — Upload Audio

| Field | Content |
|---|---|
| **Screen ID** | SCR-002 |
| **Screen Name** | Upload Audio |
| **Navigation Path** | NAV-02 > NAV-02.1 |
| **Screen Type** | Form |
| **Primary Archetype** | UA-01 (primary), UA-02 (secondary) |
| **Features on Screen** | FT-001 (Audio File Upload and Feature Extraction) |
| **Key Data Objects** | OBJ-001 (Audio File), OBJ-002 (Feature Set) |
| **AI Interactions** | AI-Assisted — ReccoBeats API extracts audio features from the uploaded .mp3; librosa fallback activates automatically if API is unavailable. Fallback state: "Processing locally — this may take a few extra seconds." |
| **Entry Points** | → from SCR-001 via "Upload .mp3" card; → from SCR-004 via "Try a different route" link (after Spotify ID error) |
| **Exit Points** | → SCR-004 (Results) after successful feature extraction + prediction; → SCR-008 (Error) if both ReccoBeats and librosa fail |
| **State Variations** | Default (empty form); loading (file uploaded, processing indicator active); error (invalid file type or corrupted .mp3 — inline error message with retry guidance); success (redirects to SCR-004) |
| **Estimated Complexity** | High (real-time processing, external API, local fallback, file handling) |

---

### SCR-003 — Search by Spotify ID

| Field | Content |
|---|---|
| **Screen ID** | SCR-003 |
| **Screen Name** | Search by Spotify ID |
| **Navigation Path** | NAV-02 > NAV-02.2 |
| **Screen Type** | Form |
| **Primary Archetype** | UA-02 (primary), UA-03 (primary) |
| **Features on Screen** | FT-002 (Spotify Track ID Lookup and Feature Retrieval) |
| **Key Data Objects** | OBJ-003 (Spotify Track Reference), OBJ-002 (Feature Set) |
| **AI Interactions** | AI-Assisted — ReccoBeats API resolves the Spotify ID to an internal ID and retrieves the feature set; missing fields completed via secondary call. Fallback state: "Track ID not found. Please verify the ID or try the Upload route." |
| **Entry Points** | → from SCR-001 via "Search by Spotify ID" card; → from SCR-002 via "Try a different route" link |
| **Exit Points** | → SCR-004 (Results) after successful feature retrieval + prediction; → SCR-008 (Error) if both ID resolution and feature retrieval fail |
| **State Variations** | Default (empty ID field with example format guidance); loading (ID submitted, resolving); error — invalid ID (inline error with suggestion to use Upload route); error — API failure (system error message with retry) |
| **Estimated Complexity** | High (external API calls, two-step resolution, field completion logic) |

---

### SCR-004 — Analysis Results

| Field | Content |
|---|---|
| **Screen ID** | SCR-004 |
| **Screen Name** | Analysis Results |
| **Navigation Path** | NAV-02 > NAV-02.3 |
| **Screen Type** | Detail |
| **Primary Archetype** | UA-01, UA-02, UA-03 |
| **Features on Screen** | FT-003 (Popularity Score Prediction), FT-004 (SHAP-Informed Feature Importance Display) |
| **Key Data Objects** | OBJ-002 (Feature Set), OBJ-004 (Prediction Result), OBJ-005 (Feature Importance) |
| **AI Interactions** | AI-Powered — LightGBM model returns predicted popularity score (0–100); SHAP values generate plain-language top-3 feature contributors. Fallback state: "Prediction unavailable. The model could not process this feature set. Please try again." Model limitation notice always displayed: "This score is an estimate based on audio features only. R² = 0.389. Factors such as marketing, artist exposure, and playlist placement are not included." |
| **Entry Points** | → from SCR-002 after successful audio processing; → from SCR-003 after successful Spotify ID lookup |
| **Exit Points** | → SCR-005 (Optimizer Setup) via "Optimize this song" CTA; → SCR-001 (Home) via "Analyze another song"; → SCR-008 (Help & Limitations) via "What does this score mean?" link |
| **State Variations** | Loading (prediction in progress — spinner with "Calculating popularity score…"); success (score + feature importance displayed); error (model failure — retry prompt); limitation banner always visible |
| **Estimated Complexity** | High (real-time AI inference, SHAP computation, plain-language rendering) |

---

### SCR-005 — Optimizer Setup

| Field | Content |
|---|---|
| **Screen ID** | SCR-005 |
| **Screen Name** | Optimizer Setup |
| **Navigation Path** | NAV-03 > NAV-03.1 |
| **Screen Type** | Wizard (step 1 of 2) |
| **Primary Archetype** | UA-01 (primary), UA-03 (secondary) |
| **Features on Screen** | FT-005 (Parameter Optimization Wizard — input step) |
| **Key Data Objects** | OBJ-002 (Feature Set), OBJ-006 (Optimization Request) |
| **AI Interactions** | None on this screen — user configures locks/free parameters before submitting to AI. |
| **Entry Points** | → from SCR-004 via "Optimize this song" CTA (feature set pre-populated from prior analysis); → from SCR-001 via "Optimize Parameters" card (user enters features manually) |
| **Exit Points** | → SCR-006 (Optimizer Results) after submitting the optimization request; → SCR-004 (Analysis Results) via "Back to Results" |
| **State Variations** | Pre-populated (feature set loaded from SCR-004); manual entry (user fills in features from scratch); loading (optimization running — grid evaluation in progress) |
| **Estimated Complexity** | Medium (form with lock/free toggles per feature, parameter grid display, pre-population logic) |

---

### SCR-006 — Optimizer Results

| Field | Content |
|---|---|
| **Screen ID** | SCR-006 |
| **Screen Name** | Optimizer Results |
| **Navigation Path** | NAV-03 > NAV-03.2 |
| **Screen Type** | Detail |
| **Primary Archetype** | UA-01 (primary), UA-03 (secondary) |
| **Features on Screen** | FT-005 (Parameter Optimization Wizard — output step) |
| **Key Data Objects** | OBJ-006 (Optimization Request), OBJ-007 (Optimization Result) |
| **AI Interactions** | AI-Powered — LightGBM model iterates over candidate grid; returns the configuration that maximizes predicted popularity. Fallback state: "Optimization could not complete. Please try with fewer free parameters or try again." |
| **Entry Points** | → from SCR-005 after optimization request submitted |
| **Exit Points** | → SCR-005 (Optimizer Setup) via "Adjust parameters" (re-run); → SCR-001 (Home) via "Analyze another song"; → SCR-002 or SCR-003 via "Analyze a new song" |
| **State Variations** | Loading (grid evaluation in progress — "Finding the best configuration…"); success (side-by-side: original score vs. optimized score, parameter deltas highlighted); error (optimization failure — fallback message with retry option) |
| **Estimated Complexity** | High (iterative AI inference, side-by-side comparison rendering, parameter delta highlighting) |

---

### SCR-007 — How It Works

| Field | Content |
|---|---|
| **Screen ID** | SCR-007 |
| **Screen Name** | How It Works |
| **Navigation Path** | NAV-04 |
| **Screen Type** | Detail (educational content) |
| **Primary Archetype** | UA-02 (primary — needs plain-language orientation), UA-01, UA-03 |
| **Features on Screen** | None — onboarding/educational content screen |
| **Key Data Objects** | None |
| **AI Interactions** | None |
| **Entry Points** | → from SCR-001 via "How It Works" link; → from global navigation |
| **Exit Points** | → SCR-001 (Home) via "Start analyzing" CTA; → SCR-002 or SCR-003 via direct action links |
| **State Variations** | Default (content displayed); no loading or error states |
| **Estimated Complexity** | Low (static content) |

---

### SCR-008 — Help & Limitations

| Field | Content |
|---|---|
| **Screen ID** | SCR-008 |
| **Screen Name** | Help & Limitations |
| **Navigation Path** | NAV-05 |
| **Screen Type** | Detail (reference content) |
| **Primary Archetype** | UA-01, UA-02, UA-03 |
| **Features on Screen** | None — reference screen |
| **Key Data Objects** | None |
| **AI Interactions** | None |
| **Entry Points** | → from SCR-004 via "What does this score mean?" link; → from global navigation; → from any error state via "Learn more" link |
| **Exit Points** | → SCR-001 (Home); → previous screen via browser back |
| **State Variations** | Default only |
| **Estimated Complexity** | Low (static content) |

---

### SCR-009 — System Error

| Field | Content |
|---|---|
| **Screen ID** | SCR-009 |
| **Screen Name** | System Error |
| **Navigation Path** | Global (triggered from any flow) |
| **Screen Type** | Error State |
| **Primary Archetype** | All |
| **Features on Screen** | None — system screen |
| **Key Data Objects** | None |
| **AI Interactions** | None |
| **Entry Points** | → from SCR-002 if both ReccoBeats and librosa fail; → from SCR-003 if all API paths fail; → from SCR-006 if optimization fails completely |
| **Exit Points** | → SCR-001 (Home) via "Start over"; → SCR-002 or SCR-003 via "Try again" |
| **State Variations** | Upload failure variant; ID lookup failure variant; optimization failure variant (distinct error messages per context) |
| **Estimated Complexity** | Low |

---

### SCR-010 — Empty State (No Features Yet)

| Field | Content |
|---|---|
| **Screen ID** | SCR-010 |
| **Screen Name** | Empty State — Optimizer No Data |
| **Navigation Path** | NAV-03 > NAV-03.1 (triggered when accessed without prior analysis) |
| **Screen Type** | Empty State |
| **Primary Archetype** | UA-01, UA-03 |
| **Features on Screen** | None — guidance screen |
| **Key Data Objects** | None |
| **AI Interactions** | None |
| **Entry Points** | → from SCR-001 via "Optimize Parameters" when no prior analysis session exists |
| **Exit Points** | → SCR-002 (Upload Audio) via "Analyze a song first"; → SCR-003 (Spotify ID Search) via "Search a song first" |
| **State Variations** | Default only — message: "To optimize parameters, start by analyzing a song first." |
| **Estimated Complexity** | Low |

---

### SCR-011 — GenAI Audio Preview

| Field | Content |
|---|---|
| **Screen ID** | SCR-011 |
| **Screen Name** | GenAI Audio Preview |
| **Navigation Path** | NAV-03 > NAV-03.3 |
| **Screen Type** | Detail |
| **Primary Archetype** | UA-01 (primary), UA-03 (secondary), UA-02 (secondary) |
| **Features on Screen** | FT-006 (GenAI Audio Preview from Optimized Parameters) |
| **Key Data Objects** | OBJ-007 (Optimization Result), OBJ-008 (Generated Audio Preview) |
| **AI Interactions** | AI-Powered — a generative audio model transforms the optimized parameter set into a short synthetic preview. Fallback state: "Preview generation could not complete. Please try again." |
| **Entry Points** | → from SCR-006 via "Generate AI Preview" CTA |
| **Exit Points** | → SCR-006 via "Back to Optimization Results"; → SCR-001 via "Start a new analysis" |
| **State Variations** | Default (parameters preloaded); loading (preview being generated); success (audio player visible with replay/regenerate actions); error (generation failed, retry available) |
| **Estimated Complexity** | High |

## IA-4 — Data Object Map

### OBJ-001 — Audio File

| Field | Content |
|---|---|
| **Object ID** | OBJ-001 |
| **Object Name** | Audio File |
| **Description** | The .mp3 file uploaded by the user for audio feature extraction via ReccoBeats or librosa |
| **Key Attributes** | file_name, file_size, file_format (.mp3), upload_timestamp, genre (optional, user-provided) |
| **Source** | User input (file upload via SCR-002) |
| **Owner Archetype** | UA-01, UA-02 |
| **Lifecycle** | Uploaded → Processed (features extracted) → Deleted (temporary file removed from disk after processing — see PI-06 HC-04 and PDF Section 4.1) |
| **Screens Where Visible** | SCR-002 (upload form) |
| **Relationships** | OBJ-001 produces OBJ-002 (Feature Set) after processing |
| **AI Involvement** | AI-Assisted — sent to ReccoBeats API for feature extraction; local librosa fallback if API unavailable |
| **Privacy / Sensitivity** | None — audio file is processed transiently and deleted; not stored persistently |

---

### OBJ-002 — Feature Set

| Field | Content |
|---|---|
| **Object ID** | OBJ-002 |
| **Object Name** | Feature Set |
| **Description** | The complete set of 13 audio features (plus genre) representing a track's musical characteristics, used as input to the prediction model |
| **Key Attributes** | danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, duration_ms, genre |
| **Source** | ReccoBeats API (from OBJ-001 or OBJ-003) / librosa local fallback / manual entry (optimizer) — see FL-01 in SM-AS |
| **Owner Archetype** | UA-01, UA-02, UA-03 |
| **Lifecycle** | Extracted → Displayed (SCR-004) → Passed to model (OBJ-004) → Passed to optimizer (OBJ-006) → Session-scoped (not persisted across sessions — see PI-05 out-of-scope) |
| **Screens Where Visible** | SCR-002 (implicit), SCR-003 (implicit), SCR-004 (displayed as context), SCR-005 (editable in wizard), SCR-006 (original values shown in comparison) |
| **Relationships** | OBJ-002 is produced-by OBJ-001 (audio upload) or OBJ-003 (Spotify ID lookup); OBJ-002 feeds OBJ-004 (Prediction Result); OBJ-002 feeds OBJ-006 (Optimization Request) |
| **AI Involvement** | AI-Enriched — logarithmic transformations applied (instrumentalness, liveness, speechiness, duration_ms) and StandardScaler normalization applied before model inference (PDF Section 3.3.2) |
| **Privacy / Sensitivity** | None |

---

### OBJ-003 — Spotify Track Reference

| Field | Content |
|---|---|
| **Object ID** | OBJ-003 |
| **Object Name** | Spotify Track Reference |
| **Description** | The Spotify track identifier entered by the user, resolved to a ReccoBeats internal ID for feature retrieval |
| **Key Attributes** | spotify_track_id (user-provided), reccobeats_internal_id (resolved), resolution_status (success / not_found) |
| **Source** | User input (SCR-003); resolved via ReccoBeats API (GET /search-track) |
| **Owner Archetype** | UA-02, UA-03 |
| **Lifecycle** | Entered → Resolved (to ReccoBeats ID) → Features fetched (OBJ-002 created) → Session-scoped |
| **Screens Where Visible** | SCR-003 (ID entry) |
| **Relationships** | OBJ-003 resolves-to ReccoBeats internal ID; OBJ-003 triggers production of OBJ-002 via track-features call |
| **AI Involvement** | None — purely a lookup / resolution operation |
| **Privacy / Sensitivity** | None |

---

### OBJ-004 — Prediction Result

| Field | Content |
|---|---|
| **Object ID** | OBJ-004 |
| **Object Name** | Prediction Result |
| **Description** | The predicted popularity score (0–100) returned by the LightGBM model for a given feature set |
| **Key Attributes** | predicted_popularity (float, 0–100), model_version (LightGBM), input_feature_set_ref (OBJ-002), prediction_timestamp |
| **Source** | AI-generated — FastAPI backend, LightGBM model (MSE = 152.29, R² = 0.389) |
| **Owner Archetype** | UA-01, UA-02, UA-03 |
| **Lifecycle** | Generated → Displayed (SCR-004) → Session-scoped (not persisted — PI-05) |
| **Screens Where Visible** | SCR-004 (primary display), SCR-006 (shown as "original score" for comparison) |
| **Relationships** | OBJ-004 is produced-by OBJ-002 (Feature Set); OBJ-004 references OBJ-005 (Feature Importance); OBJ-004 is-compared-with OBJ-007 (Optimization Result) in SCR-006 |
| **AI Involvement** | AI-Generated — LightGBM model; deterministic (same inputs → same output); model R² = 0.389, RMSE = 12.34 |
| **Privacy / Sensitivity** | None |

---

### OBJ-005 — Feature Importance

| Field | Content |
|---|---|
| **Object ID** | OBJ-005 |
| **Object Name** | Feature Importance |
| **Description** | The top contributing audio features to a prediction score, derived from SHAP value analysis of the LightGBM model, rendered as plain-language explanations |
| **Key Attributes** | top_features (list of up to 3), shap_direction (positive / negative per feature), plain_language_label (e.g., "High danceability is increasing your predicted popularity"), feature_id_ref |
| **Source** | AI-generated — SHAP library applied to LightGBM model output (PDF Section 3.3.5 — top predictors: duration_ms, danceability, valence, instrumentalness, loudness) |
| **Owner Archetype** | UA-01 (primary), UA-02 (primary), UA-03 (secondary) |
| **Lifecycle** | Generated → Displayed (SCR-004) → Session-scoped |
| **Screens Where Visible** | SCR-004 |
| **Relationships** | OBJ-005 belongs-to OBJ-004 (Prediction Result); OBJ-005 references OBJ-002 (Feature Set) fields |
| **AI Involvement** | AI-Generated — SHAP values computed at prediction time from LightGBM model; quality expectation: top-3 contributors must be directionally correct relative to model behavior documented in PDF Figure 3.8 |
| **Privacy / Sensitivity** | None |

---

### OBJ-006 — Optimization Request

| Field | Content |
|---|---|
| **Object ID** | OBJ-006 |
| **Object Name** | Optimization Request |
| **Description** | The configuration submitted by the user to the optimization wizard, specifying a base feature set and which features are locked vs. free for optimization |
| **Key Attributes** | base_features (OBJ-002 ref), locked_features (list of feature names), free_features (list of feature names), submitted_at |
| **Source** | User input (SCR-005) — pre-populated from OBJ-002 or entered manually |
| **Owner Archetype** | UA-01, UA-03 |
| **Lifecycle** | Configured → Submitted → Processed (grid evaluation) → Produces OBJ-007 → Session-scoped |
| **Screens Where Visible** | SCR-005 (configuration), SCR-006 (original values shown in comparison output) |
| **Relationships** | OBJ-006 references OBJ-002 (base Feature Set); OBJ-006 produces OBJ-007 (Optimization Result) |
| **AI Involvement** | None at input stage — AI involvement begins when the request is submitted and the model evaluates candidates |
| **Privacy / Sensitivity** | None |

---

### OBJ-007 — Optimization Result

| Field | Content |
|---|---|
| **Object ID** | OBJ-007 |
| **Object Name** | Optimization Result |
| **Description** | The parameter configuration returned by the optimization wizard that maximizes the predicted popularity score, including the optimized score and the delta from the original |
| **Key Attributes** | optimized_features (full feature set), optimized_score (float, 0–100), original_score (ref OBJ-004), score_delta, changed_parameters (list of features with original vs. optimized values) |
| **Source** | AI-Generated — LightGBM model evaluated iteratively over candidate grid (POST /wizard-optimize) |
| **Owner Archetype** | UA-01, UA-03 |
| **Lifecycle** | Generated → Displayed (SCR-006) → Session-scoped |
| **Screens Where Visible** | SCR-006 |
| **Relationships** | OBJ-007 is produced-by OBJ-006 (Optimization Request); OBJ-007 is-compared-with OBJ-004 (original Prediction Result) |
| **AI Involvement** | AI-Generated — iterative grid search using LightGBM; quality expectation: optimized score must be ≥ original score |
| **Privacy / Sensitivity** | None |

---

### OBJ-008 — Generated Audio Preview

| Field | Content |
|---|---|
| **Object ID** | OBJ-008 |
| **Object Name** | Generated Audio Preview |
| **Description** | A short synthetic audio sample generated by GenAI from the optimized feature configuration, intended to provide an audible approximation of the recommended musical profile |
| **Key Attributes** | audio_url or audio_blob, generation_timestamp, source_optimization_ref, preview_duration, generation_status |
| **Source** | AI-generated — generative audio model or synthesis endpoint |
| **Owner Archetype** | UA-01, UA-03, UA-02 |
| **Lifecycle** | Requested → Generated → Played → Regenerated or discarded → Session-scoped |
| **Screens Where Visible** | SCR-011 |
| **Relationships** | OBJ-008 is produced-by OBJ-007 (Optimization Result) |
| **AI Involvement** | AI-Generated — generative model creates a synthetic preview from optimized audio-feature targets |
| **Privacy / Sensitivity** | None |

## IA-5 — Interaction Flow Diagrams

### Flow 1: JN-TB-01 — Producer Evaluating Track via Audio Upload

**Archetype:** UA-01 — The Data-Driven Producer

#### Flow Path (text notation)

```
SCR-001 --[clicks "Upload .mp3"]--> SCR-002 --[uploads file + submits]--> SCR-004
SCR-004 --[clicks "Optimize this song"]--> SCR-005 --[submits optimization]--> SCR-006
SCR-006 --[clicks "Analyze another song"]--> SCR-001
SCR-006 --[clicks "Generate AI Preview"]--> SCR-011
SCR-011 --[clicks "Back to Optimization Results"]--> SCR-006

[Failure path A - ReccoBeats unavailable]:
SCR-002 --[librosa fallback activated]--> SCR-004  (transparent to user)

[Failure path B - both APIs fail]:
SCR-002 --[processing fails]--> SCR-009 --[retry / start over]--> SCR-002 or SCR-001
```

#### Flow Table

| From Screen | User Action | To Screen | Data Passed | Condition | AI Step? |
|---|---|---|---|---|---|
| SCR-001 | Clicks "Upload .mp3" card | SCR-002 | None | Always | No |
| SCR-002 | Uploads .mp3, selects genre (optional), clicks "Analyze" | SCR-004 | OBJ-001 → OBJ-002 extracted | Success: features extracted | Yes — FT-001 |
| SCR-002 | Processing fails (both ReccoBeats + librosa) | SCR-009 | Error context | Both extraction paths failed | No |
| SCR-009 | Clicks "Try again" | SCR-002 | None | Always | No |
| SCR-009 | Clicks "Start over" | SCR-001 | None | Always | No |
| SCR-004 | Page loads after extraction | SCR-004 | OBJ-002 → OBJ-004 + OBJ-005 | Prediction + SHAP computed | Yes — FT-003, FT-004 |
| SCR-004 | Clicks "Optimize this song" | SCR-005 | OBJ-002 pre-populated | Always (from results) | No |
| SCR-005 | Sets locks/free params, clicks "Find best configuration" | SCR-006 | OBJ-006 submitted | Always | Yes — FT-005 |
| SCR-006 | Clicks "Adjust parameters" | SCR-005 | OBJ-002 reloaded | Always | No |
| SCR-006 | Clicks "Analyze another song" | SCR-001 | None | Always | No |

#### Journey Coverage Validation — JN-TB-01

| Step # | Step Description | Screen | Covered? |
|---|---|---|---|
| 1 | Producer opens app and selects "Process Audio" | SCR-001 → SCR-002 | ✅ |
| 2 | Uploads .mp3 and selects optional genre | SCR-002 | ✅ |
| 3 | Backend extracts features (ReccoBeats + librosa fallback) | SCR-002 (processing) | ✅ |
| 4 | LightGBM returns predicted popularity score | SCR-004 | ✅ |
| 5 | SHAP-informed top 3 features displayed | SCR-004 | ✅ |
| 6 | Producer navigates to Optimization Wizard | SCR-004 → SCR-005 | ✅ |
| 7 | Reviews optimized suggestions and makes decision | SCR-006 | ✅ |

---

### Flow 2: JN-TB-02 — Independent Artist Searching by Spotify ID

**Archetype:** UA-02 — The Independent Artist Exploring Streaming

#### Flow Path (text notation)

```
SCR-001 --[clicks "Search by Spotify ID"]--> SCR-003 --[enters ID + submits]--> SCR-004
SCR-004 --[clicks "Analyze another song"]--> SCR-001
SCR-004 --[clicks "What does this score mean?"]--> SCR-008

[Failure path A - invalid Spotify ID]:
SCR-003 --[ID not found]--> SCR-003 (inline error) --[clicks "Try uploading instead"]--> SCR-002

[Failure path B - API failure]:
SCR-003 --[API fails]--> SCR-009 --[start over]--> SCR-001
```

#### Flow Table

| From Screen | User Action | To Screen | Data Passed | Condition | AI Step? |
|---|---|---|---|---|---|
| SCR-001 | Clicks "Search by Spotify ID" card | SCR-003 | None | Always | No |
| SCR-003 | Enters Spotify ID, clicks "Analyze" | SCR-004 | OBJ-003 → OBJ-002 retrieved | Success: ID resolved + features fetched | Yes — FT-002 |
| SCR-003 | ID not found in ReccoBeats | SCR-003 | Inline error shown | ID invalid or not indexed | No |
| SCR-003 | Clicks "Try uploading instead" (from error state) | SCR-002 | None | After ID lookup failure | No |
| SCR-003 | API fully fails | SCR-009 | Error context | Always on total failure | No |
| SCR-004 | Page loads after feature retrieval | SCR-004 | OBJ-002 → OBJ-004 + OBJ-005 | Prediction + SHAP computed | Yes — FT-003, FT-004 |
| SCR-004 | Clicks "What does this score mean?" | SCR-008 | None | Always | No |
| SCR-004 | Clicks "Analyze another song" | SCR-001 | None | Always | No |
| SCR-008 | Clicks "Back" or "Home" | SCR-001 | None | Always | No |

#### Journey Coverage Validation — JN-TB-02

| Step # | Step Description | Screen | Covered? |
|---|---|---|---|
| 1 | Artist opens app and selects "Search by Spotify ID" | SCR-001 → SCR-003 | ✅ |
| 2 | Enters Spotify track ID | SCR-003 | ✅ |
| 3 | Backend resolves ID and retrieves audio features | SCR-003 (processing) | ✅ |
| 4 | LightGBM predicts popularity score | SCR-004 | ✅ |
| 5 | Plain-language feature importance displayed | SCR-004 | ✅ |

---

### Flow 3: JN-TB-03 — A&R Analyst Using the Optimization Wizard

**Archetype:** UA-03 — The A&R Catalog Analyst

#### Flow Path (text notation)

```
SCR-001 --[clicks "Search by Spotify ID"]--> SCR-003 --[submits ID]--> SCR-004
SCR-004 --[clicks "Optimize this song"]--> SCR-005 --[configures locks, submits]--> SCR-006
SCR-006 --[clicks "Adjust parameters"]--> SCR-005  (iterative loop)
SCR-006 --[copies/notes results for memo]--> SCR-001
SCR-006 --[clicks "Generate AI Preview"]--> SCR-011
SCR-011 --[clicks "Back to Optimization Results"]--> SCR-006

[Empty state - no prior analysis]:
SCR-001 --[clicks "Optimize Parameters" without prior analysis]--> SCR-010
SCR-010 --[clicks "Search a song first"]--> SCR-003
```

#### Flow Table

| From Screen | User Action | To Screen | Data Passed | Condition | AI Step? |
|---|---|---|---|---|---|
| SCR-001 | Clicks "Search by Spotify ID" | SCR-003 | None | Always | No |
| SCR-003 | Enters Spotify ID, submits | SCR-004 | OBJ-003 → OBJ-002 | Success | Yes — FT-002 |
| SCR-004 | Page loads | SCR-004 | OBJ-002 → OBJ-004 + OBJ-005 | Prediction + SHAP computed | Yes — FT-003, FT-004 |
| SCR-004 | Clicks "Optimize this song" | SCR-005 | OBJ-002 pre-populated | Always | No |
| SCR-005 | Locks selected features, sets free params, submits | SCR-006 | OBJ-006 submitted | Always | Yes — FT-005 |
| SCR-006 | Page loads with optimizer output | SCR-006 | OBJ-007 displayed vs. OBJ-004 | Success | Yes — FT-005 |
| SCR-006 | Clicks "Adjust parameters" | SCR-005 | OBJ-002 reloaded | Iterative refinement | No |
| SCR-006 | Done — exits to Home | SCR-001 | None | Always | No |
| SCR-001 | Clicks "Optimize Parameters" (no prior analysis) | SCR-010 | None | No session feature set | No |
| SCR-010 | Clicks "Search a song first" | SCR-003 | None | Always | No |

#### Journey Coverage Validation — JN-TB-03

| Step # | Step Description | Screen | Covered? |
|---|---|---|---|
| 1 | Analyst arrives at wizard from Spotify ID lookup | SCR-003 → SCR-004 → SCR-005 | ✅ |
| 2 | Analyst locks non-negotiable features | SCR-005 | ✅ |
| 3 | Analyst submits optimization request | SCR-005 → SCR-006 | ✅ |
| 4 | Results display optimized config vs. original | SCR-006 | ✅ |
| 5 | Analyst uses output for recommendation memo | SCR-006 (manual copy/export) | ✅ |

---

## IA-6 — Validation Report

### Check 1 — Feature Coverage

| Feature ID | Feature Name | Screen ID | Status |
|---|---|---|---|
| FT-001 | Audio File Upload and Feature Extraction | SCR-002 | ✅ Covered |
| FT-002 | Spotify Track ID Lookup and Feature Retrieval | SCR-003 | ✅ Covered |
| FT-003 | Popularity Score Prediction | SCR-004 | ✅ Covered |
| FT-004 | SHAP-Informed Feature Importance Display | SCR-004 | ✅ Covered |
| FT-005 | Parameter Optimization Wizard | SCR-005, SCR-006 | ✅ Covered (split across 2 wizard steps — justified: wizard input and output are distinct views, neither exceeds 5 features) |

**Summary: 5 features covered, 0 orphaned, 0 duplicates.**

---

### Check 2 — Journey Walkability

| Journey ID | Step # | Step Description | Screen | Status |
|---|---|---|---|---|
| JN-TB-01 | 1 | Select "Process Audio" from home | SCR-001 → SCR-002 | ✅ |
| JN-TB-01 | 2 | Upload .mp3 + optional genre | SCR-002 | ✅ |
| JN-TB-01 | 3 | Feature extraction (ReccoBeats / librosa) | SCR-002 (processing) | ✅ |
| JN-TB-01 | 4 | Popularity score returned | SCR-004 | ✅ |
| JN-TB-01 | 5 | SHAP-informed top 3 features displayed | SCR-004 | ✅ |
| JN-TB-01 | 6 | Navigate to Optimization Wizard | SCR-004 → SCR-005 | ✅ |
| JN-TB-01 | 7 | Review optimized suggestions and decide | SCR-006 | ✅ |
| JN-TB-02 | 1 | Select "Search by Spotify ID" from home | SCR-001 → SCR-003 | ✅ |
| JN-TB-02 | 2 | Enter Spotify track ID | SCR-003 | ✅ |
| JN-TB-02 | 3 | Feature retrieval via ReccoBeats | SCR-003 (processing) | ✅ |
| JN-TB-02 | 4 | Popularity score returned | SCR-004 | ✅ |
| JN-TB-02 | 5 | Plain-language feature importance displayed | SCR-004 | ✅ |
| JN-TB-03 | 1 | Arrive at wizard from Spotify ID lookup | SCR-003 → SCR-004 → SCR-005 | ✅ |
| JN-TB-03 | 2 | Lock non-negotiable features | SCR-005 | ✅ |
| JN-TB-03 | 3 | Submit optimization request | SCR-005 → SCR-006 | ✅ |
| JN-TB-03 | 4 | Results display optimized vs. original | SCR-006 | ✅ |
| JN-TB-03 | 5 | Analyst uses output for recommendation | SCR-006 | ✅ |

**Summary: 17/17 journey steps covered, 0 dead ends, 0 missing screens.**

---

### Check 3 — Navigation Depth

| Navigation Path | Depth | Status |
|---|---|---|
| NAV-01 (Home) | Level 1 | ✅ |
| NAV-02 > NAV-02.1 (Upload Audio) | Level 2 | ✅ |
| NAV-02 > NAV-02.2 (Spotify ID Search) | Level 2 | ✅ |
| NAV-02 > NAV-02.3 (Results) | Level 2 | ✅ |
| NAV-03 > NAV-03.1 (Optimizer Setup) | Level 2 | ✅ |
| NAV-03 > NAV-03.2 (Optimizer Results) | Level 2 | ✅ |
| NAV-04 (How It Works) | Level 1 | ✅ |
| NAV-05 (Help & Limitations) | Level 1 | ✅ |

Maximum depth: 2 levels. Rule limit: 3 levels. **✅ Compliant.**

Level 1 sections with 0 features assigned: NAV-01 (routing hub — justified), NAV-04, NAV-05 (content screens — justified). All feature-bearing sections confirmed.

---

### Check 4 — Screen Density

| Screen ID | Screen Name | Feature Count | Status |
|---|---|---|---|
| SCR-001 | Home | 0 (routing hub) | ✅ Justified |
| SCR-002 | Upload Audio | 1 (FT-001) | ✅ |
| SCR-003 | Search by Spotify ID | 1 (FT-002) | ✅ |
| SCR-004 | Analysis Results | 2 (FT-003, FT-004) | ✅ |
| SCR-005 | Optimizer Setup | 1 (FT-005 input) | ✅ |
| SCR-006 | Optimizer Results | 1 (FT-005 output) | ✅ |
| SCR-007 | How It Works | 0 (content) | ✅ Justified |
| SCR-008 | Help & Limitations | 0 (content) | ✅ Justified |
| SCR-009 | System Error | 0 (system screen) | ✅ Justified |
| SCR-010 | Empty State | 0 (system screen) | ✅ Justified |

**No screen exceeds 5 features. ✅ Compliant.**

---

### Check 5 — Data Object Traceability

| Feature ID | Data Dependency (from FD) | Object ID | Status |
|---|---|---|---|
| FT-001 | .mp3 audio file (user-provided) | OBJ-001 | ✅ |
| FT-001 | ReccoBeats API audio-features endpoint | OBJ-002 (produced) | ✅ |
| FT-001 | librosa library (fallback) | OBJ-002 (produced) | ✅ |
| FT-002 | Spotify track ID (user-provided) | OBJ-003 | ✅ |
| FT-002 | ReccoBeats API (search + track-features) | OBJ-002 (produced) | ✅ |
| FT-003 | 13 audio features + genre label | OBJ-002 | ✅ |
| FT-003 | Pre-trained LightGBM model artifact | OBJ-004 (produced) | ✅ |
| FT-003 | StandardScaler + OneHotEncoder artifacts | OBJ-002 (transformed input) | ✅ |
| FT-004 | SHAP value distributions (model analysis) | OBJ-005 | ✅ |
| FT-005 | Feature set from FT-001 or FT-002 | OBJ-002 | ✅ |
| FT-005 | Locked/free feature selection (user-provided) | OBJ-006 | ✅ |
| FT-005 | Pre-trained LightGBM model (grid evaluation) | OBJ-007 (produced) | ✅ |

**Summary: 12/12 data dependencies mapped to objects, 0 missing. ✅ Compliant.**

---

### Check 6 — AI Interaction Consistency

| Feature ID | AI Level (FD) | Screen ID | AI Documented on Screen? | Fallback Documented? |
|---|---|---|---|---|
| FT-001 | AI-Assisted | SCR-002 | ✅ | ✅ — librosa fallback transparent to user; total failure routes to SCR-009 |
| FT-002 | AI-Assisted | SCR-003 | ✅ | ✅ — invalid ID shows inline error; API failure routes to SCR-009 |
| FT-003 | AI-Powered | SCR-004 | ✅ | ✅ — model failure shows retry prompt; limitation notice always visible |
| FT-004 | AI-Powered | SCR-004 | ✅ | ✅ — shares fallback with FT-003 on SCR-004 |
| FT-005 | AI-Powered | SCR-005, SCR-006 | ✅ | ✅ — optimization failure shows message with retry option on SCR-006 |

**Summary: 5/5 AI features documented with fallback states. ✅ Compliant.**

---

## Quality Checklist

| # | Check | Status |
|---|---|---|
| 1 | Navigation model has ≤ 7 Level 1 sections (web) | [x] — 5 sections (NAV-01 through NAV-05) |
| 2 | Navigation depth ≤ 3 levels everywhere | [x] — maximum depth is 2 levels |
| 3 | 100% of features from FD mapped to exactly 1 screen (no orphans) | [x] — 5/5 features covered |
| 4 | No screen hosts more than 5 features | [x] — max is 2 features on SCR-004 |
| 5 | Every screen has ≥ 1 entry point and ≥ 1 exit point | [x] — validated in screen inventory |
| 6 | Every JN-TB journey is walkable end-to-end through interaction flows | [x] — 17/17 steps covered across 3 journeys |
| 7 | Data object map covers all FD data dependencies | [x] — 12/12 dependencies mapped |
| 8 | All AI-powered screens have documented fallback states | [x] — 5/5 AI features with fallback |
| 9 | Navigation labels use archetype language, not developer jargon | [x] — "Analyze a Song", "Optimize Parameters", "How It Works" |
| 10 | Non-feature screens included: Onboarding, Error, Empty State, Help | [x] — SCR-007, SCR-008, SCR-009, SCR-010 |
| 11 | IA-VALIDATE run with 0 critical issues remaining | [x] — 0 ❌ issues; all checks passed |
| 12 | Navigation model validated against archetype mental models | [x] — three routes map directly to the three user flows from PI-02 and JN-TB |

---

## Traceability Summary

| Upstream Artifact | How It Feeds This IA |
|---|---|
| UA-01, UA-02, UA-03 (Archetypes) | Navigation organized by archetype workflow: upload (UA-01/UA-02), ID search (UA-02/UA-03), optimizer (UA-01/UA-03) |
| FT-001 through FT-005 (Feature Definitions) | Each feature assigned to exactly one screen; coverage matrix confirms no orphans |
| JN-TB-01, JN-TB-02, JN-TB-03 (Journeys To-Be) | All 17 journey steps walkable end-to-end through interaction flows; 3 complete flow diagrams produced |
| SM-AS L3 Flows (FL-01 through FL-04) | FL-01 (manual track evaluation) replaced by SCR-002 + SCR-003; FL-03 (post-release data) contextualized in SCR-008 (limitations) |
| SM-AS L4 Resources (RES-01 through RES-06) | ReccoBeats (OBJ-001 → OBJ-002), librosa fallback, LightGBM model all represented in data object map |
| PC (Product Concept) — Scope Boundaries | Batch analysis: no batch screen; social media signals: not present in any data object; user history: lifecycle of all objects is session-scoped |
| PI-02 (Three interaction routes) | Three entry flows from SCR-001 map directly to Upload, ID Search, and Optimize routes |
| PI-06 (HC-04: file deletion) | OBJ-001 lifecycle: Deleted after processing — documented in data object and SCR-002 behavior |

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
| 1.0 | 14/03/2026 | HPWSB | Information Architecture produced from MPP-PH0-INTAKE-v1.0, MPP-PH1-SD-v1.0, and ProyectoIntegrador2 PDF report |

---

_MPP-PH1-IA-v1.0 | Music Popularity Predictor Project | Universidad EAFIT_
