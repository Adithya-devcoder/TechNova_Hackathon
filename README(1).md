# SignBridge --- Continuous Sign Language Recognition

> **Hackathon Project \| TECHNOVA 2026 --- PS-41**\
> **Team size:** 4 members\
> **Status:** Prototype in development

SignBridge is a pose-based sign-language recognition prototype that aims
to convert a continuous sequence of signs into readable text. It uses
webcam input, hand/body landmark extraction, and a temporal recognition
model to interpret sign sequences.

The project is designed around **PS-41: Continuous Sign Language
Recognition at Sentence Level**. The problem statement calls for a
vocabulary of at least 50 signs, real-time inference on a laptop,
evaluation on signers held out from training, and a live demonstration
of at least 10 complete sentences.

> **Scope note:** Recognition quality depends on the chosen
> sign-language dataset, its labels, and the model trained. Do not
> describe the system as supporting every sign language or as
> production-ready unless testing demonstrates that claim.

## Contents

-   [Problem](#problem)
-   [Our Approach](#our-approach)
-   [Objectives](#objectives)
-   [Planned Features](#planned-features)
-   [Technology Stack](#technology-stack)
-   [System Workflow](#system-workflow)
-   [Team Roles](#team-roles)
-   [Suggested Repository Structure](#suggested-repository-structure)
-   [Getting Started](#getting-started)
-   [Dataset and Data Safety](#dataset-and-data-safety)
-   [Evaluation Plan](#evaluation-plan)
-   [24-Hour Hackathon Plan](#24-hour-hackathon-plan)
-   [Known Limitations](#known-limitations)
-   [Future Enhancements](#future-enhancements)
-   [Team](#team)
-   [Acknowledgements](#acknowledgements)

## Problem

Many sign-language interfaces demonstrate isolated gesture recognition,
but understanding a continuous sequence of signs is more difficult.
Signs can overlap in time, transitions affect their appearance, and
different signers may perform the same sign differently.

Our goal is to build a focused prototype that recognises a restricted
but meaningful vocabulary and produces text for complete signed
sequences.

## Our Approach

1.  Capture video frames from a laptop webcam or load a permitted sample
    video.
2.  Extract hand and, where useful, body/pose landmarks from each frame.
3.  Convert landmark sequences into model-ready features.
4.  Use a temporal model to predict a sequence of sign labels.
5.  Decode the predicted sequence into readable text.
6.  Display the recognised text and provide a way to retry or clear the
    result.
7.  Evaluate the system on held-out signers and document errors
    honestly.

The initial implementation should prioritise a reliable end-to-end
demonstration. More advanced temporal modelling can be added after the
data pipeline works.

## Objectives

-   Build a reproducible pose/landmark extraction pipeline.
-   Recognise a defined vocabulary of **at least 50 signs**, subject to
    dataset availability.
-   Support sequence-level recognition rather than only independent,
    isolated gestures.
-   Aim for real-time inference on a standard laptop.
-   Evaluate generalisation using signers excluded from training.
-   Report **Word Error Rate (WER)** and explain the evaluation split.
-   Demonstrate at least **10 complete sentences** in the live demo, as
    required by PS-41.

## Planned Features

-   **Webcam or sample-video input** for testing.
-   **Hand and pose landmark visualisation** to make the pipeline
    understandable.
-   **Continuous sequence prediction** using a temporal model.
-   **Text output** for the recognised sign sequence.
-   **Reset/retry controls** for repeated demonstrations.
-   **Evaluation summary** with WER and examples of successful and
    failed predictions.
-   **Offline demo fallback**, such as a permitted sample clip and
    cached output, in case the webcam or environment fails.

Features are planned until they are implemented and tested. Mark
completed features in the project board or README as work progresses.

## Technology Stack

  -----------------------------------------------------------------------
  Component               Proposed tool           Purpose
  ----------------------- ----------------------- -----------------------
  Programming language    Python                  Data processing and
                                                  model pipeline

  Video capture           OpenCV                  Webcam/video input and
                                                  frame handling

  Landmark extraction     MediaPipe or a          Extract spatial
                          comparable              features from frames
                          pose/hand-landmark tool 

  Model                   PyTorch                 Train or run the
  training/inference                              temporal sequence model

  User interface          Streamlit (fast         Webcam/video controls
                          prototype)              and text results

  Data processing         NumPy, pandas           Feature arrays, labels,
                                                  and evaluation records

  Version control         Git and GitHub          Collaboration and
                                                  reproducibility
  -----------------------------------------------------------------------

The final stack may change depending on dataset compatibility, hardware,
and the time available.

## System Workflow

``` text
Webcam / Sample Video
          |
          v
    Frame Extraction
          |
          v
 Hand + Pose Landmarks
          |
          v
 Feature Sequence / Normalisation
          |
          v
 Temporal Recognition Model
          |
          v
 Sequence Decoding / Text Cleanup
          |
          v
 Recognised Text + Evaluation
```

## Team Roles

Four members should own separate workstreams while agreeing on shared
data formats and integration points early.

  -----------------------------------------------------------------------
  Member                  Primary responsibility  Deliverables
  ----------------------- ----------------------- -----------------------
  **Member 1 --- Dataset  Find and inspect a      Dataset notes, label
  & Research**            suitable public         map, split files,
                          dataset; review its     sample-data checks
                          sign vocabulary,        
                          labels, licence, and    
                          signer IDs; prepare     
                          train/validation/test   
                          splits without signer   
                          leakage.                

  **Member 2 --- Landmark Implement video/frame   Reusable extraction
  Pipeline**              processing and          module, visualisation,
                          hand/body landmark      feature files
                          extraction; handle      
                          missing landmarks,      
                          normalisation, and      
                          feature saving.         

  **Member 3 ---          Build the baseline      Training/inference
  AI/Sequence Model**     recogniser and temporal code, model checkpoint
                          sequence model;         if feasible, metrics
                          train/validate it;      and error examples
                          implement sequence      
                          decoding and calculate  
                          WER.                    

  **Member 4 --- UI,      Build the user          Working interface,
  Integration & Demo**    interface, connect      integration checks,
                          input → landmarks →     demo script and slides
                          model → text, manage    
                          Git integration, and    
                          prepare the demo/pitch. 
  -----------------------------------------------------------------------

### Shared responsibilities

-   All members review the system workflow and agree on input/output
    formats.
-   Commit working code frequently and keep the main branch runnable.
-   Member 1 and Member 3 jointly verify that the test set contains
    unseen signers.
-   Member 4 coordinates integration, but every member should help debug
    blockers.
-   Everyone should be able to explain the architecture, dataset,
    evaluation, and limitations.

## Suggested Repository Structure

``` text
SignBridge/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── README.md
│   ├── raw/                 # Keep downloaded data out of Git
│   ├── processed/
│   └── splits/
├── src/
│   ├── video_input.py
│   ├── landmark_extractor.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── decoder.py
│   └── evaluation.py
├── models/                  # Usually keep large checkpoints out of Git
├── tests/
├── reports/
│   ├── evaluation.md
│   └── error_analysis.md
└── demo/
    └── demo_script.md
```

This is a suggested structure, not a claim that these files already
exist. Adjust it to match the actual implementation.

## Getting Started

### 1. Requirements

-   Python 3.10 or a version supported by the selected dependencies
-   Git
-   A laptop webcam for live input, or a permitted sample video
-   Enough disk space for the selected dataset and model files

### 2. Clone the repository

``` bash
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd SignBridge
```

Replace the placeholder with your team's public repository URL.

### 3. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

After the team finalises compatible package versions, list them in
`requirements.txt` and run:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not install arbitrary versions during final integration; use the
versions tested by the team.

### 5. Add the dataset

Follow the dataset's official instructions and licence. Keep raw dataset
files out of Git unless redistribution is explicitly permitted. Document
the dataset name, source URL, licence, selected vocabulary, and
preprocessing steps in `data/README.md`.

### 6. Run the application

Once `app.py` and the pipeline are implemented, document the exact
tested command here. For a Streamlit interface, the intended command may
be:

``` bash
streamlit run app.py
```

This command will work only after the Streamlit app and its dependencies
have been implemented.

## Dataset and Data Safety

-   Prefer a public dataset with clear usage terms and labels suitable
    for continuous signing.
-   Verify whether the dataset represents the specific sign language and
    vocabulary the team intends to demonstrate.
-   Record signer IDs and split data by signer where possible; do not
    randomly split near-duplicate clips from the same signer across
    training and test sets.
-   Do not upload private recordings, personal information, or
    restricted dataset files to the public repository.
-   Ask permission before recording anyone. Explain the purpose and
    obtain consent.
-   Include dataset attribution and comply with its licence.
-   Do not claim support for a language or vocabulary that has not been
    evaluated.

## Evaluation Plan

The primary metric required by PS-41 is **Word Error Rate (WER)** on
signers held out from training.

A common definition is:

\[ `\mathrm{WER}`{=tex} = `\frac{S + D + I}{N}`{=tex} \]

Where:

-   `S` = substitutions
-   `D` = deletions
-   `I` = insertions
-   `N` = number of words in the reference transcription

Report WER as a percentage when multiplying the ratio by 100. Lower WER
is better.

The evaluation report should include:

1.  Dataset name, source, licence, vocabulary, and number of
    signers/clips used.
2.  Train/validation/test split, explicitly identifying held-out
    signers.
3.  Baseline method and the proposed model.
4.  WER on the held-out test set.
5.  Inference speed or latency on the target laptop.
6.  Results for the 10 or more complete-sentence demo examples.
7.  Failure cases, such as similar signs, missing landmarks,
    transitions, or unseen signing styles.
8.  Any limitations caused by the dataset or restricted vocabulary.

Do not report invented metrics. If a result has not been measured, label
it **Not measured**.

### Results table template

  ------------------------------------------------------------------------
  Method       Test split            WER (%)        Inference Notes
                                                      latency 
  ------------ ------------ ---------------- ---------------- ------------
  Baseline     Held-out         Not measured     Not measured Fill after
               signers                                        testing

  Proposed     Held-out         Not measured     Not measured Fill after
  model        signers                                        testing
  ------------------------------------------------------------------------

## 24-Hour Hackathon Plan

  -----------------------------------------------------------------------
  Time window             Priority                Expected checkpoint
  ----------------------- ----------------------- -----------------------
  Hours 0--2              Confirm dataset,        Dataset and interfaces
                          vocabulary,             agreed
                          environment, and        
                          evaluation plan         

  Hours 2--5              Get a baseline running  First end-to-end
                          early                   prediction

  Hours 5--10             Landmark extraction and Reproducible feature
                          data preprocessing      pipeline

  Hours 10--15            Sequence model and UI   Working continuous-sign
                          integration             prototype

  Hours 15--18            Fix integration issues  Stable demo candidate
                          and run held-out tests  

  Hours 18--21            Measure WER, latency,   Results and evaluation
                          and failure cases       report

  Hours 21--24            Final testing, demo     Submission-ready
                          video, and six-slide    repository and
                          pitch                   presentation
  -----------------------------------------------------------------------

**Rule for the team:** get a small end-to-end baseline working early. Do
not spend the entire hackathon training a model before testing whether
the complete pipeline runs.

## Known Limitations

-   Recognition is limited to the vocabulary and sign language
    represented in the selected dataset.
-   Performance may vary with lighting, camera angle, occlusion, signing
    speed, and signer variation.
-   Landmark-based methods can lose information about fine hand shape or
    facial expression.
-   A sentence output may be incorrect; the interface should make it
    easy to retry rather than imply perfect translation.
-   A restricted-vocabulary prototype is not a replacement for a
    qualified human interpreter.
-   Real-time performance and held-out-signer generalisation must be
    measured on the target hardware.

## Future Enhancements

-   Expand the vocabulary and include more signers.
-   Compare temporal architectures such as a Transformer and a CTC-based
    model when data and time permit.
-   Improve handling of co-articulation and signer variation.
-   Add confidence indicators and an explicit "uncertain---please
    repeat" state.
-   Explore a text-to-sign avatar as a stretch goal, after the core
    recognition pipeline is reliable.

## Team

Add your actual names and links before submission.

  Member     Name       Role
  ---------- ---------- ------------------------
  Member 1   `<Name>`   Dataset & Research
  Member 2   `<Name>`   Landmark Pipeline
  Member 3   `<Name>`   AI/Sequence Model
  Member 4   `<Name>`   UI, Integration & Demo

## Acknowledgements

This project is developed for **TECHNOVA 2026, PS-41: Continuous Sign
Language Recognition at Sentence Level**. Credit all datasets, models,
libraries, and other resources used in the final implementation.

------------------------------------------------------------------------

**Submission checklist**

-   [ ] Public GitHub repository with setup and reproducibility
    instructions
-   [ ] Dataset sources, licences, and tools listed
-   [ ] Baseline and proposed-model evaluation reported
-   [ ] WER measured on held-out signers
-   [ ] At least 10 complete sentences prepared for the demo
-   [ ] Limitations and failure cases documented
-   [ ] Three-minute demo video or working live demo link
-   [ ] Six-slide pitch: Problem, Approach, Architecture, Results,
    Limitations, Next Steps
-   [ ] Team names and roles completed
