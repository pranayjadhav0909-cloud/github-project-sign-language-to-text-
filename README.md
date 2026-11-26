# Advanced Sign Language AI — Starter Project

This project scaffold supports a custom dataset for:
- Static or short-sequence alphabet classes (A-Z)
- Digit classes (0-9)
- Word-level gesture classes (e.g. HELLO, THANK_YOU, BYE)
- Sequence classification using an LSTM
- Word assembly, spelling correction, and basic next-word suggestions
- OpenCV webcam collection and prediction

## Important limitations

1. **No pretrained project model is included.** A meaningful model must be trained on your labeled recordings. Empty model files would be misleading.
2. **95% accuracy is a research target, not a guarantee.** Report accuracy only after evaluation on held-out signers and sessions. Randomly splitting neighboring samples can inflate results.
3. This is a hand-landmark prototype. Full sign-language translation needs facial expressions, body pose, two-hand coordination, timing, sign segmentation, and linguistic grammar. The included 126-feature representation contains two hands x 21 landmarks x 3 coordinates.
4. The vocabulary must use one clearly defined sign language (e.g. Indian Sign Language). ASL and ISL signs are not interchangeable. Have fluent signers validate labels and signing variants.
5. A sign alphabet is not necessarily a one-frame static pose: some letters and signs are dynamic. Record a full sequence for each class.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If MediaPipe installation fails, use a clean Python 3.10/3.11 environment and check the current MediaPipe compatibility for your operating system.

## Project workflow

1. Collect samples for every class:
   ```bash
   python collect_data.py --label A --samples 60
   python collect_data.py --label 0 --samples 60
   python collect_data.py --label HELLO --samples 60
   ```
2. Repeat for all labels. Use a minimum of 200–500 sequences per class for a serious prototype; more signers and recording sessions are preferable.
3. Train:
   ```bash
   python train_model.py
   ```
4. Run live recognition:
   ```bash
   python app.py
   ```

## Label naming

Use uppercase labels consistently:
- Letters: A, B, ..., Z
- Digits: 0, 1, ..., 9
- Words: HELLO, THANK_YOU, BYE, SORRY, PLEASE, YES, NO, HELP

The training script reads one folder per class under `dataset/`.

## Dataset collection protocol

- Capture multiple people, sessions, backgrounds, lighting conditions and camera distances.
- Include left-handed and right-handed signers where relevant.
- Record the correct language and label, signer ID, session ID and any variant.
- Do not put samples from the same signer/session in both training and test sets when claiming generalization.
- Include a REST/NEUTRAL class and a deliberate confirmation or segmentation method to avoid repeated words.
- Do not label arbitrary gestures as ISL/ASL without checking them with a fluent signer.

## Hunspell / CyHunspell

The language module tries CyHunspell first, then `pyspellchecker`. Hunspell corrects spelling; it does not infer sign-language grammar or reliably translate gestures into sentences. Next-word suggestions in this starter are a small optional n-gram baseline and should be trained on an appropriate, licensed text corpus.

## Model details

The supplied baseline is a sequence LSTM over 30 frames x 126 hand landmark features. It is a starting baseline, not a claim of state-of-the-art performance. The feature extractor sorts hands by wrist x-coordinate for consistency; this can still be ambiguous for left/right hand identity. For research-grade ISL, extend it with MediaPipe Holistic face and pose landmarks and explicit handedness.

## Evaluation

Use:
- Accuracy, macro precision, macro recall, macro F1
- Per-class precision/recall and confusion matrix
- Top-3 accuracy
- Inference latency and FPS
- Word Error Rate for continuous sentence transcription
- Signer-independent and session-independent evaluation

Do not tune the model on the final test set. Keep a separate validation set.
