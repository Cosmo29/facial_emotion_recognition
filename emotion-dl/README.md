# Facial Emotion Recognition - Deep Learning (PyTorch)

A CNN trained **from scratch** on FER-2013 to classify faces into 7 emotions
(angry, disgust, fear, happy, neutral, sad, surprise), with evaluation and two ways to run it:
a live webcam window and a browser app.

## Pipeline
1. **Face detection** - OpenCV Haar cascade finds faces.
2. **Preprocess** - crop, grayscale, resize to 48x48, normalise to [-1, 1].
3. **Classify** - 8-conv-layer VGG-style CNN (BatchNorm, Dropout, global avg pool) -> 7 classes.
4. **Smooth** - webcam predictions are averaged over the last few frames to stop flicker.

Training uses data augmentation (flip, rotation, shift, zoom, random erasing), class-weighted loss
(FER-2013 has very few "disgust" images), label smoothing, AdamW and a OneCycle LR schedule.

## Setup (Windows, Python 3.10-3.12 recommended)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Get the dataset
Download FER-2013 from Kaggle (search "fer2013 msambare") and extract so you have:
```
data/
  train/angry, disgust, fear, happy, neutral, sad, surprise  (images inside)
  test/ ... same 7 folders
```

## Run
```
python train.py              # trains, saves checkpoints/best.pt + training_curves.png
python evaluate.py           # test accuracy, per-class report, confusion_matrix.png
python realtime.py           # live webcam, press q to quit
streamlit run app.py         # browser app at http://localhost:8501
```
Useful options: `python train.py --epochs 25 --batch-size 64`.
Training on CPU is slow (roughly tens of minutes per epoch on a laptop); a GPU or Google Colab is much faster.

## Expected results
A model like this typically reaches roughly 60-68% test accuracy on FER-2013 (human agreement on this
dataset is only about 65%). "Happy" and "surprise" score highest; "fear", "sad" and "disgust" are hardest.
Your exact numbers will vary - report what `evaluate.py` prints.

## Project layout
```
model.py      CNN + shared helpers          train.py      training loop
evaluate.py   metrics + confusion matrix    realtime.py   webcam demo
app.py        Streamlit app                 requirements.txt
```
