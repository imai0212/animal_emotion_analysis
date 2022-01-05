# データの前処理

import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# データセット内の感情
str2emotion = {
    "B": "ポジティブ",
    "I": "ネガティブ",
    "F": "食べ物を欲している"
}

# 分類対象とする感情
AVAILABLE_EMOTIONS = {
    "ポジティブ",
    "ネガティブ",
    "食べ物を欲している"
}

# オーディオファイル `file_name` から特徴量を抽出
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


def load_data(test_size=0.2):
    X, y = [], []
    for audio in glob.glob("data/*.wav"):
        # オーディオファイルのベース名取得
        basename = os.path.basename(audio)
        # 感情の読み取り
        emotion = str2emotion[basename.split("_")[0]]
        # AVAILABLE_EMOTIONSのみ許可
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # 音声特徴の抽出
        features = extract_feature(audio, mfcc=True, chroma=True, mel=True)
        # データの追加
        X.append(features)
        y.append(emotion)
    # データを訓練用とテスト用に分割して返す
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
