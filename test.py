# PCマイクでのリアルタイムテスト

import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier

from utils import extract_feature

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    #「サイレント」閾値以下なら「真」を返す
    return max(snd_data) < THRESHOLD
    # 音量の平均化
def normalize(snd_data):

    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    # 最初と最後の空白部分をトリミング
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # 左側にトリム
    snd_data = _trim(snd_data)

    # 右側にトリム
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    # 長さ「秒」の「snd_data」の開始と終了に無音を追加（float）
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    マイクから単語を録音し、データを符号付き短文の配列として返す
    音声を正規化し，無音部分をトリミングする(開始と終了に0.5秒のパッドが付く)
    VLCなどが再生できるように、空白の音を切り落とさないようにする
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # リトルエンディアン式符号付きショート
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    # マイクから録音し、得られたデータを「パス」に出力する
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



if __name__ == "__main__":
    # 保存された学習済みモデルのロード
    model = pickle.load(open("result/mlp_classifier.model", "rb"))
    print("音声を流してください")
    filename = "test.wav"
    # 録音する（話し始める）
    record_to_file(filename)
    # 特徴を抽出し、形を変える
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)

    result = model.predict(features)[0]
    print("判定結果:", result)
