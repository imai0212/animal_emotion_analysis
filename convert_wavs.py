# 音声データの形式変換（16000Hz, モノラルチャンネル）
import os

# audio_pathの設定
def convert_audio(audio_path, target_path, remove=False):
    # audio_path : 変換前のオーディオwavファイルのパス
    # target_path : 変換後のwavファイルを保存するパス
    os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")

    if remove:
        # 古いファイルを削除
        os.remove(audio_path)

# wavファイルのパス変換
def convert_audios(path, target_path, remove=False):

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # wavファイル
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    # 無効な指定があった場合にヘルプと使用方法メッセージを生成
    import argparse
    parser = argparse.ArgumentParser(description="""wavファイルを16MHz、モノラル音声チャンネル（1チャンネル）に変換（圧縮）します。
                                                    wavファイルを訓練用やテスト用に圧縮するために活用して下さい。""")
    parser.add_argument("audio_path", help="変換前のwavファイルがあるフォルダ")
    parser.add_argument("target_path", help="返還後のwavファイルを保存するフォルダ")
    parser.add_argument("-r", "--remove", type=bool, help="変換後に古いwavファイルを削除するかどうか", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("指定されたaudio_pathファイルはこの操作に適していません。")
