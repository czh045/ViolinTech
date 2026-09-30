import os
import subprocess

BASE_DIR = r"Z:\ViolinTech\rouxian"

# 从01到29，跳过15和16
folders = [f"{i:02d}" for i in range(1, 30) if i not in [15, 16]]

def extract_audio():
    print("=" * 50)
    print("音频提取工具")
    print("=" * 50)
    print(f"处理文件夹: {folders}")
    print("=" * 50)

    for folder in folders:
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"跳过: {folder_path} 不存在")
            continue

        # 查找该文件夹下所有 mp4 文件
        mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

        if not mp4_files:
            print(f"跳过: {folder_path} 没有 mp4 文件")
            continue

        # 取第一个 mp4 文件
        mp4_name = mp4_files[0]
        mp4_path = os.path.join(folder_path, mp4_name)

        # 生成的 wav 文件名（保持和 mp4 同名）
        wav_name = mp4_name.replace('.mp4', '.wav')
        wav_path = os.path.join(folder_path, wav_name)

        if os.path.exists(wav_path):
            print(f"跳过: {wav_path} 已存在")
            continue

        cmd = ['ffmpeg', '-i', mp4_path, '-q:a', '0', '-map', 'a', wav_path, '-y']
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ {folder}: {mp4_name} → {wav_name}")
        except Exception as e:
            print(f"❌ {folder}: 提取失败 - {e}")

if __name__ == "__main__":
    extract_audio()