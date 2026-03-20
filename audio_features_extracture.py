import librosa
import numpy as np
from pydub import AudioSegment

SAMPLE_RATE = 22050
N_MFCC = 20
MAX_MFCC_FRAMES = 174

def extract_features_pydub(file_path):
    """
    Витягує MFCCs, Delta, Delta-Delta та Спектральні фічі.
    """
    try:
        # 1. Завантаження та підготовка аудіо (ваш існуючий блок)
        audio_segment = AudioSegment.from_file(file_path)

        # Перетворення частоти дискретизації, моно, 16-bit
        if audio_segment.frame_rate != SAMPLE_RATE:
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        audio_array = np.array(audio_segment.get_array_of_samples())
        audio_float = audio_array.astype(np.float32) / (2**15) # Нормалізація

        # 2. Обчислення STFT (Short-Time Fourier Transform)
        # Це потрібно для всіх спектральних фіч
        stft = np.abs(librosa.stft(audio_float))

        # --- ВИЛУЧЕННЯ ФІЧ (LIBROSA) ---

        # A. MFCCs (базові статичні)
        mfccs = librosa.feature.mfcc(y=audio_float, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # B. Delta та Delta-Delta MFCCs (динамічні)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        # C. Спектральні фічі (тембральні)
        # Spectral Centroid (яскравість)
        centroid = librosa.feature.spectral_centroid(S=stft, sr=SAMPLE_RATE)
        # Spectral Flatness (шум vs тон)
        flatness = librosa.feature.spectral_flatness(S=stft)
        # Spectral Roll-off (розподіл енергії)
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=SAMPLE_RATE)

        # --- 3. ОБ'ЄДНАННЯ ТА УСЕРЕДНЕННЯ (Feature Aggregation) ---

        # 1. Створення єдиного масиву з усіх фреймів
        # Транспонуємо і усереднюємо всі матриці фіч

        # MFCCs та Delta MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0) # (N_MFCC,)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0) # (N_MFCC,)
        mfccs_delta2_mean = np.mean(mfccs_delta2.T, axis=0) # (N_MFCC,)

        # Спектральні фічі
        centroid_mean = np.mean(centroid.T, axis=0) # (1,)
        flatness_mean = np.mean(flatness.T, axis=0) # (1,)
        rolloff_mean = np.mean(rolloff.T, axis=0) # (1,)

        # 2. Конкатенація всіх усереднених фіч в один вектор
        # Використовуємо np.hstack для об'єднання всіх одновимірних масивів
        all_features = np.hstack([
            mfccs_mean,
            mfccs_delta_mean,
            mfccs_delta2_mean,
            centroid_mean,
            flatness_mean,
            rolloff_mean
        ])

        # Розмір вектора фіч: (N_MFCC * 3) + 3
        # Якщо N_MFCC=20: (20*3) + 3 = 63 фічі!

        return all_features

    except FileNotFoundError:
        print(f"❌ ФАЙЛ НЕ ЗНАЙДЕНО: {file_path}. Перевірте шлях.")
        return None
    except Exception as e:
        print(f"⚠️ ПОМИЛКА ДЕКОДУВАННЯ PYDUB/LIBROSA ({e.__class__.__name__}): {file_path}. {e}")
        return None