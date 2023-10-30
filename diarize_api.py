import argparse
import logging
import os
import re

import torch
import whisperx
from deepmultilingualpunctuation import PunctuationModel
from faster_whisper import WhisperModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pydub import AudioSegment

from helpers import *

mtypes = {"cpu": "int8", "cuda": "float16"}

DEVICE=0

def diarize(model_name: str, vocal_target, original_file_name: str, suppress_numerals=False):
    # Run on GPU with FP16
    whisper_model = WhisperModel(
        model_name, device=DEVICE, compute_type=mtypes[DEVICE]
    )

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    segments, info = whisper_model.transcribe(
        vocal_target,
        beam_size=5,
        word_timestamps=True,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )
    whisper_results = []
    for segment in segments:
        whisper_results.append(segment._asdict())
    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()

    if info.language in wav2vec2_langs:
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=DEVICE
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, DEVICE
        )
        word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])
        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})


    # convert audio to mono for NeMo combatibility
    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(DEVICE)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping


    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(f"{os.path.splitext(original_file_name)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(original_file_name)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
