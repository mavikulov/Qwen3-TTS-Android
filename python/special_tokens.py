from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    END_OF_TEXT_ID: int = 151643
    IM_START_ID: int = 151644
    IM_END_ID: int = 151645
    AUDIO_START_ID: int = 151669
    AUDIO_END_ID: int = 151670
    TTS_PAD_ID: int = 151671
    TTS_BOS_ID: int = 151672
    TTS_EOD_ID: int = 151673
    TTS_BOS_SINGLE_ID: int = 151674
    AUDIO_PAD_ID: int = 151675
    ASSISTANT_TOKEN_ID: int = 77091
    NEWLINE_TOKEN_ID: int = 198
    NUM_EMBEDDING_FILES: int = 15


special_tokens = SpecialTokens()
