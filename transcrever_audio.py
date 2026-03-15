import argparse
from pathlib import Path



def transcrever(audio_path: str, model_name: str = "small", language: str = "pt") -> str:
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcreve arquivos de áudio (ex.: .ogg) com OpenAI Whisper."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="audio01.ogg",
        help="Caminho do arquivo de áudio. Padrão: audio01.ogg",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Modelo Whisper (tiny, base, small, medium, large). Padrão: small",
    )
    parser.add_argument(
        "--language",
        default="pt",
        help="Idioma do áudio em formato ISO-639-1. Padrão: pt",
    )
    args = parser.parse_args()

    audio_file = Path(args.audio)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_file}")

    texto = transcrever(str(audio_file), model_name=args.model, language=args.language)
    print("Transcrição completa:")
    print(texto)


if __name__ == "__main__":
    main()
